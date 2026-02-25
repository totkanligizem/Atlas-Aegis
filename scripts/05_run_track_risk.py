#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.config import load_yaml
from aerial_stack.legal_gate import check_dataset_usage
from aerial_stack.risk import BandThresholds, RiskWeights
from aerial_stack.track_risk import ROI, run_track_risk_dry, run_track_risk_ultralytics


def _weights_from_cfg(cfg: dict) -> RiskWeights:
    w = cfg.get("weights", {})
    return RiskWeights(
        roi_dwell=float(w.get("roi_dwell", 1.4)),
        approach=float(w.get("approach", 1.0)),
        duration=float(w.get("duration", 0.8)),
        conf_mean=float(w.get("conf_mean", 0.6)),
        conf_stability=float(w.get("conf_stability", 0.5)),
        occlusion=float(w.get("occlusion", 0.3)),
        bias=float(cfg.get("bias", -1.8)),
    )


def _bands_from_cfg(cfg: dict) -> BandThresholds:
    b = cfg.get("bands", {})
    return BandThresholds(
        green_max=float(b.get("green_max", 39.999)),
        yellow_max=float(b.get("yellow_max", 69.999)),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to inference config")
    p.add_argument("--dry-run", action="store_true", help="Run synthetic tracks without ultralytics")
    p.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Frame count used only in dry-run mode",
    )
    p.add_argument("--legal-config", default="governance/legal_status.yaml")
    p.add_argument("--dataset-key", default="")
    p.add_argument("--usage-purpose", choices=["research", "commercial"], default="research")
    p.add_argument("--skip-legal-check", action="store_true")
    p.add_argument("--budget-config", default="configs/budget.yaml")
    p.add_argument("--budget-ledger", default="logs/budget_ledger.json")
    p.add_argument("--budget-events", default="logs/budget_events.jsonl")
    p.add_argument("--budget-spend-usd", type=float, default=0.0)
    p.add_argument("--budget-is-api-call", action="store_true")
    p.add_argument("--skip-budget-check", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    risk_cfg = load_yaml(cfg.get("risk_config", "configs/risk.yaml"))

    roi_cfg = cfg.get("roi", {})
    roi = ROI(
        x1=float(roi_cfg.get("x1", 0.2)),
        y1=float(roi_cfg.get("y1", 0.2)),
        x2=float(roi_cfg.get("x2", 0.8)),
        y2=float(roi_cfg.get("y2", 0.8)),
    )

    weights = _weights_from_cfg(risk_cfg)
    bands = _bands_from_cfg(risk_cfg)
    min_frames_in_roi = int(risk_cfg.get("event", {}).get("min_frames_in_roi", 10))
    track_ttl_frames = int(cfg.get("track_ttl_frames", 60))
    out_jsonl = cfg.get("output_jsonl")
    dataset_key = str(args.dataset_key or cfg.get("dataset_key", "")).strip()

    legal_gate: dict[str, object] = {"status": "SKIPPED"}
    if dataset_key and not args.skip_legal_check:
        try:
            legal_gate = check_dataset_usage(
                legal_config_path=args.legal_config,
                dataset_key=dataset_key,
                usage_purpose=args.usage_purpose,
            )
        except Exception as exc:
            print(f"error: legal gate failed: {exc}")
            return 2

    budget_guard: dict[str, object] = {"status": "SKIPPED"}
    if not args.skip_budget_check:
        try:
            budget_guard = apply_budget_guard(
                budget_config_path=args.budget_config,
                budget_ledger_path=args.budget_ledger,
                budget_events_path=args.budget_events,
                source="track_risk",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    if args.dry_run:
        summary = run_track_risk_dry(
            num_frames=args.frames,
            roi=roi,
            min_frames_in_roi=min_frames_in_roi,
            track_ttl_frames=track_ttl_frames,
            weights=weights,
            bands=bands,
            out_jsonl=out_jsonl,
        )
    else:
        try:
            summary = run_track_risk_ultralytics(
                source=str(cfg.get("source", "")),
                model_path=str(cfg.get("model", "yolov8n.pt")),
                tracker_path=str(cfg.get("tracker", "configs/trackers/bytetrack.yaml")),
                conf=float(cfg.get("conf", 0.1)),
                iou=float(cfg.get("iou", 0.7)),
                imgsz=int(cfg.get("imgsz", 640)),
                device=(str(cfg.get("device")) if str(cfg.get("device", "")).strip() else None),
                max_frames=int(cfg.get("max_frames", 0)),
                roi=roi,
                min_frames_in_roi=min_frames_in_roi,
                track_ttl_frames=track_ttl_frames,
                weights=weights,
                bands=bands,
                out_jsonl=out_jsonl,
            )
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 2

    summary["legal_gate"] = legal_gate
    summary["budget_guard"] = budget_guard

    report_path = Path(cfg.get("output_report", "reports/track_risk_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote report: {report_path}")
    if out_jsonl:
        print(f"wrote frame logs: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
