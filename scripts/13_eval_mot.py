#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.legal_gate import check_dataset_usage
from aerial_stack.mot_eval import (
    MOTDetection,
    evaluate_event_lead_time,
    evaluate_mot,
    load_events,
    load_mot_txt,
)


def _parse_hota_thresholds(text: str) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return []
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Invalid HOTA IoU threshold (expected [0,1]): {v}")
        out.append(v)
    return out


def _synthetic_gt_pred() -> tuple[dict[int, list[MOTDetection]], dict[int, list[MOTDetection]]]:
    gt = {
        1: [MOTDetection(1, 1, (10, 10, 50, 50)), MOTDetection(1, 2, (100, 100, 150, 150))],
        2: [MOTDetection(2, 1, (12, 10, 52, 50)), MOTDetection(2, 2, (103, 100, 153, 150))],
        3: [MOTDetection(3, 1, (15, 10, 55, 50)), MOTDetection(3, 2, (106, 100, 156, 150))],
    }
    pred = {
        1: [MOTDetection(1, 11, (10, 10, 50, 50)), MOTDetection(1, 22, (100, 100, 150, 150))],
        2: [MOTDetection(2, 11, (12, 10, 52, 50)), MOTDetection(2, 22, (103, 100, 153, 150))],
        3: [MOTDetection(3, 99, (15, 10, 55, 50)), MOTDetection(3, 22, (106, 100, 156, 150))],
    }
    return gt, pred


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-mot", default="", help="Ground truth MOT file path")
    p.add_argument("--pred-mot", default="", help="Prediction MOT file path")
    p.add_argument("--gt-events", default="", help="Optional GT events JSON path")
    p.add_argument("--pred-events", default="reports/track_risk_report.json", help="Optional predicted events JSON path")
    p.add_argument(
        "--require-gt-events",
        action="store_true",
        help="Fail if --gt-events is provided but file is missing/empty.",
    )
    p.add_argument("--iou-match-threshold", type=float, default=0.5)
    p.add_argument(
        "--include-hota",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HOTA/AssA/DetA computation (heavier than basic MOTA metrics).",
    )
    p.add_argument(
        "--hota-thresholds",
        default="",
        help="Optional comma-separated IoU thresholds for HOTA (default: 0.05..0.95 step 0.05).",
    )
    p.add_argument("--output-report", default="reports/mot_eval_report.json")
    p.add_argument("--dry-run", action="store_true")
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

    legal_gate: dict[str, object] = {"status": "SKIPPED"}
    dataset_key = str(args.dataset_key).strip()
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
                source="eval_mot",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    report: dict[str, object] = {
        "status": "FAILED",
        "config": {
            "gt_mot": args.gt_mot,
            "pred_mot": args.pred_mot,
            "gt_events": args.gt_events,
            "pred_events": args.pred_events,
            "iou_match_threshold": float(args.iou_match_threshold),
            "include_hota": bool(args.include_hota),
            "hota_thresholds": _parse_hota_thresholds(args.hota_thresholds),
            "dry_run": bool(args.dry_run),
            "dataset_key": dataset_key,
        },
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
        "warnings": [],
    }

    if args.dry_run:
        hota_thresholds = _parse_hota_thresholds(args.hota_thresholds)
        gt_by_frame, pred_by_frame = _synthetic_gt_pred()
        mot_metrics = evaluate_mot(
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            iou_threshold=float(args.iou_match_threshold),
            include_hota=bool(args.include_hota),
            hota_thresholds=hota_thresholds,
        )
        lead_metrics = evaluate_event_lead_time(
            gt_events=[{"track_id": 1, "event_start_frame": 10}, {"track_id": 2, "event_start_frame": 20}],
            pred_events=[{"track_id": 1, "alarm_frame": 7}, {"track_id": 2, "alarm_frame": 18}],
        )
        report.update(
            {
                "status": "SUCCESS",
                "mot_metrics": mot_metrics,
                "event_lead_time": lead_metrics,
            }
        )
    else:
        if not args.gt_mot or not args.pred_mot:
            raise ValueError("--gt-mot and --pred-mot are required unless --dry-run is set.")

        hota_thresholds = _parse_hota_thresholds(args.hota_thresholds)
        gt_by_frame = load_mot_txt(args.gt_mot, is_gt=True)
        pred_by_frame = load_mot_txt(args.pred_mot, is_gt=False)
        mot_metrics = evaluate_mot(
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            iou_threshold=float(args.iou_match_threshold),
            include_hota=bool(args.include_hota),
            hota_thresholds=hota_thresholds,
        )

        lead_metrics: dict[str, float] = {}
        if args.gt_events:
            gt_events_path = Path(args.gt_events)
            if not gt_events_path.exists():
                msg = f"GT events file not found: {args.gt_events}"
                if args.require_gt_events:
                    raise FileNotFoundError(msg)
                report["warnings"].append(msg)  # type: ignore[index]
            gt_events = load_events(args.gt_events)
            pred_events = load_events(args.pred_events)
            if args.require_gt_events and not gt_events:
                raise ValueError(f"GT events loaded empty: {args.gt_events}")
            if not gt_events:
                report["warnings"].append(f"GT events empty or unreadable: {args.gt_events}")  # type: ignore[index]
            if not pred_events:
                report["warnings"].append(f"Pred events empty or unreadable: {args.pred_events}")  # type: ignore[index]
            lead_metrics = evaluate_event_lead_time(gt_events=gt_events, pred_events=pred_events)

        report.update(
            {
                "status": "SUCCESS",
                "mot_metrics": mot_metrics,
                "event_lead_time": lead_metrics,
                "counts": {
                    "gt_frames": len(gt_by_frame),
                    "pred_frames": len(pred_by_frame),
                },
            }
        )

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
