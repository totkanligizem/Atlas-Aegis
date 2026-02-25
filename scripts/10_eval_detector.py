#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.config import load_yaml
from aerial_stack.legal_gate import check_dataset_usage
from aerial_stack.metrics_store import end_run, init_db, log_metric, start_run
from aerial_stack.visdrone_det import SmallObjectRule, evaluate_detector_conditions, evaluate_gates


def _norm_device(raw: str) -> str | None:
    v = str(raw).strip()
    return v if v else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detector_eval.yaml")
    p.add_argument("--model", default="", help="Override model path.")
    p.add_argument("--data-root", default="", help="Override prepared YOLO dataset root.")
    p.add_argument("--split", default="", help="Override split (train/val).")
    p.add_argument("--max-images", type=int, default=-1, help="Override max_images.")
    p.add_argument("--output-report", default="", help="Override output report path.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--legal-config", default="governance/legal_status.yaml")
    p.add_argument("--dataset-key", default="visdrone2019_det")
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
    model_path = str(args.model or cfg.get("model", "yolov8n.pt"))
    data_root = str(args.data_root or cfg.get("data_root", "data/processed/visdrone_det"))
    split = str(args.split or cfg.get("split", "val"))
    conditions = list(cfg.get("conditions", ["clean", "s3_blur", "s3_low_light", "s3_jpeg", "s3_fog"]))
    conf = float(cfg.get("conf", 0.001))
    iou = float(cfg.get("iou", 0.6))
    imgsz = int(cfg.get("imgsz", 640))
    device = _norm_device(str(cfg.get("device", "")))
    max_images = int(cfg.get("max_images", 0))
    if args.max_images >= 0:
        max_images = int(args.max_images)
    iou_match_threshold = float(cfg.get("iou_match_threshold", 0.5))

    small_cfg = cfg.get("small_object", {})
    small_rule = SmallObjectRule(
        area_px2_lt=float(small_cfg.get("area_px2_lt", 32.0 * 32.0)),
        min_side_px_lt=float(small_cfg.get("min_side_px_lt", 16.0)),
    )
    gates_cfg: dict[str, float] = {
        str(k): float(v) for k, v in dict(cfg.get("gates", {})).items()
    }

    legal_gate: dict[str, Any] = {"status": "SKIPPED"}
    if not args.skip_legal_check:
        try:
            legal_gate = check_dataset_usage(
                legal_config_path=args.legal_config,
                dataset_key=args.dataset_key,
                usage_purpose=args.usage_purpose,
            )
        except Exception as exc:
            print(f"error: legal gate failed: {exc}")
            return 2

    budget_guard: dict[str, Any] = {"status": "SKIPPED"}
    if not args.skip_budget_check:
        try:
            budget_guard = apply_budget_guard(
                budget_config_path=args.budget_config,
                budget_ledger_path=args.budget_ledger,
                budget_events_path=args.budget_events,
                source="eval_detector",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    resolved = {
        "model": model_path,
        "data_root": data_root,
        "split": split,
        "conditions": conditions,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device or "",
        "max_images": max_images,
        "iou_match_threshold": iou_match_threshold,
        "small_object": {
            "area_px2_lt": small_rule.area_px2_lt,
            "min_side_px_lt": small_rule.min_side_px_lt,
        },
        "gates": gates_cfg,
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    model_candidate = Path(model_path)
    looks_like_local_path = ("/" in model_path) or ("\\" in model_path) or model_candidate.exists()
    if looks_like_local_path and not model_candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    run_id = f"det_eval_{uuid.uuid4().hex[:10]}"
    db_path = str(cfg.get("db_path", "logs/metrics.duckdb"))
    init_db(db_path)
    start_run(
        db_path=db_path,
        run_id=run_id,
        pipeline="detector_eval",
        tier=str(cfg.get("tier", "fast")),
        mode="ultralytics",
        config_obj=resolved,
    )

    summary: dict[str, Any] = {}
    status = "FAILED"
    exit_code = 1
    print(
        f"[detector_eval] run_id={run_id} tier={cfg.get('tier', 'fast')} "
        f"split={split} conditions={len(conditions)} max_images={max_images if max_images > 0 else 'all'}"
    )
    try:
        eval_out = evaluate_detector_conditions(
            model_path=model_path,
            data_root=data_root,
            split=split,
            conditions=conditions,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_images=max_images,
            iou_match_threshold=iou_match_threshold,
            small_rule=small_rule,
        )
        gates = evaluate_gates(eval_out["metrics_by_condition"], gates_cfg)
        status = "SUCCESS"

        for condition, metrics in eval_out["metrics_by_condition"].items():
            for name, val in metrics.items():
                log_metric(db_path, run_id, condition, name, float(val))
        for name, val in eval_out["robustness"].items():
            log_metric(db_path, run_id, "robustness", name, float(val))
        for chk in gates.get("checks", []):
            c_name = str(chk.get("name", "gate"))
            log_metric(db_path, run_id, "gates", f"{c_name}_value", float(chk.get("value", 0.0)))
            log_metric(
                db_path,
                run_id,
                "gates",
                f"{c_name}_passed",
                1.0 if bool(chk.get("passed", False)) else 0.0,
            )
        log_metric(db_path, run_id, "gates", "overall_pass", 1.0 if gates.get("pass", False) else 0.0)

        summary = {
            "run_id": run_id,
            "status": status,
            "config": resolved,
            "evaluation": eval_out,
            "gates": gates,
        }
        exit_code = 0
    except BaseException as exc:
        summary = {
            "run_id": run_id,
            "status": "FAILED",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "config": resolved,
        }
        status = "FAILED"
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1

    end_run(db_path=db_path, run_id=run_id, status=status, summary_obj=summary)

    report_out = Path(str(args.output_report or cfg.get("output_report", "reports/detector_eval_report.json")))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote report: {report_out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
