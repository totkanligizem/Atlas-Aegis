#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.benchmark import BenchConfig, run_benchmark
from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.config import load_yaml
from aerial_stack.legal_gate import check_dataset_usage


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument("--tier", choices=["fast", "candidate", "full"], required=True)
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--output-report", default="", help="Optional output report path override.")
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

    bench_cfg = load_yaml(args.config)
    pipe_cfg = load_yaml(args.pipeline_config)
    dataset_key = str(args.dataset_key or bench_cfg.get("dataset_key", "")).strip()

    tiers = pipe_cfg.get("loop_tiers", {})
    if args.tier not in tiers:
        raise SystemExit(f"tier not found in pipeline config: {args.tier}")
    conditions = list(tiers[args.tier].get("suites", []))
    if not conditions:
        raise SystemExit(f"empty suite for tier: {args.tier}")

    bench = BenchConfig(
        source=str(bench_cfg.get("source", "data/sample.mp4")),
        model=str(bench_cfg.get("model", "yolov8n.pt")),
        conf=float(bench_cfg.get("conf", 0.1)),
        iou=float(bench_cfg.get("iou", 0.7)),
        imgsz=int(bench_cfg.get("imgsz", 640)),
        device=(str(bench_cfg.get("device")) if str(bench_cfg.get("device", "")).strip() else None),
        max_frames=int(bench_cfg.get("max_frames", 60)),
        frame_stride=int(bench_cfg.get("frame_stride", 5)),
        db_path=str(bench_cfg.get("db_path", "logs/metrics.duckdb")),
        output_report=str(args.output_report or bench_cfg.get("output_report", "reports/benchmark_report.json")),
    )

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
                source=f"benchmark_{args.tier}",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    try:
        summary = run_benchmark(
            bench=bench,
            tier=args.tier,
            conditions=conditions,
            dry_run=bool(args.dry_run),
        )
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2

    summary["budget_guard"] = budget_guard
    summary["legal_gate"] = legal_gate
    Path(bench.output_report).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "run_id": summary.get("run_id"),
            "status": summary.get("status"),
            "tier": summary.get("tier"),
            "mode": summary.get("mode"),
            "num_conditions": summary.get("num_conditions"),
            "num_frames": summary.get("num_frames"),
            "legal_status": legal_gate.get("status", "unknown"),
            "budget_status": budget_guard.get("status", "unknown"),
            "output_report": bench.output_report,
            "db_path": bench.db_path,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
