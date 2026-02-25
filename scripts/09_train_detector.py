#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.config import load_yaml
from aerial_stack.legal_gate import check_dataset_usage


def _as_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    return bool(v)


def _as_int(cfg: dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    return int(v)


def _as_float(cfg: dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    return float(v)


def _collect_metrics(train_result: Any, trainer_obj: Any) -> dict[str, float]:
    out: dict[str, float] = {}

    if hasattr(train_result, "results_dict"):
        rd = getattr(train_result, "results_dict", {})
        if isinstance(rd, dict):
            for k, v in rd.items():
                if isinstance(v, (int, float)):
                    out[str(k)] = float(v)

    if hasattr(trainer_obj, "metrics"):
        m = getattr(trainer_obj, "metrics", {})
        if isinstance(m, dict):
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    out[str(k)] = float(v)

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detector_fast.yaml")
    p.add_argument("--data", default="data/processed/visdrone_det/dataset.yaml")
    p.add_argument("--model", default="", help="Override model weights path.")
    p.add_argument("--name", default="", help="Override run name.")
    p.add_argument("--epochs", type=int, default=-1, help="Override epochs.")
    p.add_argument("--device", default="", help="Override device, e.g. mps/cpu/0.")
    p.add_argument("--dry-run", action="store_true", help="Only print resolved train args.")
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
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    model_name = str(args.model or cfg.get("model", "yolov8n.pt"))
    run_name = str(args.name or cfg.get("run_name", "visdrone_baseline"))
    epochs = int(args.epochs if args.epochs > 0 else _as_int(cfg, "epochs", 30))

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": _as_int(cfg, "imgsz", 640),
        "batch": _as_int(cfg, "batch", 8),
        "epochs": epochs,
        "workers": _as_int(cfg, "workers", 4),
        "patience": _as_int(cfg, "patience", 20),
        "project": str(cfg.get("project", "artifacts/detect")),
        "name": run_name,
        "cache": _as_bool(cfg, "cache", False),
        "pretrained": _as_bool(cfg, "pretrained", True),
        "optimizer": str(cfg.get("optimizer", "auto")),
        "seed": _as_int(cfg, "seed", 42),
        "amp": _as_bool(cfg, "amp", True),
    }
    if "exist_ok" in cfg:
        train_args["exist_ok"] = _as_bool(cfg, "exist_ok", False)

    if "lr0" in cfg:
        train_args["lr0"] = _as_float(cfg, "lr0", 0.01)
    if "lrf" in cfg:
        train_args["lrf"] = _as_float(cfg, "lrf", 0.01)
    if "weight_decay" in cfg:
        train_args["weight_decay"] = _as_float(cfg, "weight_decay", 0.0005)
    if "warmup_epochs" in cfg:
        train_args["warmup_epochs"] = _as_float(cfg, "warmup_epochs", 3.0)
    if "close_mosaic" in cfg:
        train_args["close_mosaic"] = _as_int(cfg, "close_mosaic", 10)

    device = (args.device or str(cfg.get("device", ""))).strip()
    if device:
        train_args["device"] = device

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
                source="train_detector",
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
        print(
            json.dumps(
                {
                    "model": model_name,
                    "train_args": train_args,
                    "legal_gate": legal_gate,
                    "budget_guard": budget_guard,
                },
                indent=2,
            )
        )
        return 0

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required for detector training.") from exc

    model = YOLO(model_name)
    train_result = model.train(**train_args)
    trainer = getattr(model, "trainer", None)

    best_path = ""
    last_path = ""
    save_dir = ""
    if trainer is not None:
        best_obj = getattr(trainer, "best", None)
        last_obj = getattr(trainer, "last", None)
        save_dir_obj = getattr(trainer, "save_dir", None)
        best_path = str(best_obj) if best_obj is not None else ""
        last_path = str(last_obj) if last_obj is not None else ""
        save_dir = str(save_dir_obj) if save_dir_obj is not None else ""

    metrics = _collect_metrics(train_result=train_result, trainer_obj=trainer)

    report = {
        "config": str(args.config),
        "data": str(data_yaml),
        "model": model_name,
        "run_name": run_name,
        "train_args": train_args,
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
        "metrics": metrics,
        "artifacts": {
            "save_dir": save_dir,
            "best": best_path,
            "last": last_path,
        },
    }
    report_out = Path(str(cfg.get("output_report", "reports/train_detector_report.json")))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
