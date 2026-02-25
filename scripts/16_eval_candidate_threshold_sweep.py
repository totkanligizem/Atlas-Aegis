#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.config import load_yaml
from aerial_stack.visdrone_det import SmallObjectRule, evaluate_detector_conditions, evaluate_gates


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _as_float(d: dict[str, Any], key: str, default: float) -> float:
    return float(d.get(key, default))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detector_eval_candidate_highrecall_corruptaware.yaml")
    p.add_argument("--model", default="", help="Override model path.")
    p.add_argument("--conf-values", default="0.0005,0.001,0.0015")
    p.add_argument("--iou-values", default="0.55,0.6")
    p.add_argument("--output-report", default="reports/detector_eval_candidate_threshold_sweep_report.json")
    p.add_argument("--max-images", type=int, default=-1, help="Override max images.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    model_path = str(args.model or cfg.get("model", ""))
    if not model_path:
        raise RuntimeError("Model path is required.")

    model_candidate = Path(model_path)
    if ("/" in model_path or "\\" in model_path or model_candidate.exists()) and not model_candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    data_root = str(cfg.get("data_root", "data/processed/visdrone_det"))
    split = str(cfg.get("split", "val"))
    conditions = list(cfg.get("conditions", []))
    if not conditions:
        raise RuntimeError("No conditions configured.")
    imgsz = int(cfg.get("imgsz", 960))
    device = str(cfg.get("device", "")).strip() or None
    max_images = int(cfg.get("max_images", 0))
    if args.max_images >= 0:
        max_images = int(args.max_images)
    iou_match_threshold = _as_float(cfg, "iou_match_threshold", 0.5)

    small_cfg = dict(cfg.get("small_object", {}))
    small_rule = SmallObjectRule(
        area_px2_lt=_as_float(small_cfg, "area_px2_lt", 32.0 * 32.0),
        min_side_px_lt=_as_float(small_cfg, "min_side_px_lt", 16.0),
    )
    gates_cfg: dict[str, float] = {
        str(k): float(v) for k, v in dict(cfg.get("gates", {})).items()
    }

    conf_values = _parse_float_list(args.conf_values)
    iou_values = _parse_float_list(args.iou_values)
    total = len(conf_values) * len(iou_values)

    rows: list[dict[str, Any]] = []
    idx = 0
    for conf in conf_values:
        for iou in iou_values:
            idx += 1
            print(f"[sweep] {idx}/{total} conf={conf:.6f} iou={iou:.3f}")
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

            clean = eval_out["metrics_by_condition"].get("clean", {})
            gate_map = {str(c.get("name", "")): c for c in gates.get("checks", [])}
            s3_drop = float(gate_map.get("s3_recall_small_drop_max", {}).get("value", 0.0))
            s3_worst_name = ""
            s3_worst_drop = -1.0
            for name, val in eval_out.get("robustness", {}).items():
                if not name.endswith("_recall_small_rel_drop"):
                    continue
                if "_s3_" not in name and not name.startswith("s3_"):
                    continue
                v = float(val)
                if v > s3_worst_drop:
                    s3_worst_drop = v
                    s3_worst_name = name.replace("_recall_small_rel_drop", "")

            row = {
                "conf": conf,
                "iou": iou,
                "gate_pass": bool(gates.get("pass", False)),
                "clean_recall": float(clean.get("recall", 0.0)),
                "clean_recall_small": float(clean.get("recall_small", 0.0)),
                "s3_recall_small_drop_max": s3_drop,
                "s3_worst_condition": s3_worst_name,
                "s3_worst_rel_drop": s3_worst_drop,
            }
            rows.append(row)
            print(
                f"[sweep] gate_pass={row['gate_pass']} "
                f"clean_recall_small={row['clean_recall_small']:.4f} "
                f"s3_drop={row['s3_recall_small_drop_max']:.4f}"
            )

    rows.sort(
        key=lambda r: (
            0 if r["gate_pass"] else 1,
            float(r["s3_recall_small_drop_max"]),
            -float(r["clean_recall_small"]),
        )
    )

    out = {
        "status": "SUCCESS",
        "config": {
            "base_config": str(args.config),
            "model": model_path,
            "data_root": data_root,
            "split": split,
            "conditions": conditions,
            "imgsz": imgsz,
            "device": device or "",
            "max_images": max_images,
            "iou_match_threshold": iou_match_threshold,
            "gates": gates_cfg,
        },
        "grid": {
            "conf_values": conf_values,
            "iou_values": iou_values,
        },
        "results": rows,
        "best": rows[0] if rows else {},
    }

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["best"], indent=2))
    print(f"wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
