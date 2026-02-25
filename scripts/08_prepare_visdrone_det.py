#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.legal_gate import check_dataset_usage
from aerial_stack.visdrone_det import prepare_visdrone_det_yolo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-root",
        default="data/raw/visdrone",
        help="VisDrone raw root that contains VisDrone2019-DET-train/val.",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/visdrone_det",
        help="Output root for YOLO-formatted dataset.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output folder.")
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (slower, larger).",
    )
    p.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Optional cap for train images (0 = all).",
    )
    p.add_argument(
        "--max-val-images",
        type=int,
        default=0,
        help="Optional cap for val images (0 = all).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-out",
        default="reports/visdrone_prepare_report.json",
        help="Where to write prep report JSON.",
    )
    p.add_argument("--legal-config", default="governance/legal_status.yaml")
    p.add_argument("--dataset-key", default="visdrone2019_det")
    p.add_argument("--usage-purpose", choices=["research", "commercial"], default="research")
    p.add_argument("--skip-legal-check", action="store_true")
    args = p.parse_args()

    legal_gate: dict[str, object] = {"status": "SKIPPED"}
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

    report = prepare_visdrone_det_yolo(
        raw_visdrone_root=args.raw_root,
        out_root=args.out_root,
        symlink_images=not args.copy_images,
        force=bool(args.force),
        max_train_images=int(args.max_train_images),
        max_val_images=int(args.max_val_images),
        seed=int(args.seed),
    )
    report["legal_gate"] = legal_gate

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
