#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.config import load_yaml
from aerial_stack.corruptions import apply_condition


def _image_files(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _safe_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _write_dataset_yaml(src_dataset_yaml: Path, out_root: Path) -> Path:
    cfg = load_yaml(src_dataset_yaml)
    names = cfg.get("names", [])
    if not isinstance(names, list):
        names = []
    dataset_out = out_root / "dataset.yaml"
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  - {str(n)}" for n in names)
    lines.append("")
    dataset_out.write_text("\n".join(lines), encoding="utf-8")
    return dataset_out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", default="data/processed/visdrone_det")
    p.add_argument("--out-root", default="data/processed/visdrone_det_corruptaware")
    p.add_argument("--conditions", default="s3_blur,s3_poisson")
    p.add_argument("--max-train-images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--copy-images", action="store_true", help="Copy files instead of symlinks for clean/val images.")
    p.add_argument("--exclude-clean", action="store_true", help="Do not include clean train samples in output train split.")
    p.add_argument("--report-out", default="reports/visdrone_corruptaware_prepare_report.json")
    args = p.parse_args()

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for corruption-aware dataset build.") from exc

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    src_dataset_yaml = src_root / "dataset.yaml"
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {src_root}")
    if not src_dataset_yaml.exists():
        raise FileNotFoundError(f"Source dataset yaml not found: {src_dataset_yaml}")

    conditions = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    if not conditions:
        raise ValueError("No corruption conditions provided.")

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"
    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_val.mkdir(parents=True, exist_ok=True)
    out_labels_train.mkdir(parents=True, exist_ok=True)
    out_labels_val.mkdir(parents=True, exist_ok=True)

    src_images_train = src_root / "images" / "train"
    src_images_val = src_root / "images" / "val"
    src_labels_train = src_root / "labels" / "train"
    src_labels_val = src_root / "labels" / "val"
    for req in [src_images_train, src_images_val, src_labels_train, src_labels_val]:
        if not req.exists():
            raise FileNotFoundError(f"Missing source path: {req}")

    link_or_copy = _safe_copy if args.copy_images else _safe_symlink

    # Validation split stays clean.
    val_images = _image_files(src_images_val)
    val_kept = 0
    for src_img in val_images:
        dst_img = out_images_val / src_img.name
        link_or_copy(src_img, dst_img)
        src_lbl = src_labels_val / f"{src_img.stem}.txt"
        dst_lbl = out_labels_val / f"{src_img.stem}.txt"
        if src_lbl.exists():
            _safe_copy(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        val_kept += 1

    train_images = _image_files(src_images_train)
    rng = random.Random(int(args.seed))
    max_train = int(args.max_train_images)
    if max_train > 0 and len(train_images) > max_train:
        train_images = sorted(rng.sample(train_images, max_train))

    train_seen = 0
    train_clean_kept = 0
    train_corrupt_kept = 0
    missing_labels = 0
    read_failures = 0

    for src_img in train_images:
        src_lbl = src_labels_train / f"{src_img.stem}.txt"
        if not src_lbl.exists():
            missing_labels += 1
            continue

        if not args.exclude_clean:
            dst_img_clean = out_images_train / src_img.name
            dst_lbl_clean = out_labels_train / f"{src_img.stem}.txt"
            link_or_copy(src_img, dst_img_clean)
            _safe_copy(src_lbl, dst_lbl_clean)
            train_clean_kept += 1

        img = cv2.imread(str(src_img))
        if img is None:
            read_failures += 1
            continue

        for cond in conditions:
            aug_img = apply_condition(img, cond, seed_key=src_img.stem)
            out_img_name = f"{src_img.stem}__{cond}{src_img.suffix.lower()}"
            out_lbl_name = f"{src_img.stem}__{cond}.txt"
            dst_img_aug = out_images_train / out_img_name
            dst_lbl_aug = out_labels_train / out_lbl_name
            ok = cv2.imwrite(str(dst_img_aug), aug_img)
            if not ok:
                continue
            _safe_copy(src_lbl, dst_lbl_aug)
            train_corrupt_kept += 1

        train_seen += 1

    dataset_out = _write_dataset_yaml(src_dataset_yaml=src_dataset_yaml, out_root=out_root)

    report = {
        "status": "SUCCESS",
        "src_root": str(src_root),
        "out_root": str(out_root),
        "conditions": conditions,
        "max_train_images": max_train,
        "seed": int(args.seed),
        "exclude_clean": bool(args.exclude_clean),
        "copy_images": bool(args.copy_images),
        "counts": {
            "train_seen": train_seen,
            "train_clean_kept": train_clean_kept,
            "train_corrupt_kept": train_corrupt_kept,
            "train_total_out": train_clean_kept + train_corrupt_kept,
            "val_kept": val_kept,
            "missing_labels": missing_labels,
            "read_failures": read_failures,
        },
        "dataset_yaml": str(dataset_out),
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
