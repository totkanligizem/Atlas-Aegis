from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corruptions import apply_condition


VISDRONE_CATEGORY_TO_NAME: dict[int, str] = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


@dataclass(frozen=True)
class SmallObjectRule:
    area_px2_lt: float = 32.0 * 32.0
    min_side_px_lt: float = 16.0


def _parse_visdrone_line(line: str) -> tuple[float, float, float, float, int] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None
    try:
        left = float(parts[0])
        top = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        category = int(float(parts[5]))
    except ValueError:
        return None
    return left, top, width, height, category


def _clip_bbox_xywh(
    left: float,
    top: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(img_w - 1), left))
    y1 = max(0.0, min(float(img_h - 1), top))
    x2 = max(x1 + 1.0, min(float(img_w), left + width))
    y2 = max(y1 + 1.0, min(float(img_h), top + height))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    return x1, y1, x2, y2


def _xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def prepare_visdrone_det_yolo(
    raw_visdrone_root: str | Path,
    out_root: str | Path,
    *,
    include_categories: tuple[int, ...] = tuple(range(1, 11)),
    symlink_images: bool = True,
    force: bool = False,
    max_train_images: int = 0,
    max_val_images: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for dataset preparation.") from exc

    raw_root = Path(raw_visdrone_root)
    out_root_path = Path(out_root)
    include_set = set(include_categories)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw VisDrone root not found: {raw_root}")

    if out_root_path.exists() and any(out_root_path.iterdir()) and not force:
        raise FileExistsError(
            f"Output already exists and is not empty: {out_root_path}. Use --force to overwrite."
        )

    images_root = out_root_path / "images"
    labels_root = out_root_path / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    split_limits = {"train": max_train_images, "val": max_val_images}
    rng = random.Random(seed)

    report: dict[str, Any] = {
        "raw_root": str(raw_root),
        "output_root": str(out_root_path),
        "include_categories": sorted(include_set),
        "symlink_images": symlink_images,
        "seed": seed,
        "splits": {},
    }

    for split in ("train", "val"):
        split_src = raw_root / f"VisDrone2019-DET-{split}"
        src_images = split_src / "images"
        src_annotations = split_src / "annotations"
        if not src_images.exists() or not src_annotations.exists():
            raise FileNotFoundError(f"Missing VisDrone split directories: {split_src}")

        split_images_out = images_root / split
        split_labels_out = labels_root / split
        split_images_out.mkdir(parents=True, exist_ok=True)
        split_labels_out.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p
            for p in src_images.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        limit = split_limits[split]
        if limit > 0 and len(image_files) > limit:
            image_files = sorted(rng.sample(image_files, limit))

        images_processed = 0
        missing_annotations = 0
        read_failures = 0
        boxes_total = 0
        boxes_kept = 0
        boxes_ignored = 0
        category_kept: dict[str, int] = {VISDRONE_CATEGORY_TO_NAME[k]: 0 for k in include_set if k in VISDRONE_CATEGORY_TO_NAME}

        for src_img in image_files:
            ann_path = src_annotations / f"{src_img.stem}.txt"
            if not ann_path.exists():
                missing_annotations += 1
                continue

            img = cv2.imread(str(src_img))
            if img is None:
                read_failures += 1
                continue
            img_h, img_w = img.shape[:2]
            if img_h <= 0 or img_w <= 0:
                read_failures += 1
                continue

            dst_img = split_images_out / src_img.name
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if symlink_images:
                dst_img.symlink_to(src_img.resolve())
            else:
                dst_img.write_bytes(src_img.read_bytes())

            yolo_lines: list[str] = []
            raw = ann_path.read_text(encoding="utf-8", errors="ignore")
            for ln in raw.splitlines():
                if not ln.strip():
                    continue
                parsed = _parse_visdrone_line(ln)
                if parsed is None:
                    continue
                left, top, width, height, category = parsed
                boxes_total += 1
                if category not in include_set:
                    boxes_ignored += 1
                    continue
                clipped = _clip_bbox_xywh(left, top, width, height, img_w=img_w, img_h=img_h)
                if clipped is None:
                    boxes_ignored += 1
                    continue
                x1, y1, x2, y2 = clipped
                cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, img_w=img_w, img_h=img_h)
                cls_id = category - 1
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                boxes_kept += 1
                if category in VISDRONE_CATEGORY_TO_NAME:
                    category_kept[VISDRONE_CATEGORY_TO_NAME[category]] += 1

            (split_labels_out / f"{src_img.stem}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )
            images_processed += 1

        report["splits"][split] = {
            "images_processed": images_processed,
            "missing_annotations": missing_annotations,
            "read_failures": read_failures,
            "boxes_total": boxes_total,
            "boxes_kept": boxes_kept,
            "boxes_ignored": boxes_ignored,
            "class_counts": category_kept,
        }

    dataset_yaml = out_root_path / "dataset.yaml"
    names = [VISDRONE_CATEGORY_TO_NAME[i] for i in sorted(include_set) if i in VISDRONE_CATEGORY_TO_NAME]
    yaml_text = "\n".join(
        [
            f"path: {out_root_path.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            "names:",
            *[f"  - {n}" for n in names],
            "",
        ]
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")
    report["dataset_yaml"] = str(dataset_yaml)
    return report


def _compute_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _load_yolo_gt(
    label_file: Path,
    *,
    img_w: int,
    img_h: int,
    small_rule: SmallObjectRule,
) -> list[dict[str, Any]]:
    if not label_file.exists():
        return []
    out: list[dict[str, Any]] = []
    raw = label_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return out
    for ln in raw.splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
        except ValueError:
            continue
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        area = max(0.0, bw) * max(0.0, bh)
        is_small = area < small_rule.area_px2_lt or min(bw, bh) < small_rule.min_side_px_lt
        out.append(
            {
                "cls": cls_id,
                "bbox": (x1, y1, x2, y2),
                "small": is_small,
            }
        )
    return out


def _extract_predictions(result: Any) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return preds
    if len(boxes) == 0:
        return preds
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        preds.append(
            {
                "cls": int(clss[i]),
                "conf": float(confs[i]),
                "bbox": (x1, y1, x2, y2),
            }
        )
    preds.sort(key=lambda x: x["conf"], reverse=True)
    return preds


def _match_counts(
    gt: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    *,
    iou_match_threshold: float,
    class_aware: bool = True,
) -> dict[str, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    tp_small = 0

    for pred in preds:
        best_iou = 0.0
        best_gt_idx = -1
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            if class_aware and pred["cls"] != g["cls"]:
                continue
            iou = _compute_iou_xyxy(pred["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_gt_idx >= 0 and best_iou >= iou_match_threshold:
            matched_gt.add(best_gt_idx)
            tp += 1
            if bool(gt[best_gt_idx]["small"]):
                tp_small += 1
        else:
            fp += 1

    fn = len(gt) - len(matched_gt)
    small_total = sum(1 for g in gt if bool(g["small"]))
    fn_small = small_total - tp_small
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_small": tp_small,
        "fn_small": fn_small,
        "gt_count": len(gt),
        "pred_count": len(preds),
    }


def _safe_div(n: float, d: float) -> float:
    if d <= 0.0:
        return 0.0
    return n / d


def evaluate_detector_conditions(
    *,
    model_path: str | Path,
    data_root: str | Path,
    split: str,
    conditions: list[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_images: int,
    iou_match_threshold: float,
    small_rule: SmallObjectRule,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for detector evaluation.") from exc
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required for detector evaluation.") from exc

    root = Path(data_root)
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Split not found in prepared dataset root: images={images_dir}, labels={labels_dir}"
        )

    model = YOLO(str(model_path))

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_images > 0:
        image_files = image_files[:max_images]
    if not image_files:
        raise RuntimeError(f"No images found for split={split} at {images_dir}")

    totals: dict[str, dict[str, int]] = {}
    for c in conditions:
        totals[c] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tp_small": 0,
            "fn_small": 0,
            "gt_count": 0,
            "pred_count": 0,
        }

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        gt = _load_yolo_gt(
            labels_dir / f"{img_path.stem}.txt",
            img_w=img_w,
            img_h=img_h,
            small_rule=small_rule,
        )

        for condition in conditions:
            if condition == "clean":
                eval_img = img
            else:
                eval_img = apply_condition(img, condition=condition, seed_key=img_path.name)

            kwargs: dict[str, Any] = {
                "source": eval_img,
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "verbose": False,
            }
            if device:
                kwargs["device"] = device
            res = model.predict(**kwargs)
            preds = _extract_predictions(res[0]) if res else []
            counts = _match_counts(
                gt=gt,
                preds=preds,
                iou_match_threshold=iou_match_threshold,
                class_aware=True,
            )
            for k, v in counts.items():
                totals[condition][k] += int(v)

    metrics_by_condition: dict[str, dict[str, float]] = {}
    for condition, c in totals.items():
        tp = float(c["tp"])
        fp = float(c["fp"])
        fn = float(c["fn"])
        tp_small = float(c["tp_small"])
        fn_small = float(c["fn_small"])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        recall_small = _safe_div(tp_small, tp_small + fn_small)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        metrics_by_condition[condition] = {
            "images": float(len(image_files)),
            "gt_count": float(c["gt_count"]),
            "pred_count": float(c["pred_count"]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_small": tp_small,
            "fn_small": fn_small,
            "recall_small": recall_small,
        }

    clean_small = metrics_by_condition.get("clean", {}).get("recall_small", 0.0)
    clean_small = float(clean_small or 0.0)
    robustness: dict[str, float] = {}
    for condition, m in metrics_by_condition.items():
        if condition == "clean":
            continue
        cur = float(m.get("recall_small", 0.0))
        abs_drop = clean_small - cur
        rel_drop = _safe_div(abs_drop, max(clean_small, 1e-6))
        robustness[f"{condition}_recall_small_abs_drop"] = abs_drop
        robustness[f"{condition}_recall_small_rel_drop"] = rel_drop

    return {
        "num_images": len(image_files),
        "conditions": conditions,
        "metrics_by_condition": metrics_by_condition,
        "robustness": robustness,
    }


def evaluate_gates(
    metrics_by_condition: dict[str, dict[str, float]],
    gates: dict[str, float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    clean = metrics_by_condition.get("clean", {})
    clean_recall = float(clean.get("recall", 0.0))
    clean_precision = float(clean.get("precision", 0.0))
    clean_small = float(clean.get("recall_small", 0.0))

    if "clean_recall_small_min" in gates:
        th = float(gates["clean_recall_small_min"])
        checks.append(
            {
                "name": "clean_recall_small_min",
                "value": clean_small,
                "threshold": th,
                "passed": clean_small >= th,
            }
        )
    if "clean_recall_min" in gates:
        th = float(gates["clean_recall_min"])
        checks.append(
            {
                "name": "clean_recall_min",
                "value": clean_recall,
                "threshold": th,
                "passed": clean_recall >= th,
            }
        )
    if "clean_precision_min" in gates:
        th = float(gates["clean_precision_min"])
        checks.append(
            {
                "name": "clean_precision_min",
                "value": clean_precision,
                "threshold": th,
                "passed": clean_precision >= th,
            }
        )

    if "s3_recall_small_drop_max" in gates:
        th = float(gates["s3_recall_small_drop_max"])
        rel_drops: list[float] = []
        for condition, m in metrics_by_condition.items():
            # Accept both naming styles used in this repo:
            # 1) s3_blur, s3_fog ...
            # 2) blur_s3, fog_s3 ...
            is_s3 = condition.startswith("s3_") or condition.endswith("_s3")
            if not is_s3:
                continue
            cur_small = float(m.get("recall_small", 0.0))
            rel_drop = _safe_div(clean_small - cur_small, max(clean_small, 1e-6))
            rel_drops.append(rel_drop)
        max_rel_drop = max(rel_drops) if rel_drops else 0.0
        checks.append(
            {
                "name": "s3_recall_small_drop_max",
                "value": max_rel_drop,
                "threshold": th,
                "passed": max_rel_drop <= th,
            }
        )

    overall_pass = all(bool(c["passed"]) for c in checks) if checks else False
    return {"pass": overall_pass, "checks": checks}
