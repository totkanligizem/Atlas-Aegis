#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_class_min_conf_map(text: str) -> dict[int, float]:
    out: dict[int, float] = {}
    raw = str(text).strip()
    if not raw:
        return out
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"Invalid class-min-conf entry: {p} (expected class:conf)")
        k_s, v_s = p.split(":", 1)
        cls = int(float(k_s.strip()))
        conf = float(v_s.strip())
        if cls < 0:
            raise ValueError(f"class id must be >= 0 in class-min-conf map: {p}")
        out[cls] = max(0.0, conf)
    return out


def _load_visdrone_gt(
    ann_path: Path,
    *,
    max_frames: int,
) -> tuple[list[tuple[int, int, float, float, float, float, float, int]], dict[str, float]]:
    if not ann_path.exists():
        raise FileNotFoundError(f"GT annotation not found: {ann_path}")

    rows: list[tuple[int, int, float, float, float, float, float, int]] = []
    stats = {
        "rows_total": 0.0,
        "rows_kept": 0.0,
        "rows_dropped_invalid": 0.0,
        "rows_dropped_frame_limit": 0.0,
        "rows_dropped_category": 0.0,
    }
    valid_categories = set(range(1, 11))

    for ln in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        stats["rows_total"] += 1.0
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 8:
            stats["rows_dropped_invalid"] += 1.0
            continue

        frame_id = _safe_int(parts[0], 0)
        if frame_id <= 0:
            stats["rows_dropped_invalid"] += 1.0
            continue
        if max_frames > 0 and frame_id > max_frames:
            stats["rows_dropped_frame_limit"] += 1.0
            continue

        # VisDrone MOT ids are 0-based; evaluator expects positive ids.
        track_id = _safe_int(parts[1], -1) + 1
        left = _safe_float(parts[2], 0.0)
        top = _safe_float(parts[3], 0.0)
        width = _safe_float(parts[4], 0.0)
        height = _safe_float(parts[5], 0.0)
        score = _safe_float(parts[6], 0.0)
        category = _safe_int(parts[7], -1)
        if (
            track_id <= 0
            or width <= 0.0
            or height <= 0.0
            or score <= 0.0
            or category not in valid_categories
        ):
            if category not in valid_categories:
                stats["rows_dropped_category"] += 1.0
            else:
                stats["rows_dropped_invalid"] += 1.0
            continue

        class_id = category - 1
        rows.append((frame_id, track_id, left, top, width, height, 1.0, class_id))
        stats["rows_kept"] += 1.0

    rows.sort(key=lambda x: (x[0], x[1]))
    return rows, stats


def _load_pred_from_jsonl(
    pred_jsonl: Path,
    *,
    max_frames: int,
    min_track_age: int,
    min_conf: float,
    min_conf_relaxed: float,
    min_conf_relax_age_start: int,
    min_roi_dwell: int,
    class_min_conf_map: dict[int, float],
) -> tuple[list[tuple[int, int, float, float, float, float, float, int]], dict[str, float]]:
    if not pred_jsonl.exists():
        raise FileNotFoundError(f"Prediction JSONL not found: {pred_jsonl}")

    rows: list[tuple[int, int, float, float, float, float, float, int]] = []
    stats = {
        "frames_total": 0.0,
        "frames_kept": 0.0,
        "tracks_total": 0.0,
        "tracks_kept": 0.0,
        "tracks_dropped_invalid": 0.0,
        "tracks_dropped_frame_limit": 0.0,
        "tracks_dropped_low_age": 0.0,
        "tracks_dropped_low_conf": 0.0,
        "tracks_dropped_low_roi_dwell": 0.0,
        "tracks_conf_relax_applied": 0.0,
    }

    for ln in pred_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            payload = json.loads(ln)
        except Exception:
            continue

        stats["frames_total"] += 1.0
        frame_id_zero = _safe_int(payload.get("frame_id", -1), -1)
        frame_id = frame_id_zero + 1
        if frame_id <= 0:
            continue
        if max_frames > 0 and frame_id > max_frames:
            stats["tracks_dropped_frame_limit"] += float(
                len(payload.get("tracks", []) or [])
            )
            continue
        stats["frames_kept"] += 1.0

        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            continue
        stats["tracks_total"] += float(len(tracks))

        for t in tracks:
            track_id = _safe_int(t.get("track_id", -1), -1)
            conf = _safe_float(t.get("conf", 0.0), 0.0)
            class_id = _safe_int(t.get("class_id", -1), -1)
            age_frames = _safe_int(t.get("age_frames", 0), 0)
            roi_dwell = _safe_int(t.get("roi_dwell", 0), 0)
            class_thr = (
                float(class_min_conf_map.get(class_id, min_conf))
                if class_id >= 0
                else float(min_conf)
            )
            thr = class_thr
            if min_conf_relaxed >= 0.0 and min_conf_relax_age_start > 0 and age_frames >= min_conf_relax_age_start:
                thr = min(thr, float(min_conf_relaxed))
                if thr < class_thr:
                    stats["tracks_conf_relax_applied"] += 1.0

            if age_frames < min_track_age:
                stats["tracks_dropped_low_age"] += 1.0
                continue
            if conf < thr:
                stats["tracks_dropped_low_conf"] += 1.0
                continue
            if roi_dwell < min_roi_dwell:
                stats["tracks_dropped_low_roi_dwell"] += 1.0
                continue

            bbox = t.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                stats["tracks_dropped_invalid"] += 1.0
                continue
            x1 = _safe_float(bbox[0], 0.0)
            y1 = _safe_float(bbox[1], 0.0)
            x2 = _safe_float(bbox[2], 0.0)
            y2 = _safe_float(bbox[3], 0.0)
            width = x2 - x1
            height = y2 - y1
            if track_id <= 0 or width <= 0.0 or height <= 0.0:
                stats["tracks_dropped_invalid"] += 1.0
                continue
            rows.append((frame_id, track_id, x1, y1, width, height, conf, class_id))
            stats["tracks_kept"] += 1.0

    rows.sort(key=lambda x: (x[0], x[1]))
    return rows, stats


def _write_mot(path: Path, rows: list[tuple[int, int, float, float, float, float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{fr},{tid},{l:.3f},{t:.3f},{w:.3f},{h:.3f},{conf:.6f},{cls}"
        for fr, tid, l, t, w, h, conf, cls in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-annotation", required=True, help="VisDrone MOT annotation txt path.")
    p.add_argument("--pred-jsonl", required=True, help="track_risk jsonl path.")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--out-gt", default="data/mot/gt.txt")
    p.add_argument("--out-pred", default="data/mot/pred.txt")
    p.add_argument("--min-track-age", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument(
        "--class-min-conf-map",
        default="",
        help="Optional per-class min-conf map, e.g. '3:0.28,1:0.27'",
    )
    p.add_argument(
        "--min-conf-relaxed",
        type=float,
        default=-1.0,
        help="Optional relaxed min-conf for mature tracks (disabled if < 0).",
    )
    p.add_argument(
        "--min-conf-relax-age-start",
        type=int,
        default=0,
        help="Age threshold to apply relaxed min-conf (disabled if <= 0).",
    )
    p.add_argument("--min-roi-dwell", type=int, default=0)
    p.add_argument("--report-out", default="reports/mot_build_report.json")
    args = p.parse_args()

    gt_ann = Path(args.gt_annotation)
    pred_jsonl = Path(args.pred_jsonl)
    out_gt = Path(args.out_gt)
    out_pred = Path(args.out_pred)
    max_frames = int(args.max_frames)
    class_min_conf_map = _parse_class_min_conf_map(args.class_min_conf_map)

    gt_rows, gt_stats = _load_visdrone_gt(gt_ann, max_frames=max_frames)
    pred_rows, pred_stats = _load_pred_from_jsonl(
        pred_jsonl,
        max_frames=max_frames,
        min_track_age=max(1, int(args.min_track_age)),
        min_conf=max(0.0, float(args.min_conf)),
        min_conf_relaxed=float(args.min_conf_relaxed),
        min_conf_relax_age_start=max(0, int(args.min_conf_relax_age_start)),
        min_roi_dwell=max(0, int(args.min_roi_dwell)),
        class_min_conf_map=class_min_conf_map,
    )

    _write_mot(out_gt, gt_rows)
    _write_mot(out_pred, pred_rows)

    gt_frames = sorted({r[0] for r in gt_rows})
    pred_frames = sorted({r[0] for r in pred_rows})
    report = {
        "status": "SUCCESS",
        "inputs": {
            "gt_annotation": str(gt_ann),
            "pred_jsonl": str(pred_jsonl),
            "max_frames": max_frames,
            "min_track_age": max(1, int(args.min_track_age)),
            "min_conf": max(0.0, float(args.min_conf)),
            "class_min_conf_map": {str(k): v for k, v in sorted(class_min_conf_map.items())},
            "min_conf_relaxed": float(args.min_conf_relaxed),
            "min_conf_relax_age_start": max(0, int(args.min_conf_relax_age_start)),
            "min_roi_dwell": max(0, int(args.min_roi_dwell)),
        },
        "outputs": {
            "gt_mot": str(out_gt),
            "pred_mot": str(out_pred),
        },
        "gt_stats": gt_stats,
        "pred_stats": pred_stats,
        "frame_coverage": {
            "gt_frames_min": gt_frames[0] if gt_frames else 0,
            "gt_frames_max": gt_frames[-1] if gt_frames else 0,
            "gt_num_frames": len(gt_frames),
            "pred_frames_min": pred_frames[0] if pred_frames else 0,
            "pred_frames_max": pred_frames[-1] if pred_frames else 0,
            "pred_num_frames": len(pred_frames),
        },
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
