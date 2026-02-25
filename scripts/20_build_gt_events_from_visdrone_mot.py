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


def _infer_frame_size(sequence_dir: Path) -> tuple[int, int]:
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence dir not found: {sequence_dir}")
    images = sorted(
        p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"No image found under sequence dir: {sequence_dir}")
    sample = images[0]
    try:
        from PIL import Image  # type: ignore

        with Image.open(sample) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        try:
            import cv2  # type: ignore

            arr = cv2.imread(str(sample))
            if arr is None:
                raise RuntimeError("cv2.imread returned None")
            h, w = arr.shape[:2]
            return int(w), int(h)
        except Exception as exc:
            raise RuntimeError(f"Could not infer frame size from: {sample}") from exc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", required=True, help="VisDrone MOT annotation txt path")
    p.add_argument("--sequence-dir", default="", help="Sequence frame folder used to infer frame size")
    p.add_argument("--frame-width", type=int, default=0)
    p.add_argument("--frame-height", type=int, default=0)
    p.add_argument("--roi-x1", type=float, default=0.2)
    p.add_argument("--roi-y1", type=float, default=0.2)
    p.add_argument("--roi-x2", type=float, default=0.8)
    p.add_argument("--roi-y2", type=float, default=0.8)
    p.add_argument("--min-frames-in-roi", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--out-events", default="data/mot/gt_events.json")
    p.add_argument("--report-out", default="reports/gt_event_build_report.json")
    args = p.parse_args()

    ann_path = Path(args.annotation)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    frame_w = int(args.frame_width)
    frame_h = int(args.frame_height)
    seq_dir = Path(args.sequence_dir) if str(args.sequence_dir).strip() else None
    if frame_w <= 0 or frame_h <= 0:
        if seq_dir is None:
            raise ValueError(
                "Provide --frame-width/--frame-height or --sequence-dir to infer frame size."
            )
        frame_w, frame_h = _infer_frame_size(seq_dir)

    rx1 = float(args.roi_x1) * float(frame_w)
    ry1 = float(args.roi_y1) * float(frame_h)
    rx2 = float(args.roi_x2) * float(frame_w)
    ry2 = float(args.roi_y2) * float(frame_h)

    max_frames = int(args.max_frames)
    min_frames_in_roi = int(args.min_frames_in_roi)

    by_track: dict[int, list[tuple[int, float, float]]] = {}
    stats = {
        "rows_total": 0.0,
        "rows_kept": 0.0,
        "rows_dropped_invalid": 0.0,
        "rows_dropped_frame_limit": 0.0,
    }

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

        track_id = _safe_int(parts[1], -1) + 1  # convert to positive id
        left = _safe_float(parts[2], 0.0)
        top = _safe_float(parts[3], 0.0)
        width = _safe_float(parts[4], 0.0)
        height = _safe_float(parts[5], 0.0)
        score = _safe_float(parts[6], 0.0)
        if track_id <= 0 or width <= 0.0 or height <= 0.0 or score <= 0.0:
            stats["rows_dropped_invalid"] += 1.0
            continue

        cx = left + width / 2.0
        cy = top + height / 2.0
        by_track.setdefault(track_id, []).append((frame_id, cx, cy))
        stats["rows_kept"] += 1.0

    events: list[dict[str, Any]] = []
    candidates = 0
    for tid, rows in by_track.items():
        rows.sort(key=lambda x: x[0])
        dwell = 0
        event_frame: int | None = None
        for frame_id, cx, cy in rows:
            in_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
            dwell = dwell + 1 if in_roi else 0
            if event_frame is None and dwell >= min_frames_in_roi:
                event_frame = frame_id
                break
        if event_frame is not None:
            candidates += 1
            events.append(
                {
                    "type": "event_start",
                    "track_id": tid,
                    "event_start_frame": event_frame,
                    "frame_id": event_frame,
                }
            )

    events.sort(key=lambda e: (int(e["event_start_frame"]), int(e["track_id"])))

    out_events = Path(args.out_events)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_events.write_text(json.dumps(events, indent=2), encoding="utf-8")

    report = {
        "status": "SUCCESS",
        "inputs": {
            "annotation": str(ann_path),
            "sequence_dir": str(seq_dir) if seq_dir else "",
            "frame_width": frame_w,
            "frame_height": frame_h,
            "roi": {
                "x1": float(args.roi_x1),
                "y1": float(args.roi_y1),
                "x2": float(args.roi_x2),
                "y2": float(args.roi_y2),
            },
            "min_frames_in_roi": min_frames_in_roi,
            "max_frames": max_frames,
        },
        "stats": stats,
        "track_count": len(by_track),
        "event_track_count": candidates,
        "events_out": str(out_events),
    }
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote events: {out_events}")
    print(f"wrote report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
