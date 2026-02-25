from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics_store import end_run, init_db, log_metric, start_run


@dataclass
class BenchConfig:
    source: str
    model: str
    conf: float
    iou: float
    imgsz: int
    device: str | None
    max_frames: int
    frame_stride: int
    db_path: str
    output_report: str


def _load_frames(source: str, max_frames: int, frame_stride: int) -> list[Any]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for benchmark frame loading.") from exc

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frames: list[Any] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, frame_stride) == 0:
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def _predict_stats_ultralytics(
    frames: list[Any],
    condition: str,
    model_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
) -> dict[str, float]:
    from .corruptions import apply_condition

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for non-dry benchmark mode.") from exc

    model = YOLO(model_path)
    total_dets = 0.0
    total_conf = 0.0
    total_inf_ms = 0.0

    for i, frame in enumerate(frames):
        corrupted = apply_condition(frame, condition=condition, seed_key=f"frame-{i}")
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "source": corrupted,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        res = model.predict(**kwargs)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        total_inf_ms += inf_ms

        if not res:
            continue
        boxes = res[0].boxes
        if boxes is None:
            continue
        n = len(boxes)
        total_dets += float(n)
        if n > 0:
            total_conf += float(boxes.conf.mean().item()) if hasattr(boxes.conf, "mean") else 0.0

    nframes = max(1, len(frames))
    return {
        "frames": float(len(frames)),
        "mean_detections": total_dets / nframes,
        "mean_conf": total_conf / nframes,
        "mean_infer_ms": total_inf_ms / nframes,
    }


def _predict_stats_dry(frames: list[Any], condition: str) -> dict[str, float]:
    import hashlib
    import random

    # Use a stable hash to keep dry-run outputs reproducible across processes.
    seed = int.from_bytes(hashlib.sha256(condition.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    base = {
        "clean": 4.2,
        "s3_blur": 3.3,
        "s3_low_light": 3.0,
        "s3_jpeg": 3.5,
        "s3_fog": 3.1,
    }.get(condition, 3.0)
    nframes = max(1, len(frames))
    jitter = rng.uniform(-0.2, 0.2)
    det = max(0.0, base + jitter)
    conf = max(0.05, min(0.95, 0.58 + jitter * 0.08))
    infer_ms = max(5.0, 16.0 + rng.uniform(-2.0, 2.0))
    return {
        "frames": float(nframes),
        "mean_detections": det,
        "mean_conf": conf,
        "mean_infer_ms": infer_ms,
    }


def run_benchmark(
    bench: BenchConfig,
    tier: str,
    conditions: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    import json

    run_id = f"bench_{tier}_{uuid.uuid4().hex[:10]}"
    init_db(bench.db_path)
    start_run(
        db_path=bench.db_path,
        run_id=run_id,
        pipeline="corruption_benchmark",
        tier=tier,
        mode="dry_run" if dry_run else "ultralytics",
        config_obj={
            "source": bench.source,
            "model": bench.model,
            "conditions": conditions,
            "max_frames": bench.max_frames,
            "frame_stride": bench.frame_stride,
            "dry_run": dry_run,
        },
    )

    if dry_run:
        nframes = bench.max_frames if bench.max_frames > 0 else 120
        frames: list[Any] = [None] * nframes
    else:
        frames = _load_frames(
            source=bench.source,
            max_frames=bench.max_frames,
            frame_stride=bench.frame_stride,
        )
        if len(frames) == 0:
            summary = {"run_id": run_id, "status": "FAILED", "error": "No frames loaded from source."}
            end_run(bench.db_path, run_id=run_id, status="FAILED", summary_obj=summary)
            return summary

    results: dict[str, dict[str, float]] = {}
    for condition in conditions:
        if dry_run:
            stats = _predict_stats_dry(frames=frames, condition=condition)
        else:
            stats = _predict_stats_ultralytics(
                frames=frames,
                condition=condition,
                model_path=bench.model,
                conf=bench.conf,
                iou=bench.iou,
                imgsz=bench.imgsz,
                device=bench.device,
            )
        results[condition] = stats
        log_metric(bench.db_path, run_id, condition, "mean_detections", stats["mean_detections"])
        log_metric(bench.db_path, run_id, condition, "mean_conf", stats["mean_conf"])
        log_metric(bench.db_path, run_id, condition, "mean_infer_ms", stats["mean_infer_ms"])

    clean = results.get("clean", {"mean_detections": 1.0, "mean_conf": 1.0})
    clean_det = max(clean["mean_detections"], 1e-6)
    clean_conf = max(clean["mean_conf"], 1e-6)

    for condition, stats in results.items():
        det_drop = (clean_det - stats["mean_detections"]) / clean_det
        conf_drop = (clean_conf - stats["mean_conf"]) / clean_conf
        log_metric(bench.db_path, run_id, condition, "rel_drop_det_vs_clean", det_drop)
        log_metric(bench.db_path, run_id, condition, "rel_drop_conf_vs_clean", conf_drop)

    summary = {
        "run_id": run_id,
        "status": "SUCCESS",
        "tier": tier,
        "mode": "dry_run" if dry_run else "ultralytics",
        "num_conditions": len(conditions),
        "num_frames": len(frames),
        "results": results,
    }
    end_run(bench.db_path, run_id=run_id, status="SUCCESS", summary_obj=summary)

    out = Path(bench.output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
