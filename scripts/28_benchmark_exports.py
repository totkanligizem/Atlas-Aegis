#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_runtime import apply_budget_guard
from aerial_stack.config import load_yaml
from aerial_stack.legal_gate import check_dataset_usage


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
BACKEND_PYTORCH = "pytorch"


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


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _normalize_backend_name(raw: str) -> str:
    x = str(raw).strip().lower()
    if x in {"onnx"}:
        return "onnx"
    if x in {"coreml", "mlmodel", "mlpackage"}:
        return "coreml"
    return x


def _model_size_factor(model_path: str) -> float:
    name = Path(model_path).name.lower()
    if "yolov8n" in name:
        return 1.0
    if "yolov8s" in name:
        return 1.55
    if "yolov8m" in name:
        return 2.2
    if "yolov8l" in name:
        return 2.9
    if "yolov8x" in name:
        return 3.6
    return 1.35


def _backend_latency_factor(backend: str) -> float:
    b = _normalize_backend_name(backend)
    if b == BACKEND_PYTORCH:
        return 1.0
    if b == "onnx":
        return 0.88
    if b == "coreml":
        return 0.78
    return 1.0


def _device_factor(device: str) -> float:
    d = str(device).strip().lower()
    if d == "mps":
        return 0.60
    if d == "cpu" or not d:
        return 1.0
    return 0.85


def _resolve_device(device_raw: str) -> tuple[str | None, str | None]:
    requested = str(device_raw).strip()
    if not requested or requested.lower() == "auto":
        return None, None
    low = requested.lower()
    if low == "cpu":
        return "cpu", None
    try:
        import torch  # type: ignore
    except Exception:
        return requested, "torch is not available to validate requested device."

    if low == "mps":
        if not bool(torch.backends.mps.is_available()):
            return requested, "mps requested but torch.backends.mps.is_available() is false."
        return "mps", None
    if low == "cuda" or low.startswith("cuda:") or low.isdigit():
        if not bool(torch.cuda.is_available()):
            return requested, "cuda requested but torch.cuda.is_available() is false."
        return requested, None
    return requested, None


def _collect_image_paths(source_dir: Path, *, max_frames: int, frame_stride: int) -> list[Path]:
    files = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    step = max(1, int(frame_stride))
    selected = files[::step]
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def _dry_source_info(source: str, *, max_frames: int, frame_stride: int) -> dict[str, Any]:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if src.is_dir():
        selected = _collect_image_paths(src, max_frames=max_frames, frame_stride=frame_stride)
        if not selected:
            raise RuntimeError(f"No images found under source dir: {src}")
        return {
            "source_type": "image_dir",
            "frames_total": len(selected),
            "selected_items": len(selected),
            "frame_stride": int(frame_stride),
        }
    total = int(max_frames) if int(max_frames) > 0 else 120
    return {
        "source_type": "video_file",
        "frames_total": total,
        "selected_items": total,
        "frame_stride": int(frame_stride),
    }


def _load_real_frames(source: str, *, max_frames: int, frame_stride: int) -> tuple[list[Any], dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for real export benchmark mode.") from exc

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    if src.is_dir():
        selected = _collect_image_paths(src, max_frames=max_frames, frame_stride=frame_stride)
        if not selected:
            raise RuntimeError(f"No images found under source dir: {src}")
        frames: list[Any] = []
        read_failures = 0
        for p in selected:
            img = cv2.imread(str(p))
            if img is None:
                read_failures += 1
                continue
            frames.append(img)
        if not frames:
            raise RuntimeError(f"No readable images found under source dir: {src}")
        info = {
            "source_type": "image_dir",
            "frames_total": len(frames),
            "selected_items": len(selected),
            "read_failures": int(read_failures),
            "frame_stride": int(frame_stride),
        }
        return frames, info

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")
    frames = []
    idx = 0
    step = max(1, int(frame_stride))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video source: {src}")
    info = {
        "source_type": "video_file",
        "frames_total": len(frames),
        "decoded_items": len(frames),
        "frame_stride": int(frame_stride),
    }
    return frames, info


def _build_metrics(latencies_ms: list[float], detections: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p90_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
            "fps_mean": 0.0,
            "fps_median": 0.0,
            "fps_p05": 0.0,
            "detections_mean": 0.0,
        }
    fps_vals = [1000.0 / x for x in latencies_ms if x > 1e-9]
    return {
        "latency_mean_ms": float(statistics.mean(latencies_ms)),
        "latency_median_ms": float(statistics.median(latencies_ms)),
        "latency_p90_ms": _percentile(latencies_ms, 0.90),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "latency_p99_ms": _percentile(latencies_ms, 0.99),
        "latency_min_ms": float(min(latencies_ms)),
        "latency_max_ms": float(max(latencies_ms)),
        "fps_mean": float(statistics.mean(fps_vals)) if fps_vals else 0.0,
        "fps_median": float(statistics.median(fps_vals)) if fps_vals else 0.0,
        "fps_p05": _percentile(fps_vals, 0.05) if fps_vals else 0.0,
        "detections_mean": float(statistics.mean(detections)) if detections else 0.0,
    }


def _gate_for_metrics(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    target_fps = float(thresholds.get("target_fps", 0.0))
    target_latency = float(thresholds.get("target_latency_ms", 0.0))
    target_p95 = float(thresholds.get("target_p95_latency_ms", 0.0))

    if target_fps > 0.0:
        val = float(metrics.get("fps_mean", 0.0))
        checks.append(
            {
                "name": "fps_mean_min",
                "value": val,
                "threshold": target_fps,
                "passed": bool(val >= target_fps),
            }
        )
    if target_latency > 0.0:
        val = float(metrics.get("latency_median_ms", 0.0))
        checks.append(
            {
                "name": "latency_median_ms_max",
                "value": val,
                "threshold": target_latency,
                "passed": bool(val <= target_latency),
            }
        )
    if target_p95 > 0.0:
        val = float(metrics.get("latency_p95_ms", 0.0))
        checks.append(
            {
                "name": "latency_p95_ms_max",
                "value": val,
                "threshold": target_p95,
                "passed": bool(val <= target_p95),
            }
        )

    gate_pass = all(bool(c.get("passed", False)) for c in checks) if checks else True
    return {"pass": bool(gate_pass), "checks": checks}


def _requirement_of_backend(profile_cfg: dict[str, Any], backend: str) -> str:
    reqs = profile_cfg.get("backend_requirements", {})
    if isinstance(reqs, dict):
        raw = str(reqs.get(backend, "")).strip().lower()
        if raw in {"required", "optional"}:
            return raw
    return "required" if backend == BACKEND_PYTORCH else "optional"


def _synthetic_series(
    *,
    profile_name: str,
    model_path: str,
    imgsz: int,
    device: str,
    backend: str,
    measured_frames: int,
) -> tuple[list[float], list[float]]:
    key = f"{profile_name}|{model_path}|{imgsz}|{device}|{backend}"
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)

    size_factor = _model_size_factor(model_path)
    img_factor = (max(32, int(imgsz)) / 640.0) ** 1.05
    dev_factor = _device_factor(device)
    backend_factor = _backend_latency_factor(backend)
    base_ms = 32.0 * size_factor * img_factor * dev_factor * backend_factor

    latencies = [
        max(2.0, rng.gauss(base_ms, max(2.0, base_ms * 0.07)))
        for _ in range(max(10, measured_frames))
    ]
    det_base = 4.0 + (size_factor - 1.0) * 1.3 + max(0.0, (imgsz - 640) / 640.0) * 0.5
    detections = [
        max(0.0, rng.gauss(det_base, 0.35))
        for _ in range(max(10, measured_frames))
    ]
    return latencies, detections


def _empty_backend_result(
    *,
    backend: str,
    requirement: str,
    num_frames_total: int,
    warmup_frames: int,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "requirement": requirement,
        "status": "ERROR",
        "num_frames_total": int(num_frames_total),
        "warmup_frames": int(max(0, warmup_frames)),
        "num_frames_measured": 0,
        "export_path": "",
        "metrics": {},
        "gate": {"pass": False, "checks": []},
        "speedup_vs_pytorch_fps": 0.0,
        "reason": "",
        "error": "",
    }


def _error_is_optional_dependency(msg: str, backend: str) -> bool:
    m = str(msg).lower()
    b = _normalize_backend_name(backend)
    if b == "onnx":
        keys = ["onnx", "onnxruntime", "module named", "protobuf"]
        return any(k in m for k in keys)
    if b == "coreml":
        keys = ["coreml", "coremltools", "module named", "xcode", "xcrun", "macos"]
        return any(k in m for k in keys)
    return False


def _resolve_export_path(export_out: Any, expected_dir: Path, backend: str) -> Path | None:
    candidates: list[Path] = []
    if isinstance(export_out, (str, Path)):
        p = Path(str(export_out))
        candidates.append(p)
    suffixes = [".onnx"] if backend == "onnx" else [".mlpackage", ".mlmodel"]

    for c in candidates:
        if c.exists():
            return c

    if expected_dir.exists():
        found: list[Path] = []
        for suf in suffixes:
            found.extend(expected_dir.rglob(f"*{suf}"))
        if found:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return found[0]

    return None


def _benchmark_yolo_backend(
    *,
    model_ref: str,
    frames: list[Any],
    conf: float,
    iou: float,
    imgsz: int,
    warmup_frames: int,
    device: str | None,
) -> tuple[dict[str, float], int]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ultralytics import failed: {exc}") from exc

    model = YOLO(model_ref)
    warmup_eff = min(int(warmup_frames), max(0, len(frames) - 1))
    latencies_ms: list[float] = []
    detections: list[float] = []

    for idx, frame in enumerate(frames):
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(conf),
            "iou": float(iou),
            "imgsz": int(imgsz),
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        t0 = time.perf_counter()
        result = model.predict(**kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if idx < warmup_eff:
            continue
        n_det = 0.0
        if result:
            boxes = result[0].boxes
            if boxes is not None:
                n_det = float(len(boxes))
        latencies_ms.append(float(dt_ms))
        detections.append(float(n_det))

    if not latencies_ms:
        raise RuntimeError("No measured frames remain after warmup.")
    return _build_metrics(latencies_ms, detections), len(latencies_ms)


def _export_backend_model(
    *,
    model_path: str,
    imgsz: int,
    backend: str,
    export_root: Path,
    profile_name: str,
    device: str | None,
) -> Path:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ultralytics import failed: {exc}") from exc

    export_dir = export_root / profile_name / backend
    export_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    kwargs: dict[str, Any] = {
        "format": backend,
        "imgsz": int(imgsz),
        "project": str(export_root),
        "name": f"{profile_name}_{backend}",
        "exist_ok": True,
        "verbose": False,
    }
    if device and backend != "coreml":
        kwargs["device"] = device
    export_out = model.export(**kwargs)

    resolved = _resolve_export_path(export_out, export_root / f"{profile_name}_{backend}", backend)
    if resolved is None:
        resolved = _resolve_export_path(export_out, export_dir, backend)
    if resolved is None:
        raise RuntimeError(
            f"Export completed but output file could not be resolved for backend={backend}"
        )
    return resolved


def _profile_header(profile_cfg: dict[str, Any], *, frames_total: int, warmup_frames: int) -> dict[str, Any]:
    model_path = str(profile_cfg.get("model", "")).strip()
    if not model_path:
        raise ValueError("profile.model is required.")
    return {
        "name": str(profile_cfg.get("name", Path(model_path).stem)),
        "model": model_path,
        "imgsz": max(32, _safe_int(profile_cfg.get("imgsz", 640), 640)),
        "device_requested": str(profile_cfg.get("device", "")).strip(),
        "device_effective": "",
        "target_thresholds": {
            "target_fps": max(0.0, _safe_float(profile_cfg.get("target_fps", 0.0), 0.0)),
            "target_latency_ms": max(
                0.0, _safe_float(profile_cfg.get("target_latency_ms", 0.0), 0.0)
            ),
            "target_p95_latency_ms": max(
                0.0, _safe_float(profile_cfg.get("target_p95_latency_ms", 0.0), 0.0)
            ),
        },
        "num_frames_total": int(frames_total),
        "warmup_frames": int(max(0, warmup_frames)),
        "backends": [],
        "status": "FAIL",
        "required_backends": [],
    }


def _run_profile_dry(
    profile_cfg: dict[str, Any],
    *,
    backends: list[str],
    frames_total: int,
    warmup_frames: int,
) -> dict[str, Any]:
    out = _profile_header(profile_cfg, frames_total=frames_total, warmup_frames=warmup_frames)
    dev_eff, dev_issue = _resolve_device(str(out["device_requested"]))
    out["device_effective"] = str(dev_eff or "auto")

    measured = int(frames_total) - min(int(warmup_frames), max(0, int(frames_total) - 1))
    measured = max(1, measured)
    thresholds = out["target_thresholds"]

    pytorch_fps = 0.0
    for backend in backends:
        req = _requirement_of_backend(profile_cfg, backend)
        row = _empty_backend_result(
            backend=backend,
            requirement=req,
            num_frames_total=frames_total,
            warmup_frames=warmup_frames,
        )
        if req == "required":
            out["required_backends"].append(backend)
        if dev_issue and req == "required":
            row["status"] = "ERROR"
            row["error"] = dev_issue
            out["backends"].append(row)
            continue
        if dev_issue and req == "optional":
            row["status"] = "SKIPPED"
            row["reason"] = dev_issue
            out["backends"].append(row)
            continue

        lat, det = _synthetic_series(
            profile_name=str(out["name"]),
            model_path=str(out["model"]),
            imgsz=int(out["imgsz"]),
            device=str(dev_eff or "auto"),
            backend=backend,
            measured_frames=measured,
        )
        row["num_frames_measured"] = measured
        row["metrics"] = _build_metrics(lat, det)
        row["gate"] = _gate_for_metrics(row["metrics"], thresholds)
        row["status"] = "PASS" if bool(row["gate"].get("pass", False)) else "FAIL"
        if backend == BACKEND_PYTORCH:
            pytorch_fps = float(row["metrics"].get("fps_mean", 0.0) or 0.0)
        out["backends"].append(row)

    for row in out["backends"]:
        m = row.get("metrics", {})
        if not isinstance(m, dict):
            continue
        fps = float(m.get("fps_mean", 0.0) or 0.0)
        if pytorch_fps > 1e-9:
            row["speedup_vs_pytorch_fps"] = fps / pytorch_fps

    profile_pass = True
    for row in out["backends"]:
        if row.get("requirement") != "required":
            continue
        if str(row.get("status")) != "PASS":
            profile_pass = False
            break
    out["status"] = "PASS" if profile_pass else "FAIL"
    return out


def _run_profile_real(
    profile_cfg: dict[str, Any],
    *,
    backends: list[str],
    frames: list[Any],
    warmup_frames: int,
    conf: float,
    iou: float,
    export_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    out = _profile_header(profile_cfg, frames_total=len(frames), warmup_frames=warmup_frames)
    warnings: list[str] = []
    model_path = str(out["model"])
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Profile model path does not exist: {model_path}")

    dev_eff, dev_issue = _resolve_device(str(out["device_requested"]))
    out["device_effective"] = str(dev_eff or "auto")
    thresholds = out["target_thresholds"]

    pytorch_fps = 0.0
    for backend in backends:
        req = _requirement_of_backend(profile_cfg, backend)
        row = _empty_backend_result(
            backend=backend,
            requirement=req,
            num_frames_total=len(frames),
            warmup_frames=warmup_frames,
        )
        if req == "required":
            out["required_backends"].append(backend)

        if dev_issue and req == "required":
            row["status"] = "ERROR"
            row["error"] = dev_issue
            out["backends"].append(row)
            continue
        if dev_issue and req == "optional":
            row["status"] = "SKIPPED"
            row["reason"] = dev_issue
            out["backends"].append(row)
            continue

        try:
            if backend == BACKEND_PYTORCH:
                bench_ref = model_path
                bench_device = dev_eff
            else:
                exported = _export_backend_model(
                    model_path=model_path,
                    imgsz=int(out["imgsz"]),
                    backend=backend,
                    export_root=export_root,
                    profile_name=str(out["name"]),
                    device=dev_eff,
                )
                row["export_path"] = str(exported)
                bench_ref = str(exported)
                bench_device = None

            metrics, measured = _benchmark_yolo_backend(
                model_ref=bench_ref,
                frames=frames,
                conf=conf,
                iou=iou,
                imgsz=int(out["imgsz"]),
                warmup_frames=warmup_frames,
                device=bench_device,
            )
            row["num_frames_measured"] = measured
            row["metrics"] = metrics
            row["gate"] = _gate_for_metrics(metrics, thresholds)
            row["status"] = "PASS" if bool(row["gate"].get("pass", False)) else "FAIL"
            if backend == BACKEND_PYTORCH:
                pytorch_fps = float(metrics.get("fps_mean", 0.0) or 0.0)
        except Exception as exc:
            msg = str(exc)
            if req == "optional" and _error_is_optional_dependency(msg, backend):
                row["status"] = "SKIPPED"
                row["reason"] = msg
                warnings.append(
                    f"profile={out['name']} backend={backend} optional backend skipped: {msg}"
                )
            else:
                row["status"] = "ERROR"
                row["error"] = msg
        out["backends"].append(row)

    for row in out["backends"]:
        m = row.get("metrics", {})
        if not isinstance(m, dict):
            continue
        fps = float(m.get("fps_mean", 0.0) or 0.0)
        if pytorch_fps > 1e-9:
            row["speedup_vs_pytorch_fps"] = fps / pytorch_fps

    profile_pass = True
    for row in out["backends"]:
        if row.get("requirement") != "required":
            continue
        if str(row.get("status")) != "PASS":
            profile_pass = False
            break
    out["status"] = "PASS" if profile_pass else "FAIL"
    return out, warnings


def _summary(report_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "profiles_total": len(report_profiles),
        "profiles_pass": sum(1 for p in report_profiles if str(p.get("status")) == "PASS"),
    }
    totals["overall_gate_pass"] = bool(
        totals["profiles_pass"] == totals["profiles_total"] if totals["profiles_total"] > 0 else True
    )

    backend_counts: dict[str, dict[str, int]] = {}
    best_backend = ""
    best_fps = 0.0
    for p in report_profiles:
        for b in p.get("backends", []):
            if not isinstance(b, dict):
                continue
            backend = str(b.get("backend", "unknown"))
            status = str(b.get("status", "unknown"))
            backend_counts.setdefault(backend, {"PASS": 0, "FAIL": 0, "SKIPPED": 0, "ERROR": 0})
            if status in backend_counts[backend]:
                backend_counts[backend][status] += 1

            m = b.get("metrics", {})
            if not isinstance(m, dict):
                continue
            fps = float(m.get("fps_mean", 0.0) or 0.0)
            if status == "PASS" and fps > best_fps:
                best_fps = fps
                best_backend = f"{p.get('name', '')}:{backend}"

    totals["backend_status_counts"] = backend_counts
    totals["best_backend_by_fps_mean"] = best_backend
    totals["best_fps_mean"] = float(best_fps)
    return totals


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Export Backend Benchmark (PyTorch vs ONNX/CoreML)")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- mode: `{report.get('mode', 'unknown')}`")
    lines.append(f"- gate_status: `{report.get('gate_status', 'FAIL')}`")
    sm = report.get("summary", {})
    lines.append(
        f"- profiles_pass: `{sm.get('profiles_pass', 0)}` / `{sm.get('profiles_total', 0)}`"
    )
    lines.append(
        f"- best_backend_by_fps_mean: `{sm.get('best_backend_by_fps_mean', '')}` "
        f"({float(sm.get('best_fps_mean', 0.0) or 0.0):.2f} FPS)"
    )
    lines.append("")
    lines.append("## Backend Results")
    lines.append(
        "| profile | backend | requirement | status | fps_mean | latency_median_ms | "
        "latency_p95_ms | speedup_vs_pytorch_fps | export_path |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for p in report.get("profiles", []):
        if not isinstance(p, dict):
            continue
        for b in p.get("backends", []):
            if not isinstance(b, dict):
                continue
            m = b.get("metrics", {})
            if not isinstance(m, dict):
                m = {}
            lines.append(
                f"| {p.get('name', '')} | {b.get('backend', '')} | {b.get('requirement', '')} | "
                f"{b.get('status', '')} | {float(m.get('fps_mean', 0.0) or 0.0):.2f} | "
                f"{float(m.get('latency_median_ms', 0.0) or 0.0):.2f} | "
                f"{float(m.get('latency_p95_ms', 0.0) or 0.0):.2f} | "
                f"{float(b.get('speedup_vs_pytorch_fps', 0.0) or 0.0):.3f} | "
                f"{b.get('export_path', '')} |"
            )
    lines.append("")
    warns = report.get("warnings", [])
    if isinstance(warns, list) and warns:
        lines.append("## Warnings")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/export_benchmark.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", default="", help="Optional source override.")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--frame-stride", type=int, default=-1)
    p.add_argument("--warmup-frames", type=int, default=-1)
    p.add_argument("--out-report", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--fail-on-gate", action="store_true")
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

    cfg = load_yaml(args.config)
    source = str(args.source or cfg.get("source", "")).strip()
    if not source:
        raise ValueError("source is required in config or via --source.")
    max_frames = _safe_int(args.max_frames if args.max_frames >= 0 else cfg.get("max_frames", 120), 120)
    frame_stride = _safe_int(args.frame_stride if args.frame_stride >= 0 else cfg.get("frame_stride", 1), 1)
    warmup_frames = _safe_int(args.warmup_frames if args.warmup_frames >= 0 else cfg.get("warmup_frames", 5), 5)
    conf = _safe_float(cfg.get("conf", 0.1), 0.1)
    iou = _safe_float(cfg.get("iou", 0.7), 0.7)
    export_root = Path(str(cfg.get("export_dir", "runs/export_bench")))
    profiles = cfg.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles list is required in export benchmark config.")

    raw_formats = cfg.get("formats", ["onnx", "coreml"])
    if not isinstance(raw_formats, list):
        raw_formats = ["onnx", "coreml"]
    formats = [_normalize_backend_name(x) for x in raw_formats]
    formats = [x for x in formats if x in {"onnx", "coreml"}]
    backends = [BACKEND_PYTORCH, *formats]

    out_report = Path(str(args.out_report or cfg.get("output_report", "reports/export_benchmark_report.json")))
    out_md = Path(str(args.out_md or cfg.get("output_md", "reports/export_benchmark_report.md")))

    dataset_key = str(args.dataset_key or cfg.get("dataset_key", "")).strip()
    legal_gate: dict[str, Any] = {"status": "SKIPPED"}
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

    budget_guard: dict[str, Any] = {"status": "SKIPPED"}
    if not args.skip_budget_check:
        try:
            budget_guard = apply_budget_guard(
                budget_config_path=args.budget_config,
                budget_ledger_path=args.budget_ledger,
                budget_events_path=args.budget_events,
                source="export_benchmark",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    mode = "dry_run" if bool(args.dry_run) else "ultralytics"
    source_info: dict[str, Any]
    frames: list[Any] = []
    if args.dry_run:
        source_info = _dry_source_info(source, max_frames=max_frames, frame_stride=frame_stride)
    else:
        frames, source_info = _load_real_frames(source, max_frames=max_frames, frame_stride=frame_stride)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = len([x for x in profiles if isinstance(x, dict)])
    done = 0
    for p_cfg in profiles:
        if not isinstance(p_cfg, dict):
            continue
        done += 1
        profile_name = str(p_cfg.get("name", f"profile_{done}"))
        print(f"[export-benchmark] {done}/{total} profile={profile_name} mode={mode}", flush=True)
        if args.dry_run:
            row = _run_profile_dry(
                p_cfg,
                backends=backends,
                frames_total=int(source_info.get("frames_total", 0)),
                warmup_frames=warmup_frames,
            )
        else:
            row, warns = _run_profile_real(
                p_cfg,
                backends=backends,
                frames=frames,
                warmup_frames=warmup_frames,
                conf=conf,
                iou=iou,
                export_root=export_root,
            )
            warnings.extend(warns)
        rows.append(row)

    summary = _summary(rows)
    gate_status = "PASS" if bool(summary.get("overall_gate_pass", False)) else "FAIL"
    report: dict[str, Any] = {
        "status": "SUCCESS",
        "gate_status": gate_status,
        "mode": mode,
        "inputs": {
            "config": str(args.config),
            "source": source,
            "max_frames": int(max_frames),
            "frame_stride": int(frame_stride),
            "warmup_frames": int(max(0, warmup_frames)),
            "conf": float(conf),
            "iou": float(iou),
            "dataset_key": dataset_key,
            "backends": backends,
            "export_dir": str(export_root),
        },
        "source_info": source_info,
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
        "profiles": rows,
        "summary": summary,
        "warnings": warnings,
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "gate_status": gate_status,
                "profiles_pass": summary.get("profiles_pass", 0),
                "profiles_total": summary.get("profiles_total", 0),
                "best_backend_by_fps_mean": summary.get("best_backend_by_fps_mean", ""),
                "best_fps_mean": summary.get("best_fps_mean", 0.0),
                "warnings": len(warnings),
                "out_report": str(out_report),
                "out_md": str(out_md),
            },
            indent=2,
        )
    )
    if bool(args.fail_on_gate) and gate_status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
