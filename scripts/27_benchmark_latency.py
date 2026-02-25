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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
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


def _looks_like_local_path(model: str) -> bool:
    return "/" in model or "\\" in model or model.endswith(".pt")


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
        raise RuntimeError("opencv-python is required for real latency benchmark mode.") from exc

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


def _gate_for_profile(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
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


def _synthetic_series(
    *,
    profile_name: str,
    model_path: str,
    imgsz: int,
    device: str,
    measured_frames: int,
) -> tuple[list[float], list[float]]:
    key = f"{profile_name}|{model_path}|{imgsz}|{device}"
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)

    size_factor = _model_size_factor(model_path)
    img_factor = (max(32, int(imgsz)) / 640.0) ** 1.05
    if device == "mps":
        device_factor = 0.60
    elif device == "cpu":
        device_factor = 1.00
    else:
        device_factor = 0.85

    base_ms = 32.0 * size_factor * img_factor * device_factor
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


def _base_profile_result(
    profile: dict[str, Any],
    *,
    num_frames_total: int,
    warmup_frames: int,
) -> dict[str, Any]:
    model_path = str(profile.get("model", "")).strip()
    device_requested = str(profile.get("device", "")).strip()
    return {
        "name": str(profile.get("name", "profile")),
        "required": bool(profile.get("required", True)),
        "status": "ERROR",
        "model": model_path,
        "imgsz": max(32, _safe_int(profile.get("imgsz", 640), 640)),
        "device_requested": device_requested,
        "device_effective": "",
        "allow_missing_model": bool(profile.get("allow_missing_model", False)),
        "num_frames_total": int(num_frames_total),
        "warmup_frames": int(max(0, warmup_frames)),
        "num_frames_measured": 0,
        "thresholds": {
            "target_fps": max(0.0, _safe_float(profile.get("target_fps", 0.0), 0.0)),
            "target_latency_ms": max(
                0.0, _safe_float(profile.get("target_latency_ms", 0.0), 0.0)
            ),
            "target_p95_latency_ms": max(
                0.0, _safe_float(profile.get("target_p95_latency_ms", 0.0), 0.0)
            ),
        },
        "gate": {"pass": False, "checks": []},
        "metrics": {},
        "reason": "",
        "error": "",
    }


def _run_profile_dry(
    profile: dict[str, Any],
    *,
    num_frames_total: int,
    warmup_frames: int,
) -> dict[str, Any]:
    row = _base_profile_result(profile, num_frames_total=num_frames_total, warmup_frames=warmup_frames)
    device_effective, device_issue = _resolve_device(str(row["device_requested"]))
    row["device_effective"] = str(device_effective or "auto")

    if device_issue:
        row["status"] = "SKIPPED" if not bool(row["required"]) else "ERROR"
        row["reason"] = device_issue
        return row

    measured = int(num_frames_total) - min(int(warmup_frames), max(0, int(num_frames_total) - 1))
    measured = max(1, measured)
    row["num_frames_measured"] = measured

    latencies, detections = _synthetic_series(
        profile_name=str(row["name"]),
        model_path=str(row["model"]),
        imgsz=int(row["imgsz"]),
        device=str(device_effective or "auto"),
        measured_frames=measured,
    )
    row["metrics"] = _build_metrics(latencies, detections)
    row["gate"] = _gate_for_profile(row["metrics"], row["thresholds"])
    row["status"] = "PASS" if bool(row["gate"].get("pass", False)) else "FAIL"
    return row


def _run_profile_real(
    profile: dict[str, Any],
    *,
    frames: list[Any],
    warmup_frames: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    row = _base_profile_result(profile, num_frames_total=len(frames), warmup_frames=warmup_frames)
    model_path = str(row["model"])
    required = bool(row["required"])
    allow_missing = bool(row["allow_missing_model"])

    if _looks_like_local_path(model_path) and model_path and not Path(model_path).exists():
        if allow_missing and not required:
            row["status"] = "SKIPPED"
            row["reason"] = f"Model path missing and allow_missing_model=true: {model_path}"
            return row
        row["status"] = "ERROR"
        row["error"] = f"Model path does not exist: {model_path}"
        return row

    device_effective, device_issue = _resolve_device(str(row["device_requested"]))
    row["device_effective"] = str(device_effective or "auto")
    if device_issue:
        row["status"] = "SKIPPED" if not required else "ERROR"
        row["reason"] = device_issue
        return row

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"ultralytics import failed: {exc}"
        return row

    try:
        model = YOLO(model_path)
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"YOLO model load failed: {exc}"
        return row

    if not frames:
        row["status"] = "ERROR"
        row["error"] = "No frames loaded from source."
        return row

    warmup_eff = min(int(warmup_frames), max(0, len(frames) - 1))
    latencies_ms: list[float] = []
    detections: list[float] = []

    for idx, frame in enumerate(frames):
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(conf),
            "iou": float(iou),
            "imgsz": int(row["imgsz"]),
            "verbose": False,
        }
        if device_effective:
            kwargs["device"] = device_effective
        t0 = time.perf_counter()
        try:
            result = model.predict(**kwargs)
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = f"Inference failed at frame {idx}: {exc}"
            return row
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
        row["status"] = "ERROR"
        row["error"] = "No measured frames remain after warmup."
        return row

    row["num_frames_measured"] = len(latencies_ms)
    row["metrics"] = _build_metrics(latencies_ms, detections)
    row["gate"] = _gate_for_profile(row["metrics"], row["thresholds"])
    row["status"] = "PASS" if bool(row["gate"].get("pass", False)) else "FAIL"
    return row


def _make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_rows = [r for r in rows if bool(r.get("required", False))]
    required_profiles = len(required_rows)
    required_passed = sum(1 for r in required_rows if str(r.get("status", "")) == "PASS")
    overall_gate_pass = required_passed == required_profiles if required_profiles > 0 else True

    best_profile = ""
    best_fps = 0.0
    for r in rows:
        if str(r.get("status", "")) != "PASS":
            continue
        m = r.get("metrics", {})
        if not isinstance(m, dict):
            continue
        fps = float(m.get("fps_mean", 0.0) or 0.0)
        if fps > best_fps:
            best_fps = fps
            best_profile = str(r.get("name", ""))

    return {
        "profiles_total": len(rows),
        "required_profiles": required_profiles,
        "required_passed": required_passed,
        "overall_gate_pass": bool(overall_gate_pass),
        "best_profile_by_fps_mean": best_profile,
        "best_fps_mean": float(best_fps),
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Latency / FPS Benchmark")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- mode: `{report.get('mode', 'unknown')}`")
    lines.append(f"- gate_status: `{report.get('gate_status', 'FAIL')}`")
    summary = report.get("summary", {})
    lines.append(
        f"- required_passed: `{summary.get('required_passed', 0)}` / "
        f"`{summary.get('required_profiles', 0)}`"
    )
    lines.append(
        f"- best_profile_by_fps_mean: `{summary.get('best_profile_by_fps_mean', '')}` "
        f"({float(summary.get('best_fps_mean', 0.0) or 0.0):.2f} FPS)"
    )
    lines.append("")
    lines.append("## Profiles")
    lines.append(
        "| profile | status | required | device | fps_mean | latency_median_ms | "
        "latency_p95_ms | target_fps | target_latency_ms | target_p95_latency_ms |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in report.get("profiles", []):
        if not isinstance(row, dict):
            continue
        m = row.get("metrics", {})
        if not isinstance(m, dict):
            m = {}
        t = row.get("thresholds", {})
        if not isinstance(t, dict):
            t = {}
        lines.append(
            f"| {row.get('name', '')} | {row.get('status', '')} | "
            f"{1 if bool(row.get('required', False)) else 0} | "
            f"{row.get('device_effective', row.get('device_requested', ''))} | "
            f"{float(m.get('fps_mean', 0.0) or 0.0):.2f} | "
            f"{float(m.get('latency_median_ms', 0.0) or 0.0):.2f} | "
            f"{float(m.get('latency_p95_ms', 0.0) or 0.0):.2f} | "
            f"{float(t.get('target_fps', 0.0) or 0.0):.2f} | "
            f"{float(t.get('target_latency_ms', 0.0) or 0.0):.2f} | "
            f"{float(t.get('target_p95_latency_ms', 0.0) or 0.0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latency_benchmark.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", default="", help="Optional source override.")
    p.add_argument("--max-frames", type=int, default=-1, help="Optional max_frames override.")
    p.add_argument("--frame-stride", type=int, default=-1, help="Optional frame_stride override.")
    p.add_argument("--warmup-frames", type=int, default=-1, help="Optional warmup_frames override.")
    p.add_argument("--out-report", default="", help="Optional output report override.")
    p.add_argument("--out-md", default="", help="Optional output markdown override.")
    p.add_argument("--fail-on-gate", action="store_true", help="Return code 2 when gate fails.")
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
        raise ValueError("source is required in latency benchmark config or --source.")
    max_frames = _safe_int(args.max_frames if args.max_frames >= 0 else cfg.get("max_frames", 120), 120)
    frame_stride = _safe_int(args.frame_stride if args.frame_stride >= 0 else cfg.get("frame_stride", 1), 1)
    warmup_frames = _safe_int(args.warmup_frames if args.warmup_frames >= 0 else cfg.get("warmup_frames", 5), 5)
    conf = _safe_float(cfg.get("conf", 0.1), 0.1)
    iou = _safe_float(cfg.get("iou", 0.7), 0.7)
    profiles = cfg.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("latency benchmark config must include a non-empty 'profiles' list.")

    out_report = Path(str(args.out_report or cfg.get("output_report", "reports/latency_benchmark_report.json")))
    out_md = Path(str(args.out_md or cfg.get("output_md", "reports/latency_benchmark_report.md")))

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
                source="latency_benchmark",
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
    total = len([x for x in profiles if isinstance(x, dict)])
    done = 0
    for raw in profiles:
        if not isinstance(raw, dict):
            continue
        done += 1
        name = str(raw.get("name", f"profile_{done}"))
        print(f"[latency-benchmark] {done}/{total} profile={name} mode={mode}", flush=True)
        if args.dry_run:
            row = _run_profile_dry(
                raw,
                num_frames_total=int(source_info.get("frames_total", 0)),
                warmup_frames=warmup_frames,
            )
        else:
            row = _run_profile_real(
                raw,
                frames=frames,
                warmup_frames=warmup_frames,
                conf=conf,
                iou=iou,
            )
        rows.append(row)

    summary = _make_summary(rows)
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
        },
        "source_info": source_info,
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
        "profiles": rows,
        "summary": summary,
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
                "profiles_total": summary.get("profiles_total", 0),
                "required_passed": summary.get("required_passed", 0),
                "required_profiles": summary.get("required_profiles", 0),
                "best_profile_by_fps_mean": summary.get("best_profile_by_fps_mean", ""),
                "best_fps_mean": summary.get("best_fps_mean", 0.0),
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
