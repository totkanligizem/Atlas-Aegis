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
from aerial_stack.mot_eval import MOTDetection, evaluate_mot
from aerial_stack.risk import BandThresholds, RiskWeights
from aerial_stack.track_risk import ROI, run_track_risk_ultralytics


def _parse_csv_floats(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(float(p)))
    return out


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


def _weights_from_cfg(cfg: dict[str, Any]) -> RiskWeights:
    w = cfg.get("weights", {})
    return RiskWeights(
        roi_dwell=float(w.get("roi_dwell", 1.4)),
        approach=float(w.get("approach", 1.0)),
        duration=float(w.get("duration", 0.8)),
        conf_mean=float(w.get("conf_mean", 0.6)),
        conf_stability=float(w.get("conf_stability", 0.5)),
        occlusion=float(w.get("occlusion", 0.3)),
        bias=float(cfg.get("bias", -1.8)),
    )


def _bands_from_cfg(cfg: dict[str, Any]) -> BandThresholds:
    b = cfg.get("bands", {})
    return BandThresholds(
        green_max=float(b.get("green_max", 39.999)),
        yellow_max=float(b.get("yellow_max", 69.999)),
    )


def _load_visdrone_gt(
    annotation_path: Path,
    *,
    max_frames: int,
) -> tuple[dict[int, list[MOTDetection]], dict[str, float]]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"GT annotation not found: {annotation_path}")

    by_frame: dict[int, list[MOTDetection]] = {}
    stats = {
        "rows_total": 0.0,
        "rows_kept": 0.0,
        "rows_dropped_invalid": 0.0,
        "rows_dropped_frame_limit": 0.0,
        "rows_dropped_category": 0.0,
    }
    valid_categories = set(range(1, 11))

    for ln in annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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

        track_id = _safe_int(parts[1], -1) + 1  # VisDrone MOT ids are 0-based.
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
        det = MOTDetection(
            frame_id=frame_id,
            track_id=track_id,
            bbox_xyxy=(left, top, left + width, top + height),
            conf=1.0,
            class_id=class_id,
        )
        by_frame.setdefault(frame_id, []).append(det)
        stats["rows_kept"] += 1.0

    return by_frame, stats


def _load_pred_from_jsonl(
    jsonl_path: Path,
    *,
    max_frames: int,
    min_track_age: int,
    min_conf: float,
    min_roi_dwell: int,
) -> tuple[dict[int, list[MOTDetection]], dict[str, float]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Prediction jsonl not found: {jsonl_path}")

    by_frame: dict[int, list[MOTDetection]] = {}
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
    }

    for ln in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            payload = json.loads(ln)
        except Exception:
            continue

        stats["frames_total"] += 1.0
        frame_id = _safe_int(payload.get("frame_id", -1), -1) + 1
        if frame_id <= 0:
            continue
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            continue
        if max_frames > 0 and frame_id > max_frames:
            stats["tracks_dropped_frame_limit"] += float(len(tracks))
            continue

        stats["frames_kept"] += 1.0
        stats["tracks_total"] += float(len(tracks))

        for t in tracks:
            track_id = _safe_int(t.get("track_id", -1), -1)
            class_id = _safe_int(t.get("class_id", -1), -1)
            conf = _safe_float(t.get("conf", 0.0), 0.0)
            age_frames = _safe_int(t.get("age_frames", 0), 0)
            roi_dwell = _safe_int(t.get("roi_dwell", 0), 0)
            if age_frames < min_track_age:
                stats["tracks_dropped_low_age"] += 1.0
                continue
            if conf < min_conf:
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
            if track_id <= 0 or (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
                stats["tracks_dropped_invalid"] += 1.0
                continue

            det = MOTDetection(
                frame_id=frame_id,
                track_id=track_id,
                bbox_xyxy=(x1, y1, x2, y2),
                conf=conf,
                class_id=class_id,
            )
            by_frame.setdefault(frame_id, []).append(det)
            stats["tracks_kept"] += 1.0

    return by_frame, stats


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    mm = row["mot_metrics"]
    mota = float(mm.get("mota", 0.0))
    precision = float(mm.get("precision", 0.0))
    id_switches = float(mm.get("id_switches", 0.0))
    return mota, precision, -id_switches


def _to_tag(conf: float, iou: float, ttl: int) -> str:
    cs = f"{conf:.3f}".replace(".", "p")
    is_ = f"{iou:.2f}".replace(".", "p")
    return f"c{cs}_i{is_}_ttl{ttl}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inference-config", default="configs/inference_visdrone_mot_val.yaml")
    p.add_argument(
        "--gt-annotation",
        default="data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt",
    )
    p.add_argument("--conf-values", default="0.10,0.15,0.20")
    p.add_argument("--iou-values", default="0.60,0.70")
    p.add_argument("--ttl-values", default="45,60")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--eval-iou-threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="logs/mot_sweep_visdrone_val")
    p.add_argument("--report-out", default="reports/mot_tuning_sweep_visdrone_val_report.json")
    p.add_argument("--min-track-age", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--min-roi-dwell", type=int, default=0)
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

    inference_cfg = load_yaml(args.inference_config)
    risk_cfg = load_yaml(str(inference_cfg.get("risk_config", "configs/risk.yaml")))

    source = str(inference_cfg.get("source", ""))
    model_path = str(inference_cfg.get("model", "weights/yolov8n.pt"))
    tracker_path = str(inference_cfg.get("tracker", "configs/trackers/bytetrack.yaml"))
    imgsz = int(inference_cfg.get("imgsz", 960))
    device = str(inference_cfg.get("device", "")).strip() or None
    min_frames_in_roi = int(risk_cfg.get("event", {}).get("min_frames_in_roi", 10))
    roi_cfg = inference_cfg.get("roi", {})
    roi = ROI(
        x1=float(roi_cfg.get("x1", 0.2)),
        y1=float(roi_cfg.get("y1", 0.2)),
        x2=float(roi_cfg.get("x2", 0.8)),
        y2=float(roi_cfg.get("y2", 0.8)),
    )
    weights = _weights_from_cfg(risk_cfg)
    bands = _bands_from_cfg(risk_cfg)
    dataset_key = str(args.dataset_key or inference_cfg.get("dataset_key", "")).strip()

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
                source="mot_sweep",
                requested_spend_usd=float(args.budget_spend_usd),
                is_api_call=bool(args.budget_is_api_call),
            )
        except Exception as exc:
            print(f"error: budget guard failed: {exc}")
            return 2
        if bool(budget_guard.get("blocked", False)):
            print(f"error: budget guard blocked run: {json.dumps(budget_guard, indent=2)}")
            return 3

    gt_by_frame, gt_stats = _load_visdrone_gt(
        Path(args.gt_annotation),
        max_frames=int(args.max_frames),
    )

    conf_values = _parse_csv_floats(args.conf_values)
    iou_values = _parse_csv_floats(args.iou_values)
    ttl_values = _parse_csv_ints(args.ttl_values)
    if not conf_values or not iou_values or not ttl_values:
        raise ValueError("conf/iou/ttl value lists must be non-empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    total = len(conf_values) * len(iou_values) * len(ttl_values)
    idx = 0
    for confv in conf_values:
        for iouv in iou_values:
            for ttl in ttl_values:
                idx += 1
                tag = _to_tag(confv, iouv, ttl)
                out_jsonl = out_dir / f"track_{tag}.jsonl"
                print(
                    f"[mot_sweep] {idx}/{total} conf={confv:.3f} iou={iouv:.2f} ttl={ttl} tag={tag}",
                    flush=True,
                )

                summary = run_track_risk_ultralytics(
                    source=source,
                    model_path=model_path,
                    tracker_path=tracker_path,
                    conf=float(confv),
                    iou=float(iouv),
                    imgsz=imgsz,
                    device=device,
                    max_frames=int(args.max_frames),
                    roi=roi,
                    min_frames_in_roi=min_frames_in_roi,
                    track_ttl_frames=int(ttl),
                    weights=weights,
                    bands=bands,
                    out_jsonl=str(out_jsonl),
                )

                pred_by_frame, pred_stats = _load_pred_from_jsonl(
                    out_jsonl,
                    max_frames=int(args.max_frames),
                    min_track_age=max(1, int(args.min_track_age)),
                    min_conf=max(0.0, float(args.min_conf)),
                    min_roi_dwell=max(0, int(args.min_roi_dwell)),
                )
                mot_metrics = evaluate_mot(
                    gt_by_frame=gt_by_frame,
                    pred_by_frame=pred_by_frame,
                    iou_threshold=float(args.eval_iou_threshold),
                )

                row = {
                    "tag": tag,
                    "params": {
                        "conf": float(confv),
                        "iou": float(iouv),
                        "track_ttl_frames": int(ttl),
                    },
                    "summary": {
                        "tracking_backend": summary.get("tracking_backend", ""),
                        "num_frames": int(summary.get("num_frames", 0)),
                        "event_count": int(summary.get("event_count", 0)),
                        "final_active_tracks": int(summary.get("final_active_tracks", 0)),
                        "output_jsonl": str(out_jsonl),
                    },
                    "pred_stats": pred_stats,
                    "mot_metrics": mot_metrics,
                }
                runs.append(row)

    runs_sorted = sorted(runs, key=_rank_key, reverse=True)
    best = runs_sorted[0] if runs_sorted else {}
    report = {
        "status": "SUCCESS",
        "inputs": {
            "inference_config": args.inference_config,
            "gt_annotation": args.gt_annotation,
            "max_frames": int(args.max_frames),
            "eval_iou_threshold": float(args.eval_iou_threshold),
            "conf_values": conf_values,
            "iou_values": iou_values,
            "ttl_values": ttl_values,
            "dataset_key": dataset_key,
            "min_track_age": max(1, int(args.min_track_age)),
            "min_conf": max(0.0, float(args.min_conf)),
            "min_roi_dwell": max(0, int(args.min_roi_dwell)),
        },
        "legal_gate": legal_gate,
        "budget_guard": budget_guard,
        "gt_stats": gt_stats,
        "runs": runs,
        "best_run": best,
        "ranking_rule": "sort by (mota desc, precision desc, id_switches asc)",
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_run": best}, indent=2))
    print(f"wrote report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
