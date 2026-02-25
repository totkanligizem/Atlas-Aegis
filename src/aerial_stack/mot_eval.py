from __future__ import annotations

import math
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MOTDetection:
    frame_id: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    conf: float = 1.0
    class_id: int = -1


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    if union <= 0.0:
        return 0.0
    return inter / union


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


def load_mot_txt(path: str | Path, *, is_gt: bool = False) -> dict[int, list[MOTDetection]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MOT file not found: {p}")

    by_frame: dict[int, list[MOTDetection]] = {}
    raw = p.read_text(encoding="utf-8", errors="ignore")
    for line in raw.splitlines():
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue

        if "," in ln:
            parts = [x.strip() for x in ln.split(",")]
        else:
            parts = ln.split()
        if len(parts) < 6:
            continue

        try:
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            class_id = int(float(parts[7])) if len(parts) > 7 else -1
        except ValueError:
            continue

        if frame_id <= 0 or track_id <= 0:
            continue
        if width <= 0 or height <= 0:
            continue
        if is_gt and len(parts) > 6 and conf <= 0.0:
            continue

        det = MOTDetection(
            frame_id=frame_id,
            track_id=track_id,
            bbox_xyxy=(left, top, left + width, top + height),
            conf=conf,
            class_id=class_id,
        )
        by_frame.setdefault(frame_id, []).append(det)

    return by_frame


def _evaluate_mot_single_threshold(
    gt_by_frame: dict[int, list[MOTDetection]],
    pred_by_frame: dict[int, list[MOTDetection]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    frame_ids = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    gt_total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    id_switches = 0
    iou_sum = 0.0

    # For AssA computation, count how often each GT track is associated with each predicted track.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_matched_counts: dict[int, int] = defaultdict(int)
    pred_matched_counts: dict[int, int] = defaultdict(int)

    prev_gt_to_pred: dict[int, int] = {}

    for frame_id in frame_ids:
        gt_list = gt_by_frame.get(frame_id, [])
        pr_list = pred_by_frame.get(frame_id, [])
        gt_total += len(gt_list)

        candidates: list[tuple[float, int, int]] = []
        for gi, g in enumerate(gt_list):
            for pi, p in enumerate(pr_list):
                if g.class_id >= 0 and p.class_id >= 0 and g.class_id != p.class_id:
                    continue
                iou = _iou_xyxy(g.bbox_xyxy, p.bbox_xyxy)
                if iou >= iou_threshold:
                    candidates.append((iou, gi, pi))

        candidates.sort(key=lambda x: x[0], reverse=True)
        used_gt: set[int] = set()
        used_pr: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        for iou, gi, pi in candidates:
            if gi in used_gt or pi in used_pr:
                continue
            used_gt.add(gi)
            used_pr.add(pi)
            matches.append((gi, pi, iou))

        tp = len(matches)
        fp = len(pr_list) - tp
        fn = len(gt_list) - tp

        tp_total += tp
        fp_total += max(0, fp)
        fn_total += max(0, fn)

        for gi, pi, iou in matches:
            g = gt_list[gi]
            p = pr_list[pi]
            iou_sum += iou
            prev = prev_gt_to_pred.get(g.track_id)
            if prev is not None and prev != p.track_id:
                id_switches += 1
            prev_gt_to_pred[g.track_id] = p.track_id

            pair_counts[(g.track_id, p.track_id)] += 1
            gt_matched_counts[g.track_id] += 1
            pred_matched_counts[p.track_id] += 1

    precision = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    recall = float(tp_total / gt_total) if gt_total > 0 else 0.0
    mota = float(1.0 - (fn_total + fp_total + id_switches) / gt_total) if gt_total > 0 else 0.0
    motp = float(iou_sum / tp_total) if tp_total > 0 else 0.0

    return {
        "frames": float(len(frame_ids)),
        "gt_detections": float(gt_total),
        "tp": float(tp_total),
        "fp": float(fp_total),
        "fn": float(fn_total),
        "id_switches": float(id_switches),
        "precision": precision,
        "recall": recall,
        "mota": mota,
        "motp_iou": motp,
        "_pair_counts": pair_counts,
        "_gt_matched_counts": gt_matched_counts,
        "_pred_matched_counts": pred_matched_counts,
    }


def _compute_assa_from_counts(
    *,
    tp_total: float,
    pair_counts: dict[tuple[int, int], int],
    gt_matched_counts: dict[int, int],
    pred_matched_counts: dict[int, int],
) -> float:
    if tp_total <= 0.0:
        return 0.0

    weighted_sum = 0.0
    for (gt_tid, pred_tid), pair_tp in pair_counts.items():
        fn_assoc = gt_matched_counts.get(gt_tid, 0) - pair_tp
        fp_assoc = pred_matched_counts.get(pred_tid, 0) - pair_tp
        denom = float(pair_tp + fn_assoc + fp_assoc)
        if denom <= 0.0:
            continue
        assoc = float(pair_tp) / denom
        weighted_sum += float(pair_tp) * assoc

    return weighted_sum / float(tp_total)


def evaluate_mot(
    gt_by_frame: dict[int, list[MOTDetection]],
    pred_by_frame: dict[int, list[MOTDetection]],
    *,
    iou_threshold: float = 0.5,
    include_hota: bool = False,
    hota_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    base = _evaluate_mot_single_threshold(
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        iou_threshold=float(iou_threshold),
    )
    out: dict[str, Any] = {
        "frames": float(base["frames"]),
        "gt_detections": float(base["gt_detections"]),
        "tp": float(base["tp"]),
        "fp": float(base["fp"]),
        "fn": float(base["fn"]),
        "id_switches": float(base["id_switches"]),
        "precision": float(base["precision"]),
        "recall": float(base["recall"]),
        "mota": float(base["mota"]),
        "motp_iou": float(base["motp_iou"]),
    }

    if not include_hota:
        return out

    thresholds = hota_thresholds
    if thresholds is None or len(thresholds) == 0:
        thresholds = [i / 100.0 for i in range(5, 100, 5)]

    curve: list[dict[str, float]] = []
    for alpha in thresholds:
        if alpha < 0.0 or alpha > 1.0:
            continue
        st = _evaluate_mot_single_threshold(
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            iou_threshold=float(alpha),
        )
        tp = float(st["tp"])
        fp = float(st["fp"])
        fn = float(st["fn"])
        det_denom = tp + fp + fn
        deta = (tp / det_denom) if det_denom > 0.0 else 0.0
        assa = _compute_assa_from_counts(
            tp_total=tp,
            pair_counts=st["_pair_counts"],
            gt_matched_counts=st["_gt_matched_counts"],
            pred_matched_counts=st["_pred_matched_counts"],
        )
        hota = math.sqrt(max(0.0, deta * assa))
        curve.append(
            {
                "alpha_iou": float(alpha),
                "hota": float(hota),
                "deta": float(deta),
                "assa": float(assa),
            }
        )

    if not curve:
        out.update(
            {
                "hota": 0.0,
                "deta": 0.0,
                "assa": 0.0,
                "hota_50": 0.0,
                "deta_50": 0.0,
                "assa_50": 0.0,
                "hota_alpha_count": 0.0,
                "hota_curve": [],
            }
        )
        return out

    curve.sort(key=lambda x: x["alpha_iou"])
    hota_mean = sum(x["hota"] for x in curve) / len(curve)
    deta_mean = sum(x["deta"] for x in curve) / len(curve)
    assa_mean = sum(x["assa"] for x in curve) / len(curve)

    closest_50 = min(curve, key=lambda x: abs(x["alpha_iou"] - 0.5))
    out.update(
        {
            "hota": float(hota_mean),
            "deta": float(deta_mean),
            "assa": float(assa_mean),
            "hota_50": float(closest_50["hota"]),
            "deta_50": float(closest_50["deta"]),
            "assa_50": float(closest_50["assa"]),
            "hota_alpha_count": float(len(curve)),
            "hota_curve": curve,
        }
    )
    return out


def load_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return [e for e in payload["events"] if isinstance(e, dict)]
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    return []


def evaluate_event_lead_time(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
) -> dict[str, float]:
    gt_start: dict[int, int] = {}
    for e in gt_events:
        try:
            tid = int(e.get("track_id", e.get("object_id", -1)))
        except Exception:
            continue
        if tid <= 0:
            continue
        frame = e.get("event_start_frame")
        if frame is None and str(e.get("type", "")) == "event_start":
            frame = e.get("frame_id")
        if frame is None:
            continue
        try:
            fi = int(frame)
        except Exception:
            continue
        prev = gt_start.get(tid)
        gt_start[tid] = fi if prev is None else min(prev, fi)

    pred_alarm: dict[int, int] = {}
    for e in pred_events:
        try:
            tid = int(e.get("track_id", e.get("object_id", -1)))
        except Exception:
            continue
        if tid <= 0:
            continue
        frame = e.get("alarm_frame")
        if frame is None and str(e.get("type", "")) in {"alarm_red", "alarm_yellow", "alarm"}:
            frame = e.get("frame_id")
        if frame is None:
            continue
        try:
            fi = int(frame)
        except Exception:
            continue
        prev = pred_alarm.get(tid)
        pred_alarm[tid] = fi if prev is None else min(prev, fi)

    common = sorted(set(gt_start.keys()) & set(pred_alarm.keys()))
    leads = [float(gt_start[tid] - pred_alarm[tid]) for tid in common]

    if not leads:
        return {
            "matched_tracks": 0.0,
            "lead_time_frames_median": 0.0,
            "lead_time_frames_p10": 0.0,
            "lead_time_frames_p90": 0.0,
            "lead_time_frames_mean": 0.0,
        }

    mean = float(sum(leads) / len(leads))
    return {
        "matched_tracks": float(len(leads)),
        "lead_time_frames_median": _percentile(leads, 0.5),
        "lead_time_frames_p10": _percentile(leads, 0.1),
        "lead_time_frames_p90": _percentile(leads, 0.9),
        "lead_time_frames_mean": mean,
    }
