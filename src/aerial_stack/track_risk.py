from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk import BandThresholds, RiskFeatures, RiskWeights, band_from_score, score_risk


@dataclass
class ROI:
    # Normalized [0, 1] coordinates.
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
        rx1 = self.x1 * frame_w
        ry1 = self.y1 * frame_h
        rx2 = self.x2 * frame_w
        ry2 = self.y2 * frame_h
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    age_frames: int = 0
    conf_history: list[float] = field(default_factory=list)
    prev_area: float | None = None
    area_slope: float = 0.0
    roi_dwell: int = 0
    occlusion_count: int = 0
    event_start_frame: int | None = None
    alarm_emitted: bool = False


def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_pstdev(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
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


def _greedy_iou_assignment(
    prev_boxes: dict[int, tuple[float, float, float, float]],
    det_boxes: list[tuple[float, float, float, float]],
    *,
    min_iou: float = 0.3,
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for tid, pb in prev_boxes.items():
        for det_idx, db in enumerate(det_boxes):
            iou = _iou_xyxy(pb, db)
            if iou >= min_iou:
                candidates.append((iou, tid, det_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    det_to_track: dict[int, int] = {}
    for _, tid, det_idx in candidates:
        if tid in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(tid)
        used_dets.add(det_idx)
        det_to_track[det_idx] = tid
    return det_to_track


def _frame_line(
    frame_id: int,
    tracks: list[dict[str, Any]],
    out_path: Path | None,
) -> None:
    if out_path is None:
        return
    payload = {"frame_id": frame_id, "tracks": tracks}
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _build_track_payload(
    state: TrackState,
    cls_id: int,
    conf: float,
    bbox_xyxy: list[float],
    risk_score: float,
    band: str,
) -> dict[str, Any]:
    return {
        "track_id": state.track_id,
        "class_id": cls_id,
        "conf": round(conf, 4),
        "bbox_xyxy": [round(v, 3) for v in bbox_xyxy],
        "age_frames": state.age_frames,
        "roi_dwell": state.roi_dwell,
        "occlusion_count": state.occlusion_count,
        "risk_score": round(risk_score, 3),
        "band": band,
    }


def run_track_risk_dry(
    num_frames: int,
    roi: ROI,
    min_frames_in_roi: int,
    track_ttl_frames: int,
    weights: RiskWeights,
    bands: BandThresholds,
    out_jsonl: str | None = None,
) -> dict[str, Any]:
    random.seed(17)
    frame_w, frame_h = 1280, 720
    states: dict[int, TrackState] = {}
    events: list[dict[str, Any]] = []
    out_path = Path(out_jsonl) if out_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    for frame_id in range(num_frames):
        active_ids = [1, 2] if frame_id < num_frames * 0.8 else [1]
        frame_tracks: list[dict[str, Any]] = []
        seen: set[int] = set()

        for tid in active_ids:
            seen.add(tid)
            x = 300 + 4 * frame_id + (tid - 1) * 120
            y = 200 + (tid - 1) * 50
            w = 60 + (frame_id % 15)
            h = 40 + (tid * 2)
            x1, y1, x2, y2 = x, y, x + w, y + h
            conf = max(0.2, min(0.98, 0.55 + 0.003 * frame_id - 0.02 * (tid - 1)))

            state = states.get(tid)
            if state is None:
                state = TrackState(track_id=tid, first_frame=frame_id, last_frame=frame_id)
                states[tid] = state

            if frame_id > state.last_frame + 1:
                state.occlusion_count += frame_id - state.last_frame - 1

            state.last_frame = frame_id
            state.age_frames += 1
            state.conf_history.append(conf)
            if len(state.conf_history) > 30:
                state.conf_history = state.conf_history[-30:]

            area = _bbox_area(x1, y1, x2, y2)
            if state.prev_area and state.prev_area > 0.0:
                state.area_slope = (area - state.prev_area) / state.prev_area
            else:
                state.area_slope = 0.0
            state.prev_area = area

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            in_roi = roi.contains(cx, cy, frame_w=frame_w, frame_h=frame_h)
            state.roi_dwell = state.roi_dwell + 1 if in_roi else 0

            if state.event_start_frame is None and state.roi_dwell >= min_frames_in_roi:
                state.event_start_frame = frame_id
                events.append({"type": "event_start", "track_id": tid, "frame_id": frame_id})

            features = RiskFeatures(
                duration_frames=state.age_frames,
                conf_mean=_safe_mean(state.conf_history),
                conf_std=_safe_pstdev(state.conf_history),
                bbox_area_slope=state.area_slope,
                roi_dwell=state.roi_dwell,
                occlusion_count=state.occlusion_count,
            )
            risk_score = score_risk(features, weights)
            band = band_from_score(risk_score, bands)

            if state.event_start_frame is not None and (not state.alarm_emitted) and band == "RED":
                state.alarm_emitted = True
                events.append({"type": "alarm_red", "track_id": tid, "frame_id": frame_id})

            frame_tracks.append(
                _build_track_payload(
                    state=state,
                    cls_id=0,
                    conf=conf,
                    bbox_xyxy=[x1, y1, x2, y2],
                    risk_score=risk_score,
                    band=band,
                )
            )

        stale_ids: list[int] = []
        for tid, st in states.items():
            if tid not in seen:
                st.occlusion_count += 1
                if frame_id - st.last_frame > track_ttl_frames:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del states[tid]

        _frame_line(frame_id=frame_id, tracks=frame_tracks, out_path=out_path)

    return {
        "mode": "dry_run",
        "num_frames": num_frames,
        "events": events,
        "event_count": len(events),
        "final_active_tracks": len(states),
        "output_jsonl": out_jsonl,
    }


def run_track_risk_ultralytics(
    source: str,
    model_path: str,
    tracker_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_frames: int,
    roi: ROI,
    min_frames_in_roi: int,
    track_ttl_frames: int,
    weights: RiskWeights,
    bands: BandThresholds,
    out_jsonl: str | None,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics not installed. Install dependencies or run with --dry-run."
        ) from exc

    model = YOLO(model_path)
    out_path = Path(out_jsonl) if out_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    states: dict[int, TrackState] = {}
    track_bboxes: dict[int, tuple[float, float, float, float]] = {}
    events: list[dict[str, Any]] = []
    frame_count = 0
    next_track_id = 1
    tracking_backend = "ultralytics_track"

    track_kwargs: dict[str, Any] = {
        "source": source,
        "tracker": tracker_path,
        "conf": conf,
        "iou": iou,
        "persist": True,
        "stream": True,
        "verbose": False,
        "imgsz": imgsz,
    }
    if device:
        track_kwargs["device"] = device

    tracking_supported = True
    try:
        result_stream = model.track(**track_kwargs)
    except ModuleNotFoundError as exc:
        # Ultralytics ByteTrack requires `lap`; if unavailable, fall back to
        # predict + lightweight IoU-based ID propagation.
        if "lap" not in str(exc).lower():
            raise
        tracking_supported = False
        tracking_backend = "ultralytics_predict_iou_fallback"
        predict_kwargs: dict[str, Any] = {
            "source": source,
            "conf": conf,
            "iou": iou,
            "stream": True,
            "verbose": False,
            "imgsz": imgsz,
        }
        if device:
            predict_kwargs["device"] = device
        result_stream = model.predict(**predict_kwargs)

    for frame_id, result in enumerate(result_stream):
        if max_frames > 0 and frame_id >= max_frames:
            break

        frame_count += 1
        frame_h, frame_w = result.orig_shape
        frame_tracks: list[dict[str, Any]] = []
        seen: set[int] = set()

        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.cpu().tolist()
            tids: list[int] = []

            if tracking_supported:
                ids_obj = getattr(boxes, "id", None)
                if ids_obj is not None:
                    tids = [int(x) for x in ids_obj.int().cpu().tolist()]
                    if tids:
                        next_track_id = max(next_track_id, max(tids) + 1)
            else:
                det_boxes = [
                    (
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                    )
                    for b in xyxy
                ]
                prev_boxes = {
                    tid: bb
                    for tid, bb in track_bboxes.items()
                    if tid in states and (frame_id - states[tid].last_frame) <= track_ttl_frames
                }
                det_to_track = _greedy_iou_assignment(prev_boxes, det_boxes, min_iou=0.3)
                tids = []
                for det_idx in range(len(det_boxes)):
                    tid = det_to_track.get(det_idx)
                    if tid is None:
                        tid = next_track_id
                        next_track_id += 1
                    tids.append(int(tid))

            for i, tid in enumerate(tids):
                seen.add(tid)
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                confv = float(confs[i])
                clsv = int(clss[i])

                state = states.get(tid)
                if state is None:
                    state = TrackState(track_id=tid, first_frame=frame_id, last_frame=frame_id)
                    states[tid] = state

                if frame_id > state.last_frame + 1:
                    state.occlusion_count += frame_id - state.last_frame - 1

                state.last_frame = frame_id
                state.age_frames += 1
                state.conf_history.append(confv)
                if len(state.conf_history) > 30:
                    state.conf_history = state.conf_history[-30:]

                area = _bbox_area(x1, y1, x2, y2)
                if state.prev_area and state.prev_area > 0.0:
                    state.area_slope = (area - state.prev_area) / state.prev_area
                else:
                    state.area_slope = 0.0
                state.prev_area = area
                track_bboxes[tid] = (x1, y1, x2, y2)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                in_roi = roi.contains(cx, cy, frame_w=frame_w, frame_h=frame_h)
                state.roi_dwell = state.roi_dwell + 1 if in_roi else 0

                if state.event_start_frame is None and state.roi_dwell >= min_frames_in_roi:
                    state.event_start_frame = frame_id
                    events.append(
                        {"type": "event_start", "track_id": tid, "frame_id": frame_id}
                    )

                features = RiskFeatures(
                    duration_frames=state.age_frames,
                    conf_mean=_safe_mean(state.conf_history),
                    conf_std=_safe_pstdev(state.conf_history),
                    bbox_area_slope=state.area_slope,
                    roi_dwell=state.roi_dwell,
                    occlusion_count=state.occlusion_count,
                )
                risk_score = score_risk(features, weights)
                band = band_from_score(risk_score, bands)

                if state.event_start_frame is not None and (not state.alarm_emitted) and band == "RED":
                    state.alarm_emitted = True
                    events.append(
                        {"type": "alarm_red", "track_id": tid, "frame_id": frame_id}
                    )

                frame_tracks.append(
                    _build_track_payload(
                        state=state,
                        cls_id=clsv,
                        conf=confv,
                        bbox_xyxy=[x1, y1, x2, y2],
                        risk_score=risk_score,
                        band=band,
                    )
                )

        stale_ids: list[int] = []
        for tid, st in states.items():
            if tid not in seen:
                st.occlusion_count += 1
                if frame_id - st.last_frame > track_ttl_frames:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del states[tid]
            track_bboxes.pop(tid, None)

        _frame_line(frame_id=frame_id, tracks=frame_tracks, out_path=out_path)

    return {
        "mode": "ultralytics",
        "tracking_backend": tracking_backend,
        "source": source,
        "num_frames": frame_count,
        "events": events,
        "event_count": len(events),
        "final_active_tracks": len(states),
        "output_jsonl": out_jsonl,
        "model": model_path,
        "tracker": tracker_path if tracking_supported else "iou_fallback",
    }
