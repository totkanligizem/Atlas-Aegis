#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.mot_eval import MOTDetection, load_mot_txt


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


def _analyze_sequence(
    *,
    label: str,
    gt_by_frame: dict[int, list[MOTDetection]],
    pred_by_frame: dict[int, list[MOTDetection]],
    iou_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    frame_ids = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    prev_gt_to_pred: dict[int, int] = {}
    frame_rows: list[dict[str, float]] = []
    idsw_by_gt_track: dict[int, int] = {}

    totals = {
        "frames": 0.0,
        "gt_detections": 0.0,
        "pred_detections": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "id_switches": 0.0,
    }

    for frame_id in frame_ids:
        gt_list = gt_by_frame.get(frame_id, [])
        pr_list = pred_by_frame.get(frame_id, [])

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
        frame_idsw = 0.0

        for iou, gi, pi in candidates:
            if gi in used_gt or pi in used_pr:
                continue
            used_gt.add(gi)
            used_pr.add(pi)
            matches.append((gi, pi, iou))

        tp = float(len(matches))
        fp = float(max(0, len(pr_list) - len(matches)))
        fn = float(max(0, len(gt_list) - len(matches)))

        for gi, pi, _iou in matches:
            g = gt_list[gi]
            p = pr_list[pi]
            prev = prev_gt_to_pred.get(g.track_id)
            if prev is not None and prev != p.track_id:
                frame_idsw += 1.0
                idsw_by_gt_track[g.track_id] = idsw_by_gt_track.get(g.track_id, 0) + 1
            prev_gt_to_pred[g.track_id] = p.track_id

        totals["frames"] += 1.0
        totals["gt_detections"] += float(len(gt_list))
        totals["pred_detections"] += float(len(pr_list))
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["id_switches"] += frame_idsw

        score = fn * 3.0 + fp * 1.0 + frame_idsw * 5.0
        frame_rows.append(
            {
                "frame_id": float(frame_id),
                "gt_count": float(len(gt_list)),
                "pred_count": float(len(pr_list)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "id_switches": frame_idsw,
                "score": score,
            }
        )

    fps = [r["fp"] for r in frame_rows]
    fns = [r["fn"] for r in frame_rows]
    idsw = [r["id_switches"] for r in frame_rows]
    worst_frames = sorted(frame_rows, key=lambda r: r["score"], reverse=True)[: max(1, top_k)]
    top_idsw_tracks = sorted(idsw_by_gt_track.items(), key=lambda kv: kv[1], reverse=True)[:10]

    precision = 0.0
    if (totals["tp"] + totals["fp"]) > 0.0:
        precision = totals["tp"] / (totals["tp"] + totals["fp"])
    recall = 0.0
    if totals["gt_detections"] > 0.0:
        recall = totals["tp"] / totals["gt_detections"]
    mota = 0.0
    if totals["gt_detections"] > 0.0:
        mota = 1.0 - (
            (totals["fn"] + totals["fp"] + totals["id_switches"]) / totals["gt_detections"]
        )

    return {
        "label": label,
        "summary": {
            "frames": totals["frames"],
            "gt_detections": totals["gt_detections"],
            "pred_detections": totals["pred_detections"],
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
            "id_switches": totals["id_switches"],
            "precision": precision,
            "recall": recall,
            "mota": mota,
        },
        "distribution": {
            "fp_mean": float(sum(fps) / len(fps)) if fps else 0.0,
            "fp_p95": _percentile(fps, 0.95),
            "fn_mean": float(sum(fns) / len(fns)) if fns else 0.0,
            "fn_p95": _percentile(fns, 0.95),
            "idsw_mean": float(sum(idsw) / len(idsw)) if idsw else 0.0,
            "idsw_p95": _percentile(idsw, 0.95),
            "idsw_frames": float(sum(1 for x in idsw if x > 0.0)),
        },
        "worst_frames": worst_frames,
        "top_idsw_tracks": [{"track_id": int(k), "id_switches": int(v)} for k, v in top_idsw_tracks],
    }


def _build_md(report: dict[str, Any]) -> str:
    s1 = report["seq1"]
    s2 = report["seq2"]
    d = report["delta_seq2_minus_seq1"]
    lines: list[str] = []
    lines.append("# MOT Error Slice Comparison")
    lines.append("")
    lines.append("## Sequence Summary")
    lines.append(
        f"- {s1['label']}: MOTA={s1['summary']['mota']:.4f}, "
        f"precision={s1['summary']['precision']:.4f}, recall={s1['summary']['recall']:.4f}, "
        f"IDSW={s1['summary']['id_switches']:.0f}, FP={s1['summary']['fp']:.0f}, FN={s1['summary']['fn']:.0f}"
    )
    lines.append(
        f"- {s2['label']}: MOTA={s2['summary']['mota']:.4f}, "
        f"precision={s2['summary']['precision']:.4f}, recall={s2['summary']['recall']:.4f}, "
        f"IDSW={s2['summary']['id_switches']:.0f}, FP={s2['summary']['fp']:.0f}, FN={s2['summary']['fn']:.0f}"
    )
    lines.append("")
    lines.append("## seq2 - seq1 delta")
    lines.append(
        f"- MOTA: {d['mota']:+.4f}, precision: {d['precision']:+.4f}, recall: {d['recall']:+.4f}, "
        f"IDSW: {d['id_switches']:+.0f}, FP: {d['fp']:+.0f}, FN: {d['fn']:+.0f}"
    )
    lines.append(
        f"- Burst indicators (mean/p95): FP {s2['distribution']['fp_mean']:.2f}/{s2['distribution']['fp_p95']:.2f}, "
        f"FN {s2['distribution']['fn_mean']:.2f}/{s2['distribution']['fn_p95']:.2f}, "
        f"IDSW {s2['distribution']['idsw_mean']:.2f}/{s2['distribution']['idsw_p95']:.2f}"
    )
    lines.append("")
    lines.append(f"## Top Problem Frames ({s2['label']})")
    lines.append("| rank | frame | score | fp | fn | idsw |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(s2["worst_frames"], 1):
        lines.append(
            f"| {i} | {int(row['frame_id'])} | {row['score']:.1f} | "
            f"{row['fp']:.0f} | {row['fn']:.0f} | {row['id_switches']:.0f} |"
        )
    lines.append("")
    lines.append(f"## Top ID Switch Tracks ({s2['label']})")
    lines.append("| rank | gt_track_id | id_switches |")
    lines.append("|---:|---:|---:|")
    for i, row in enumerate(s2["top_idsw_tracks"], 1):
        lines.append(f"| {i} | {row['track_id']} | {row['id_switches']} |")
    lines.append("")
    lines.append("## Recommended Next Tuning Focus")
    lines.append("- Prioritize reducing FP bursts in top-score frames (conf/iou gate retune around these slices).")
    lines.append("- Add short track suppression / minimum age before reporting to reduce unstable ID switches.")
    lines.append("- Re-check detector confidence calibration on seq2-like conditions before next MOT sweep.")
    return "\n".join(lines) + "\n"


def _delta(seq1: dict[str, Any], seq2: dict[str, Any]) -> dict[str, float]:
    klist = ["mota", "precision", "recall", "id_switches", "fp", "fn"]
    out: dict[str, float] = {}
    for k in klist:
        out[k] = float(seq2["summary"].get(k, 0.0)) - float(seq1["summary"].get(k, 0.0))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq1-gt", default="data/mot/gt.txt")
    p.add_argument("--seq1-pred", default="data/mot/pred.txt")
    p.add_argument("--seq1-label", default="visdrone_val_seq1")
    p.add_argument("--seq2-gt", default="data/mot/gt_seq2.txt")
    p.add_argument("--seq2-pred", default="data/mot/pred_seq2.txt")
    p.add_argument("--seq2-label", default="visdrone_val_seq2")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-json", default="reports/mot_error_slices_report.json")
    p.add_argument("--out-md", default="reports/mot_error_slices_report.md")
    args = p.parse_args()

    seq1_gt = load_mot_txt(args.seq1_gt, is_gt=True)
    seq1_pred = load_mot_txt(args.seq1_pred, is_gt=False)
    seq2_gt = load_mot_txt(args.seq2_gt, is_gt=True)
    seq2_pred = load_mot_txt(args.seq2_pred, is_gt=False)

    seq1 = _analyze_sequence(
        label=args.seq1_label,
        gt_by_frame=seq1_gt,
        pred_by_frame=seq1_pred,
        iou_threshold=float(args.iou_threshold),
        top_k=max(1, int(args.top_k)),
    )
    seq2 = _analyze_sequence(
        label=args.seq2_label,
        gt_by_frame=seq2_gt,
        pred_by_frame=seq2_pred,
        iou_threshold=float(args.iou_threshold),
        top_k=max(1, int(args.top_k)),
    )
    report = {
        "status": "SUCCESS",
        "inputs": {
            "seq1_gt": args.seq1_gt,
            "seq1_pred": args.seq1_pred,
            "seq2_gt": args.seq2_gt,
            "seq2_pred": args.seq2_pred,
            "iou_threshold": float(args.iou_threshold),
            "top_k": int(args.top_k),
        },
        "seq1": seq1,
        "seq2": seq2,
        "delta_seq2_minus_seq1": _delta(seq1, seq2),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_build_md(report), encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
