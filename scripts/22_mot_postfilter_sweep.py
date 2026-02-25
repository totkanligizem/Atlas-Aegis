#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.mot_eval import MOTDetection, evaluate_mot, load_mot_txt


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(float(p)))
    return out


def _parse_csv_floats(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


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


def _parse_class_min_conf_grid(text: str) -> list[dict[int, float]]:
    raw = str(text).strip()
    if not raw:
        return [{}]

    classes: list[int] = []
    value_lists: list[list[float]] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(
                f"Invalid class-min-conf-grid entry: {p} (expected class:v1|v2)"
            )
        k_s, vals_s = p.split(":", 1)
        cls = int(float(k_s.strip()))
        if cls < 0:
            raise ValueError(f"class id must be >= 0 in class-min-conf-grid: {p}")
        vals: list[float] = []
        for v in vals_s.split("|"):
            vv = v.strip()
            if not vv:
                continue
            vals.append(max(0.0, float(vv)))
        if not vals:
            raise ValueError(f"class-min-conf-grid requires at least one value: {p}")
        classes.append(cls)
        value_lists.append(vals)

    if not classes:
        return [{}]

    out: list[dict[int, float]] = []
    for combo in product(*value_lists):
        cur: dict[int, float] = {}
        for i, cls in enumerate(classes):
            cur[int(cls)] = float(combo[i])
        out.append(cur)
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


def _load_jsonl_tracks(path: Path, *, max_frames: int) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"track jsonl not found: {path}")
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            payload = json.loads(ln)
        except Exception:
            continue
        frame_id = _safe_int(payload.get("frame_id", -1), -1) + 1
        if frame_id <= 0:
            continue
        if max_frames > 0 and frame_id > max_frames:
            continue
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            continue
        by_frame[frame_id] = tracks
    return by_frame


def _build_pred(
    by_frame_raw: dict[int, list[dict[str, Any]]],
    *,
    min_track_age: int,
    min_conf: float,
    class_min_conf_map: dict[int, float],
    min_conf_relaxed: float,
    min_conf_relax_age_start: int,
    min_roi_dwell: int,
) -> tuple[dict[int, list[MOTDetection]], dict[str, float]]:
    out: dict[int, list[MOTDetection]] = {}
    stats = {
        "frames_total": float(len(by_frame_raw)),
        "tracks_total": 0.0,
        "tracks_kept": 0.0,
        "tracks_dropped_low_age": 0.0,
        "tracks_dropped_low_conf": 0.0,
        "tracks_dropped_low_roi_dwell": 0.0,
        "tracks_dropped_invalid": 0.0,
        "tracks_conf_relax_applied": 0.0,
    }
    for frame_id, tracks in by_frame_raw.items():
        rows: list[MOTDetection] = []
        stats["tracks_total"] += float(len(tracks))
        for t in tracks:
            age_frames = _safe_int(t.get("age_frames", 0), 0)
            conf = _safe_float(t.get("conf", 0.0), 0.0)
            roi_dwell = _safe_int(t.get("roi_dwell", 0), 0)
            class_id = _safe_int(t.get("class_id", -1), -1)
            class_thr = (
                float(class_min_conf_map.get(class_id, min_conf))
                if class_id >= 0
                else float(min_conf)
            )
            thr = class_thr
            if (
                min_conf_relaxed >= 0.0
                and min_conf_relax_age_start > 0
                and age_frames >= min_conf_relax_age_start
            ):
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

            track_id = _safe_int(t.get("track_id", -1), -1)
            bbox = t.get("bbox_xyxy", [])
            if track_id <= 0 or (not isinstance(bbox, list)) or len(bbox) != 4:
                stats["tracks_dropped_invalid"] += 1.0
                continue
            x1 = _safe_float(bbox[0], 0.0)
            y1 = _safe_float(bbox[1], 0.0)
            x2 = _safe_float(bbox[2], 0.0)
            y2 = _safe_float(bbox[3], 0.0)
            if x2 <= x1 or y2 <= y1:
                stats["tracks_dropped_invalid"] += 1.0
                continue
            rows.append(
                MOTDetection(
                    frame_id=frame_id,
                    track_id=track_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=conf,
                    class_id=class_id,
                )
            )
            stats["tracks_kept"] += 1.0
        if rows:
            out[frame_id] = rows
    return out, stats


def _row_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("seq2_mota", 0.0)),
        float(row.get("joint_mota", 0.0)),
        -float(row.get("seq2_id_switches", 0.0)),
    )


def _row_key_balanced(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("joint_mota", 0.0)),
        float(row.get("seq2_mota", 0.0)),
        -float(row.get("seq2_id_switches", 0.0)),
    )


def _to_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MOT Post-Filter Sweep")
    lines.append("")
    lines.append("## Constraints")
    lines.append(
        f"- seq2 recall >= {report['constraints']['min_seq2_recall']:.2f}, "
        f"seq1 recall >= {report['constraints']['min_seq1_recall']:.2f}"
    )
    lines.append("")
    lines.append("## Recommended")
    rec = report.get("recommended", {})
    if not rec:
        lines.append("- no candidate found")
    else:
        p = rec["params"]
        lines.append(
            f"- params: `age={p['min_track_age']}, conf={p['min_conf']:.2f}, "
            f"conf_relaxed={p['min_conf_relaxed']:.2f}, relax_age={p['min_conf_relax_age_start']}, "
            f"roi={p['min_roi_dwell']}, class_map={p['class_min_conf_map']}`; "
            f"seq1 MOTA={rec['seq1_mota']:.4f}, seq2 MOTA={rec['seq2_mota']:.4f}, "
            f"seq1 recall={rec['seq1_recall']:.4f}, seq2 recall={rec['seq2_recall']:.4f}, "
            f"seq2 IDSW={rec['seq2_id_switches']:.0f}"
        )
    lines.append("")
    lines.append("## Top 10 (Constrained)")
    lines.append(
        "| rank | age | conf | conf_relaxed | relax_age | roi | class_map | seq1_mota | seq2_mota | "
        "joint_mota | seq1_recall | seq2_recall | seq2_idsw | seq2_fp | seq2_fn |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(report.get("top10_constrained", []), 1):
        p = r["params"]
        lines.append(
            f"| {i} | {p['min_track_age']} | {p['min_conf']:.2f} | {p['min_conf_relaxed']:.2f} | "
            f"{p['min_conf_relax_age_start']} | {p['min_roi_dwell']} | {p['class_min_conf_map']} | "
            f"{r['seq1_mota']:.4f} | {r['seq2_mota']:.4f} | {r['joint_mota']:.4f} | "
            f"{r['seq1_recall']:.4f} | {r['seq2_recall']:.4f} | {r['seq2_id_switches']:.0f} | "
            f"{r['seq2_fp']:.0f} | {r['seq2_fn']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq1-gt", default="data/mot/gt.txt")
    p.add_argument("--seq1-track-jsonl", default="logs/track_risk_visdrone_mot_val.jsonl")
    p.add_argument("--seq2-gt", default="data/mot/gt_seq2.txt")
    p.add_argument("--seq2-track-jsonl", default="logs/track_risk_visdrone_mot_val_seq2.jsonl")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--age-values", default="3,4,5,6,8,10")
    p.add_argument("--conf-values", default="0.25,0.27,0.30")
    p.add_argument("--class-min-conf-map", default="")
    p.add_argument(
        "--class-min-conf-grid",
        default="",
        help="Optional class threshold grid, e.g. '1:0.30|0.35|0.40,0:0.30|0.34'",
    )
    p.add_argument("--conf-relaxed-values", default="-1")
    p.add_argument("--conf-relax-age-values", default="0")
    p.add_argument("--roi-values", default="0,1,2")
    p.add_argument("--min-seq2-recall", type=float, default=0.30)
    p.add_argument("--min-seq1-recall", type=float, default=0.42)
    p.add_argument("--out-json", default="reports/mot_postfilter_sweep_report.json")
    p.add_argument("--out-md", default="reports/mot_postfilter_sweep_report.md")
    args = p.parse_args()

    age_values = _parse_csv_ints(args.age_values)
    conf_values = _parse_csv_floats(args.conf_values)
    class_min_conf_map = _parse_class_min_conf_map(args.class_min_conf_map)
    class_min_conf_grid = _parse_class_min_conf_grid(args.class_min_conf_grid)
    conf_relaxed_values = _parse_csv_floats(args.conf_relaxed_values)
    conf_relax_age_values = _parse_csv_ints(args.conf_relax_age_values)
    roi_values = _parse_csv_ints(args.roi_values)
    if (
        not age_values
        or not conf_values
        or not conf_relaxed_values
        or not conf_relax_age_values
        or not roi_values
    ):
        raise ValueError("age/conf/conf-relaxed/relax-age/roi lists must be non-empty.")

    seq1_gt = load_mot_txt(args.seq1_gt, is_gt=True)
    seq2_gt = load_mot_txt(args.seq2_gt, is_gt=True)
    seq1_raw = _load_jsonl_tracks(Path(args.seq1_track_jsonl), max_frames=int(args.max_frames))
    seq2_raw = _load_jsonl_tracks(Path(args.seq2_track_jsonl), max_frames=int(args.max_frames))

    rows: list[dict[str, Any]] = []
    total = (
        len(age_values)
        * len(conf_values)
        * len(class_min_conf_grid)
        * len(conf_relaxed_values)
        * len(conf_relax_age_values)
        * len(roi_values)
    )
    idx = 0
    for age in age_values:
        for conf in conf_values:
            for class_grid_map in class_min_conf_grid:
                eff_class_map = dict(class_min_conf_map)
                for k, v in class_grid_map.items():
                    eff_class_map[int(k)] = float(v)
                eff_map_print = {str(k): eff_class_map[k] for k in sorted(eff_class_map)}
                for conf_relaxed in conf_relaxed_values:
                    for conf_relax_age_start in conf_relax_age_values:
                        for roi in roi_values:
                            idx += 1
                            print(
                                f"[postfilter_sweep] {idx}/{total} age={age} conf={conf:.2f} "
                                f"class_map={eff_map_print} conf_relaxed={conf_relaxed:.2f} "
                                f"relax_age={conf_relax_age_start} roi={roi}",
                                flush=True,
                            )
                            seq1_pred, seq1_stats = _build_pred(
                                seq1_raw,
                                min_track_age=max(1, int(age)),
                                min_conf=max(0.0, float(conf)),
                                class_min_conf_map=eff_class_map,
                                min_conf_relaxed=float(conf_relaxed),
                                min_conf_relax_age_start=max(0, int(conf_relax_age_start)),
                                min_roi_dwell=max(0, int(roi)),
                            )
                            seq2_pred, seq2_stats = _build_pred(
                                seq2_raw,
                                min_track_age=max(1, int(age)),
                                min_conf=max(0.0, float(conf)),
                                class_min_conf_map=eff_class_map,
                                min_conf_relaxed=float(conf_relaxed),
                                min_conf_relax_age_start=max(0, int(conf_relax_age_start)),
                                min_roi_dwell=max(0, int(roi)),
                            )
                            m1 = evaluate_mot(
                                seq1_gt, seq1_pred, iou_threshold=float(args.iou_threshold)
                            )
                            m2 = evaluate_mot(
                                seq2_gt, seq2_pred, iou_threshold=float(args.iou_threshold)
                            )
                            row = {
                                "params": {
                                    "min_track_age": int(age),
                                    "min_conf": float(conf),
                                    "class_min_conf_map": {
                                        str(k): eff_class_map[k] for k in sorted(eff_class_map)
                                    },
                                    "min_conf_relaxed": float(conf_relaxed),
                                    "min_conf_relax_age_start": int(conf_relax_age_start),
                                    "min_roi_dwell": int(roi),
                                },
                                "seq1_mota": float(m1.get("mota", 0.0)),
                                "seq2_mota": float(m2.get("mota", 0.0)),
                                "seq1_recall": float(m1.get("recall", 0.0)),
                                "seq2_recall": float(m2.get("recall", 0.0)),
                                "seq1_precision": float(m1.get("precision", 0.0)),
                                "seq2_precision": float(m2.get("precision", 0.0)),
                                "seq1_id_switches": float(m1.get("id_switches", 0.0)),
                                "seq2_id_switches": float(m2.get("id_switches", 0.0)),
                                "seq1_fp": float(m1.get("fp", 0.0)),
                                "seq2_fp": float(m2.get("fp", 0.0)),
                                "seq1_fn": float(m1.get("fn", 0.0)),
                                "seq2_fn": float(m2.get("fn", 0.0)),
                                "joint_mota": (
                                    float(m1.get("mota", 0.0)) + float(m2.get("mota", 0.0))
                                )
                                / 2.0,
                                "seq1_filter_stats": seq1_stats,
                                "seq2_filter_stats": seq2_stats,
                            }
                            rows.append(row)

    constrained = [
        r
        for r in rows
        if r["seq2_recall"] >= float(args.min_seq2_recall)
        and r["seq1_recall"] >= float(args.min_seq1_recall)
    ]

    best_by_seq2 = sorted(rows, key=_row_key, reverse=True)[0] if rows else {}
    best_by_joint = sorted(rows, key=_row_key_balanced, reverse=True)[0] if rows else {}
    if constrained:
        recommended = sorted(constrained, key=_row_key, reverse=True)[0]
    else:
        recommended = best_by_seq2

    report = {
        "status": "SUCCESS",
        "inputs": {
            "seq1_gt": args.seq1_gt,
            "seq1_track_jsonl": args.seq1_track_jsonl,
            "seq2_gt": args.seq2_gt,
            "seq2_track_jsonl": args.seq2_track_jsonl,
            "max_frames": int(args.max_frames),
            "iou_threshold": float(args.iou_threshold),
            "age_values": age_values,
            "conf_values": conf_values,
            "class_min_conf_map": {str(k): v for k, v in sorted(class_min_conf_map.items())},
            "class_min_conf_grid": args.class_min_conf_grid,
            "class_min_conf_grid_size": len(class_min_conf_grid),
            "conf_relaxed_values": conf_relaxed_values,
            "conf_relax_age_values": conf_relax_age_values,
            "roi_values": roi_values,
        },
        "constraints": {
            "min_seq2_recall": float(args.min_seq2_recall),
            "min_seq1_recall": float(args.min_seq1_recall),
        },
        "num_runs": len(rows),
        "num_constrained": len(constrained),
        "recommended": recommended,
        "best_by_seq2_mota": best_by_seq2,
        "best_by_joint_mota": best_by_joint,
        "top10_constrained": sorted(constrained, key=_row_key, reverse=True)[:10],
        "all_runs": rows,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "recommended": recommended.get("params", {}),
                "num_runs": len(rows),
                "num_constrained": len(constrained),
                "out_json": str(out_json),
                "out_md": str(out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
