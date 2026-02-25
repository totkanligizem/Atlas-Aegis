#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


def _pick_existing_report(candidates: list[Path], fallback: Path) -> Path:
    existing: list[Path] = [p for p in candidates if p.exists()]
    if not existing:
        return fallback
    # Prefer the most recently updated report to avoid stale defaults.
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


DEFAULT_TRACK_REPORT = _pick_existing_report(
    [
        ROOT / "reports/track_risk_visdrone_mot_val_report.json",
        ROOT / "reports/track_risk_report.json",
    ],
    ROOT / "reports/track_risk_report.json",
)
DEFAULT_BENCH_REPORT = ROOT / "reports/benchmark_report.json"
DEFAULT_LATENCY_REPORT = ROOT / "reports/latency_benchmark_report.json"
DEFAULT_EXPORT_BENCH_REPORT = ROOT / "reports/export_benchmark_report.json"
DEFAULT_MOT_REPORT = _pick_existing_report(
    [
        ROOT / "reports/mot_eval_visdrone_val_with_events_report.json",
        ROOT / "reports/mot_eval_visdrone_val_report.json",
        ROOT / "reports/mot_eval_report.json",
    ],
    ROOT / "reports/mot_eval_report.json",
)
DEFAULT_MOT_SWEEP_REPORT = ROOT / "reports/mot_tuning_sweep_visdrone_val_report.json"
DEFAULT_MOT_POSTFILTER_REPORT = _pick_existing_report(
    [
        ROOT / "reports/mot_postfilter_sweep_classmap_report.json",
        ROOT / "reports/mot_postfilter_sweep_recall_report.json",
        ROOT / "reports/mot_postfilter_sweep_report.json",
    ],
    ROOT / "reports/mot_postfilter_sweep_report.json",
)
DEFAULT_MOT_PROFILE_COMPARE_REPORT = ROOT / "reports/mot_profile_comparison_report.json"
DEFAULT_MOT_RELEASE_GATE_REPORT = ROOT / "reports/mot_profile_release_gate_report.json"
DEFAULT_MOT_ERROR_REPORT = ROOT / "reports/mot_error_slices_report.json"
DEFAULT_DETECTOR_EVAL_REPORT = _pick_existing_report(
    [
        ROOT / "reports/detector_eval_full41_highrecall_blur_rescue_report.json",
        ROOT / "reports/detector_eval_candidate_highrecall_blur_rescue_report.json",
        ROOT / "reports/detector_eval_full41_highrecall_poisson_focus_1600_report.json",
        ROOT / "reports/detector_eval_candidate_highrecall_poisson_focus_1600_report.json",
        ROOT / "reports/detector_eval_full41_highrecall_poisson_focus_mid_report.json",
        ROOT / "reports/detector_eval_candidate_highrecall_poisson_focus_mid_report.json",
        ROOT / "reports/detector_eval_candidate_highrecall_corruptaware_quick_report.json",
        ROOT / "reports/detector_eval_highrecall_corruptaware_quick_fast_report.json",
        ROOT / "reports/detector_eval_full41_highrecall_ep1_report.json",
        ROOT / "reports/detector_eval_full41_fast_ep3_report.json",
        ROOT / "reports/detector_eval_candidate_highrecall_full_ep1_report.json",
        ROOT / "reports/detector_eval_highrecall_full_ep1_report.json",
        ROOT / "reports/detector_eval_candidate_ep3_report.json",
        ROOT / "reports/detector_eval_full_ep3_report.json",
        ROOT / "reports/detector_eval_candidate_ep2_report.json",
        ROOT / "reports/detector_eval_full_ep2_report.json",
        ROOT / "reports/detector_eval_candidate_ep1_report.json",
        ROOT / "reports/detector_eval_full_ep1_report.json",
        ROOT / "reports/detector_eval_report.json",
        ROOT / "reports/detector_eval_smoke_trained_report.json",
    ],
    ROOT / "reports/detector_eval_report.json",
)
DEFAULT_TRACK_JSONL = _pick_existing_report(
    [
        ROOT / "logs/track_risk_visdrone_mot_val.jsonl",
        ROOT / "logs/track_risk.jsonl",
    ],
    ROOT / "logs/track_risk.jsonl",
)
DEFAULT_METRICS_DB = ROOT / "logs/metrics.duckdb"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_track_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    frame_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame_id = int(payload.get("frame_id", 0))
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            continue

        bands = Counter()
        risks: list[float] = []
        for t in tracks:
            band = str(t.get("band", "UNKNOWN"))
            bands[band] += 1
            risk = float(t.get("risk_score", 0.0))
            risks.append(risk)
            track_rows.append(
                {
                    "frame_id": frame_id,
                    "track_id": int(t.get("track_id", -1)),
                    "class_id": int(t.get("class_id", -1)),
                    "conf": float(t.get("conf", 0.0)),
                    "risk_score": risk,
                    "band": band,
                    "roi_dwell": int(t.get("roi_dwell", 0)),
                    "age_frames": int(t.get("age_frames", 0)),
                    "occlusion_count": int(t.get("occlusion_count", 0)),
                }
            )

        frame_rows.append(
            {
                "frame_id": frame_id,
                "track_count": len(tracks),
                "risk_max": max(risks) if risks else 0.0,
                "risk_mean": (sum(risks) / len(risks)) if risks else 0.0,
                "green_count": bands.get("GREEN", 0),
                "yellow_count": bands.get("YELLOW", 0),
                "red_count": bands.get("RED", 0),
            }
        )

    return pd.DataFrame(frame_rows), pd.DataFrame(track_rows)


def _bench_results_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    results = report.get("results", {})
    if not isinstance(results, dict):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    clean_det = float(results.get("clean", {}).get("mean_detections", 0.0) or 0.0)
    clean_conf = float(results.get("clean", {}).get("mean_conf", 0.0) or 0.0)
    clean_det = max(clean_det, 1e-6)
    clean_conf = max(clean_conf, 1e-6)

    for condition, stat in results.items():
        mean_det = float(stat.get("mean_detections", 0.0))
        mean_conf = float(stat.get("mean_conf", 0.0))
        rows.append(
            {
                "condition": condition,
                "frames": float(stat.get("frames", 0.0)),
                "mean_detections": mean_det,
                "mean_conf": mean_conf,
                "mean_infer_ms": float(stat.get("mean_infer_ms", 0.0)),
                "rel_drop_det_vs_clean": (clean_det - mean_det) / clean_det,
                "rel_drop_conf_vs_clean": (clean_conf - mean_conf) / clean_conf,
            }
        )
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def _latency_profiles_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    profiles = report.get("profiles", [])
    if not isinstance(profiles, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        metrics = p.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        thresholds = p.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        gate = p.get("gate", {})
        if not isinstance(gate, dict):
            gate = {}

        rows.append(
            {
                "profile": str(p.get("name", "")),
                "status": str(p.get("status", "UNKNOWN")),
                "required": bool(p.get("required", False)),
                "device": str(p.get("device_effective", p.get("device_requested", ""))),
                "imgsz": int(float(p.get("imgsz", 0) or 0)),
                "frames_measured": int(float(p.get("num_frames_measured", 0) or 0)),
                "fps_mean": float(metrics.get("fps_mean", 0.0) or 0.0),
                "fps_p05": float(metrics.get("fps_p05", 0.0) or 0.0),
                "latency_median_ms": float(metrics.get("latency_median_ms", 0.0) or 0.0),
                "latency_p95_ms": float(metrics.get("latency_p95_ms", 0.0) or 0.0),
                "target_fps": float(thresholds.get("target_fps", 0.0) or 0.0),
                "target_latency_ms": float(thresholds.get("target_latency_ms", 0.0) or 0.0),
                "target_p95_latency_ms": float(
                    thresholds.get("target_p95_latency_ms", 0.0) or 0.0
                ),
                "gate_pass": bool(gate.get("pass", False)),
                "reason": str(p.get("reason", "")),
                "error": str(p.get("error", "")),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["required", "fps_mean"], ascending=[False, False]).reset_index(drop=True)


def _latency_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    summary = report.get("summary", {})
    if isinstance(summary, dict):
        return summary
    return {}


def _export_backend_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    profiles = report.get("profiles", [])
    if not isinstance(profiles, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        pname = str(p.get("name", ""))
        pstatus = str(p.get("status", "UNKNOWN"))
        backends = p.get("backends", [])
        if not isinstance(backends, list):
            continue
        for b in backends:
            if not isinstance(b, dict):
                continue
            metrics = b.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "profile": pname,
                    "profile_status": pstatus,
                    "backend": str(b.get("backend", "")),
                    "requirement": str(b.get("requirement", "")),
                    "status": str(b.get("status", "UNKNOWN")),
                    "fps_mean": float(metrics.get("fps_mean", 0.0) or 0.0),
                    "latency_median_ms": float(metrics.get("latency_median_ms", 0.0) or 0.0),
                    "latency_p95_ms": float(metrics.get("latency_p95_ms", 0.0) or 0.0),
                    "speedup_vs_pytorch_fps": float(
                        b.get("speedup_vs_pytorch_fps", 0.0) or 0.0
                    ),
                    "num_frames_measured": int(float(b.get("num_frames_measured", 0) or 0)),
                    "export_path": str(b.get("export_path", "")),
                    "reason": str(b.get("reason", "")),
                    "error": str(b.get("error", "")),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["profile", "backend"])
        .reset_index(drop=True)
    )


def _export_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    out = report.get("summary", {})
    if isinstance(out, dict):
        return out
    return {}


def _detector_eval_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    ev = report.get("evaluation", {})
    if not isinstance(ev, dict):
        return pd.DataFrame()
    mbc = ev.get("metrics_by_condition", {})
    if not isinstance(mbc, dict):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    clean_small = float(mbc.get("clean", {}).get("recall_small", 0.0) or 0.0)
    clean_small = max(clean_small, 1e-6)
    for condition, metrics in mbc.items():
        if not isinstance(metrics, dict):
            continue
        cur_small = float(metrics.get("recall_small", 0.0) or 0.0)
        rows.append(
            {
                "condition": condition,
                "precision": float(metrics.get("precision", 0.0) or 0.0),
                "recall": float(metrics.get("recall", 0.0) or 0.0),
                "f1": float(metrics.get("f1", 0.0) or 0.0),
                "recall_small": cur_small,
                "small_rel_drop_vs_clean": (clean_small - cur_small) / clean_small,
                "tp": float(metrics.get("tp", 0.0) or 0.0),
                "fp": float(metrics.get("fp", 0.0) or 0.0),
                "fn": float(metrics.get("fn", 0.0) or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def _detector_gate_df(report: dict[str, Any] | None) -> tuple[pd.DataFrame, bool]:
    if not report:
        return pd.DataFrame(), False
    gates = report.get("gates", {})
    if not isinstance(gates, dict):
        return pd.DataFrame(), False
    checks = gates.get("checks", [])
    if not isinstance(checks, list):
        return pd.DataFrame(), bool(gates.get("pass", False))
    rows: list[dict[str, Any]] = []
    for c in checks:
        rows.append(
            {
                "name": str(c.get("name", "")),
                "value": float(c.get("value", 0.0) or 0.0),
                "threshold": float(c.get("threshold", 0.0) or 0.0),
                "passed": bool(c.get("passed", False)),
            }
        )
    return pd.DataFrame(rows), bool(gates.get("pass", False))


def _mot_metrics(report: dict[str, Any] | None) -> dict[str, float]:
    if not report:
        return {}
    mm = report.get("mot_metrics", {})
    if not isinstance(mm, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in mm.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _mot_hota_curve(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    mm = report.get("mot_metrics", {})
    if not isinstance(mm, dict):
        return pd.DataFrame()
    curve = mm.get("hota_curve", [])
    if not isinstance(curve, list):
        return pd.DataFrame()
    rows: list[dict[str, float]] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "alpha_iou": float(row.get("alpha_iou", 0.0) or 0.0),
                "hota": float(row.get("hota", 0.0) or 0.0),
                "assa": float(row.get("assa", 0.0) or 0.0),
                "deta": float(row.get("deta", 0.0) or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("alpha_iou").reset_index(drop=True)


def _lead_metrics(report: dict[str, Any] | None) -> dict[str, float]:
    if not report:
        return {}
    lm = report.get("event_lead_time", {})
    if not isinstance(lm, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in lm.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _mot_sweep_runs_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        params = r.get("params", {})
        mm = r.get("mot_metrics", {})
        rows.append(
            {
                "tag": str(r.get("tag", "")),
                "conf": float((params or {}).get("conf", 0.0) or 0.0),
                "iou": float((params or {}).get("iou", 0.0) or 0.0),
                "ttl": int(float((params or {}).get("track_ttl_frames", 0) or 0)),
                "mota": float((mm or {}).get("mota", 0.0) or 0.0),
                "precision": float((mm or {}).get("precision", 0.0) or 0.0),
                "recall": float((mm or {}).get("recall", 0.0) or 0.0),
                "id_switches": float((mm or {}).get("id_switches", 0.0) or 0.0),
                "tp": float((mm or {}).get("tp", 0.0) or 0.0),
                "fp": float((mm or {}).get("fp", 0.0) or 0.0),
                "fn": float((mm or {}).get("fn", 0.0) or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["mota", "precision", "id_switches"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _mot_sweep_best(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    best = report.get("best_run", {})
    if isinstance(best, dict):
        return best
    return {}


def _mot_error_delta(report: dict[str, Any] | None) -> dict[str, float]:
    if not report:
        return {}
    delta = report.get("delta_seq2_minus_seq1", {})
    if not isinstance(delta, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in delta.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def _mot_error_worst_frames(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    seq2 = report.get("seq2", {})
    if not isinstance(seq2, dict):
        return pd.DataFrame()
    rows = seq2.get("worst_frames", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keep = [c for c in ["frame_id", "score", "fp", "fn", "id_switches", "tp", "gt_count", "pred_count"] if c in out.columns]
    if keep:
        out = out[keep]
    return out


def _mot_error_top_tracks(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    seq2 = report.get("seq2", {})
    if not isinstance(seq2, dict):
        return pd.DataFrame()
    rows = seq2.get("top_idsw_tracks", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _mot_postfilter_recommended(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    rec = report.get("recommended", {})
    if isinstance(rec, dict):
        return rec
    return {}


def _mot_postfilter_top10(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    rows = report.get("top10_constrained", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = r.get("params", {})
        out.append(
            {
                "min_track_age": int(float((p or {}).get("min_track_age", 0) or 0)),
                "min_conf": float((p or {}).get("min_conf", 0.0) or 0.0),
                "class_min_conf_map": json.dumps((p or {}).get("class_min_conf_map", {}), ensure_ascii=True),
                "min_conf_relaxed": float((p or {}).get("min_conf_relaxed", -1.0) or -1.0),
                "min_conf_relax_age_start": int(
                    float((p or {}).get("min_conf_relax_age_start", 0) or 0)
                ),
                "min_roi_dwell": int(float((p or {}).get("min_roi_dwell", 0) or 0)),
                "seq1_mota": float(r.get("seq1_mota", 0.0) or 0.0),
                "seq2_mota": float(r.get("seq2_mota", 0.0) or 0.0),
                "joint_mota": float(r.get("joint_mota", 0.0) or 0.0),
                "seq1_recall": float(r.get("seq1_recall", 0.0) or 0.0),
                "seq2_recall": float(r.get("seq2_recall", 0.0) or 0.0),
                "seq2_idsw": float(r.get("seq2_id_switches", 0.0) or 0.0),
                "seq2_fp": float(r.get("seq2_fp", 0.0) or 0.0),
                "seq2_fn": float(r.get("seq2_fn", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(out)


def _mot_profile_winners(report: dict[str, Any] | None) -> dict[str, str]:
    if not report:
        return {}
    winners = report.get("winners", {})
    if not isinstance(winners, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in winners.items():
        out[str(k)] = str(v)
    return out


def _mot_profile_delta_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    profiles = report.get("profiles", {})
    deltas = report.get("delta_recall_minus_stability", {})
    if not isinstance(profiles, dict) or not isinstance(deltas, dict):
        return pd.DataFrame()

    metric_order = [
        "mota",
        "hota",
        "assa",
        "deta",
        "precision",
        "recall",
        "motp_iou",
        "id_switches",
        "fp",
        "fn",
    ]
    rows: list[dict[str, Any]] = []
    for seq in ("seq1", "seq2"):
        seq_profiles = profiles.get(seq, {})
        seq_deltas = deltas.get(seq, {})
        if not isinstance(seq_profiles, dict):
            continue
        recall_m = seq_profiles.get("recall", {})
        stability_m = seq_profiles.get("stability", {})
        if not isinstance(recall_m, dict) or not isinstance(stability_m, dict):
            continue
        for metric in metric_order:
            rows.append(
                {
                    "sequence": seq,
                    "metric": metric,
                    "recall": float(recall_m.get(metric, 0.0) or 0.0),
                    "stability": float(stability_m.get(metric, 0.0) or 0.0),
                    "delta_recall_minus_stability": float(seq_deltas.get(metric, 0.0) or 0.0),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["sequence"] = pd.Categorical(out["sequence"], categories=["seq1", "seq2"], ordered=True)
    out["metric"] = pd.Categorical(out["metric"], categories=metric_order, ordered=True)
    return out.sort_values(["sequence", "metric"]).reset_index(drop=True)


def _mot_release_gate_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    sel = report.get("selection", {})
    if not isinstance(sel, dict):
        return {}
    out: dict[str, Any] = {}
    out["gate_status"] = str(sel.get("gate_status", "UNKNOWN"))
    out["selected_profile"] = str(sel.get("selected_profile", "n/a"))
    out["objective"] = str(sel.get("objective", "n/a"))
    thresholds = sel.get("thresholds", {})
    if isinstance(thresholds, dict):
        out["thresholds"] = thresholds
    else:
        out["thresholds"] = {}
    rationale = sel.get("rationale", [])
    if isinstance(rationale, list):
        out["rationale"] = [str(x) for x in rationale]
    else:
        out["rationale"] = []
    return out


def _mot_release_gate_checks_df(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    sel = report.get("selection", {})
    if not isinstance(sel, dict):
        return pd.DataFrame()
    profiles = sel.get("profiles", {})
    if not isinstance(profiles, dict):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for profile_name in ("recall", "stability"):
        p = profiles.get(profile_name, {})
        if not isinstance(p, dict):
            continue
        gate = p.get("gate", {})
        if not isinstance(gate, dict):
            continue
        checks = gate.get("checks", {})
        if not isinstance(checks, dict):
            continue
        for metric, chk in checks.items():
            if not isinstance(chk, dict):
                continue
            rows.append(
                {
                    "profile": profile_name,
                    "check": str(metric),
                    "value": float(chk.get("value", 0.0) or 0.0),
                    "threshold": float(chk.get("threshold", 0.0) or 0.0),
                    "passed": bool(chk.get("passed", False)),
                }
            )
    return pd.DataFrame(rows)


def _fetch_query(db_path: Path, query: str) -> tuple[pd.DataFrame, str]:
    if not db_path.exists():
        return pd.DataFrame(), "missing"

    try:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path), read_only=True)
        cur = con.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        con.close()
        return pd.DataFrame(rows, columns=cols), "duckdb"
    except Exception:
        pass

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        con.close()
        return pd.DataFrame(rows, columns=cols), "sqlite"
    except Exception:
        return pd.DataFrame(), "unreadable"


def _load_runs_metrics(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    runs_df, backend = _fetch_query(
        db_path,
        """
        SELECT run_id, pipeline, tier, mode, status, started_at, ended_at
        FROM runs
        ORDER BY started_at DESC
        """,
    )
    metrics_df, backend_metrics = _fetch_query(
        db_path,
        """
        SELECT run_id, condition, metric_name, metric_value, recorded_at
        FROM metrics
        ORDER BY recorded_at DESC
        """,
    )

    if backend == "missing":
        return runs_df, metrics_df, backend
    if backend in ("duckdb", "sqlite"):
        return runs_df, metrics_df, backend
    return runs_df, metrics_df, backend_metrics


def _kpi_value(report: dict[str, Any] | None, key: str, default: int | float = 0) -> int | float:
    if not report:
        return default
    v = report.get(key, default)
    if isinstance(v, (int, float)):
        return v
    return default


def _red_alarm_count(report: dict[str, Any] | None) -> int:
    if not report:
        return 0
    events = report.get("events", [])
    if not isinstance(events, list):
        return 0
    return sum(1 for e in events if str(e.get("type", "")) == "alarm_red")


def _event_table(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    events = report.get("events", [])
    if not isinstance(events, list):
        return pd.DataFrame()
    rows = []
    for e in events:
        rows.append(
            {
                "type": str(e.get("type", "")),
                "track_id": int(e.get("track_id", -1)),
                "frame_id": int(e.get("frame_id", -1)),
            }
        )
    return pd.DataFrame(rows)


def _track_leaderboard(track_df: pd.DataFrame) -> pd.DataFrame:
    if track_df.empty:
        return pd.DataFrame()
    grouped = (
        track_df.groupby("track_id", as_index=False)
        .agg(
            max_risk=("risk_score", "max"),
            mean_risk=("risk_score", "mean"),
            max_roi_dwell=("roi_dwell", "max"),
            max_occlusion=("occlusion_count", "max"),
            frames_seen=("frame_id", "count"),
            last_frame=("frame_id", "max"),
        )
        .sort_values(["max_risk", "frames_seen"], ascending=[False, False])
    )
    return grouped.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Aerial MOT + Risk Dashboard", layout="wide")
    st.title("Robust Aerial Detection + MOT + Risk Banding Dashboard")
    st.caption("Local-first run observability for GitHub/LinkedIn-ready project demos.")

    st.sidebar.header("Input Paths")
    track_report_path = Path(
        st.sidebar.text_input("Track report JSON", str(DEFAULT_TRACK_REPORT))
    )
    bench_report_path = Path(
        st.sidebar.text_input("Benchmark report JSON", str(DEFAULT_BENCH_REPORT))
    )
    latency_report_path = Path(
        st.sidebar.text_input("Latency/FPS report JSON", str(DEFAULT_LATENCY_REPORT))
    )
    export_bench_report_path = Path(
        st.sidebar.text_input("Export benchmark report JSON", str(DEFAULT_EXPORT_BENCH_REPORT))
    )
    mot_report_path = Path(
        st.sidebar.text_input("MOT eval report JSON", str(DEFAULT_MOT_REPORT))
    )
    mot_sweep_report_path = Path(
        st.sidebar.text_input("MOT sweep report JSON", str(DEFAULT_MOT_SWEEP_REPORT))
    )
    mot_postfilter_report_path = Path(
        st.sidebar.text_input("MOT post-filter sweep report JSON", str(DEFAULT_MOT_POSTFILTER_REPORT))
    )
    mot_profile_compare_report_path = Path(
        st.sidebar.text_input(
            "MOT profile comparison report JSON", str(DEFAULT_MOT_PROFILE_COMPARE_REPORT)
        )
    )
    mot_release_gate_report_path = Path(
        st.sidebar.text_input(
            "MOT release gate report JSON", str(DEFAULT_MOT_RELEASE_GATE_REPORT)
        )
    )
    mot_error_report_path = Path(
        st.sidebar.text_input("MOT error slices report JSON", str(DEFAULT_MOT_ERROR_REPORT))
    )
    detector_eval_report_path = Path(
        st.sidebar.text_input("Detector eval report JSON", str(DEFAULT_DETECTOR_EVAL_REPORT))
    )
    track_jsonl_path = Path(
        st.sidebar.text_input("Track frame JSONL", str(DEFAULT_TRACK_JSONL))
    )
    metrics_db_path = Path(
        st.sidebar.text_input("Metrics DB", str(DEFAULT_METRICS_DB))
    )
    st.sidebar.caption("Change paths for different experiment folders.")

    track_report = _read_json(track_report_path)
    bench_report = _read_json(bench_report_path)
    latency_report = _read_json(latency_report_path)
    export_bench_report = _read_json(export_bench_report_path)
    mot_report = _read_json(mot_report_path)
    mot_sweep_report = _read_json(mot_sweep_report_path)
    mot_postfilter_report = _read_json(mot_postfilter_report_path)
    mot_profile_compare_report = _read_json(mot_profile_compare_report_path)
    mot_release_gate_report = _read_json(mot_release_gate_report_path)
    mot_error_report = _read_json(mot_error_report_path)
    detector_eval_report = _read_json(detector_eval_report_path)
    frames_df, tracks_df = _load_track_frames(track_jsonl_path)
    bench_df = _bench_results_df(bench_report)
    latency_df = _latency_profiles_df(latency_report)
    latency_summary = _latency_summary(latency_report)
    export_bench_df = _export_backend_df(export_bench_report)
    export_bench_summary = _export_summary(export_bench_report)
    detector_df = _detector_eval_df(detector_eval_report)
    gate_df, gate_pass = _detector_gate_df(detector_eval_report)
    mot_metrics = _mot_metrics(mot_report)
    mot_hota_curve_df = _mot_hota_curve(mot_report)
    lead_metrics = _lead_metrics(mot_report)
    mot_sweep_df = _mot_sweep_runs_df(mot_sweep_report)
    mot_sweep_best = _mot_sweep_best(mot_sweep_report)
    mot_postfilter_rec = _mot_postfilter_recommended(mot_postfilter_report)
    mot_postfilter_df = _mot_postfilter_top10(mot_postfilter_report)
    mot_profile_winners = _mot_profile_winners(mot_profile_compare_report)
    mot_profile_delta_df = _mot_profile_delta_df(mot_profile_compare_report)
    mot_release_gate_summary = _mot_release_gate_summary(mot_release_gate_report)
    mot_release_gate_checks_df = _mot_release_gate_checks_df(mot_release_gate_report)
    mot_error_delta = _mot_error_delta(mot_error_report)
    mot_error_worst_df = _mot_error_worst_frames(mot_error_report)
    mot_error_tracks_df = _mot_error_top_tracks(mot_error_report)
    runs_df, metrics_df, backend = _load_runs_metrics(metrics_db_path)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Frames", int(_kpi_value(track_report, "num_frames", 0)))
    col2.metric("Events (all)", int(_kpi_value(track_report, "event_count", 0)))
    col3.metric("Red Alarms", _red_alarm_count(track_report))
    col4.metric("Active Tracks", int(_kpi_value(track_report, "final_active_tracks", 0)))
    col5.metric("Benchmark Conditions", int(_kpi_value(bench_report, "num_conditions", 0)))

    st.subheader("Risk Timeline")
    if frames_df.empty:
        st.warning("No frame-level risk logs found.")
    else:
        left, right = st.columns(2)
        with left:
            st.line_chart(frames_df.set_index("frame_id")[["risk_max", "risk_mean"]], height=300)
        with right:
            st.area_chart(
                frames_df.set_index("frame_id")[["green_count", "yellow_count", "red_count"]],
                height=300,
            )
        st.dataframe(frames_df.tail(20), width="stretch", hide_index=True)

    st.subheader("Event Log")
    events_df = _event_table(track_report)
    if events_df.empty:
        st.info("No events available.")
    else:
        st.dataframe(events_df, width="stretch", hide_index=True)

    st.subheader("Track Leaderboard")
    leaderboard = _track_leaderboard(tracks_df)
    if leaderboard.empty:
        st.info("No track data available.")
    else:
        st.dataframe(leaderboard.head(25), width="stretch", hide_index=True)

    st.subheader("Corruption Benchmark")
    if bench_df.empty:
        st.warning("No benchmark report found.")
    else:
        left, right = st.columns(2)
        with left:
            st.bar_chart(
                bench_df.set_index("condition")[["mean_detections", "mean_conf"]],
                height=320,
            )
        with right:
            st.bar_chart(
                bench_df.set_index("condition")[["rel_drop_det_vs_clean", "rel_drop_conf_vs_clean"]],
                height=320,
            )
        st.dataframe(bench_df, width="stretch", hide_index=True)

    st.subheader("Latency / FPS Benchmark")
    if latency_df.empty:
        st.info("No latency benchmark report found.")
    else:
        gate_status = str(latency_report.get("gate_status", "UNKNOWN")).upper() if isinstance(latency_report, dict) else "UNKNOWN"
        required_profiles = int(float(latency_summary.get("required_profiles", 0) or 0))
        required_passed = int(float(latency_summary.get("required_passed", 0) or 0))
        best_profile = str(latency_summary.get("best_profile_by_fps_mean", "") or "")
        best_fps = float(latency_summary.get("best_fps_mean", 0.0) or 0.0)

        c1, c2, c3, c4 = st.columns(4)
        if gate_status == "PASS":
            c1.success(f"Latency Gate: {gate_status}")
        else:
            c1.error(f"Latency Gate: {gate_status}")
        c2.metric("Required Profiles", f"{required_passed}/{required_profiles}")
        c3.metric("Best FPS (mean)", f"{best_fps:.2f}")
        c4.metric("Best Profile", best_profile if best_profile else "n/a")

        left, right = st.columns(2)
        with left:
            fps_cols = [c for c in ["fps_mean", "target_fps"] if c in latency_df.columns]
            if fps_cols:
                st.bar_chart(latency_df.set_index("profile")[fps_cols], height=260)
        with right:
            lat_cols = [
                c
                for c in [
                    "latency_median_ms",
                    "latency_p95_ms",
                    "target_latency_ms",
                    "target_p95_latency_ms",
                ]
                if c in latency_df.columns
            ]
            if lat_cols:
                st.bar_chart(latency_df.set_index("profile")[lat_cols], height=260)

        st.dataframe(latency_df, width="stretch", hide_index=True)

    st.subheader("Export Backend Benchmark (PyTorch / ONNX / CoreML)")
    if export_bench_df.empty:
        st.info("No export benchmark report found.")
    else:
        export_gate_status = (
            str(export_bench_report.get("gate_status", "UNKNOWN")).upper()
            if isinstance(export_bench_report, dict)
            else "UNKNOWN"
        )
        profiles_total = int(float(export_bench_summary.get("profiles_total", 0) or 0))
        profiles_pass = int(float(export_bench_summary.get("profiles_pass", 0) or 0))
        best_backend = str(export_bench_summary.get("best_backend_by_fps_mean", "") or "")
        best_backend_fps = float(export_bench_summary.get("best_fps_mean", 0.0) or 0.0)

        c1, c2, c3, c4 = st.columns(4)
        if export_gate_status == "PASS":
            c1.success(f"Export Gate: {export_gate_status}")
        else:
            c1.error(f"Export Gate: {export_gate_status}")
        c2.metric("Profiles PASS", f"{profiles_pass}/{profiles_total}")
        c3.metric("Best Backend", best_backend if best_backend else "n/a")
        c4.metric("Best Backend FPS", f"{best_backend_fps:.2f}")

        fps_df = export_bench_df[["profile", "backend", "fps_mean"]].copy()
        if not fps_df.empty:
            pivot = fps_df.pivot_table(
                index="profile", columns="backend", values="fps_mean", aggfunc="mean"
            ).fillna(0.0)
            st.bar_chart(pivot, height=260)

        speed_df = export_bench_df[
            (export_bench_df["backend"] != "pytorch") & (export_bench_df["status"] == "PASS")
        ][["profile", "backend", "speedup_vs_pytorch_fps"]].copy()
        if not speed_df.empty:
            speed_pivot = speed_df.pivot_table(
                index="profile",
                columns="backend",
                values="speedup_vs_pytorch_fps",
                aggfunc="mean",
            ).fillna(0.0)
            st.caption("Speedup vs PyTorch (FPS ratio)")
            st.bar_chart(speed_pivot, height=220)

        st.dataframe(export_bench_df, width="stretch", hide_index=True)

    st.subheader("Detector Quality Gates")
    if detector_df.empty and gate_df.empty:
        st.info("No detector eval report found.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if gate_pass:
                st.success("Detector Gate: PASS")
            else:
                st.error("Detector Gate: FAIL")
            if not gate_df.empty:
                st.dataframe(gate_df, width="stretch", hide_index=True)
        with c2:
            if not detector_df.empty:
                st.bar_chart(
                    detector_df.set_index("condition")[["recall_small", "small_rel_drop_vs_clean"]],
                    height=300,
                )
        if not detector_df.empty:
            st.dataframe(detector_df, width="stretch", hide_index=True)

    st.subheader("MOT Ground-Truth Metrics")
    if not mot_metrics:
        st.info("No MOT evaluation report found.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MOTA", f"{mot_metrics.get('mota', 0.0):.4f}")
        m2.metric("HOTA", f"{mot_metrics.get('hota', 0.0):.4f}")
        m3.metric("AssA", f"{mot_metrics.get('assa', 0.0):.4f}")
        m4.metric("DetA", f"{mot_metrics.get('deta', 0.0):.4f}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("MOTP (IoU)", f"{mot_metrics.get('motp_iou', 0.0):.4f}")
        m6.metric("ID Switches", int(mot_metrics.get("id_switches", 0.0)))
        m7.metric("Recall", f"{mot_metrics.get('recall', 0.0):.4f}")
        m8.metric("Precision", f"{mot_metrics.get('precision', 0.0):.4f}")
        mm_df = pd.DataFrame(
            [
                {"metric": k, "value": v}
                for k, v in sorted(mot_metrics.items())
            ]
        )
        st.dataframe(mm_df, width="stretch", hide_index=True)
        if not mot_hota_curve_df.empty:
            st.caption("HOTA / AssA / DetA curve across IoU thresholds.")
            st.line_chart(
                mot_hota_curve_df.set_index("alpha_iou")[["hota", "assa", "deta"]],
                height=220,
            )
        if lead_metrics:
            st.caption("Event lead-time metrics (frames, positive means earlier alarm).")
            lm_df = pd.DataFrame(
                [
                    {"metric": k, "value": v}
                    for k, v in sorted(lead_metrics.items())
                ]
            )
            st.dataframe(lm_df, width="stretch", hide_index=True)

    st.subheader("MOT Tuning Sweep")
    if mot_sweep_df.empty:
        st.info("No MOT tuning sweep report found.")
    else:
        best_params = mot_sweep_best.get("params", {}) if isinstance(mot_sweep_best, dict) else {}
        best_mm = mot_sweep_best.get("mot_metrics", {}) if isinstance(mot_sweep_best, dict) else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best MOTA", f"{float((best_mm or {}).get('mota', 0.0) or 0.0):.4f}")
        c2.metric("Best Precision", f"{float((best_mm or {}).get('precision', 0.0) or 0.0):.4f}")
        c3.metric("Best Recall", f"{float((best_mm or {}).get('recall', 0.0) or 0.0):.4f}")
        c4.metric("Best IDSW", int(float((best_mm or {}).get("id_switches", 0.0) or 0.0)))
        st.caption(
            "Best params: "
            f"conf={float((best_params or {}).get('conf', 0.0) or 0.0):.3f}, "
            f"iou={float((best_params or {}).get('iou', 0.0) or 0.0):.2f}, "
            f"ttl={int(float((best_params or {}).get('track_ttl_frames', 0) or 0))}"
        )
        st.dataframe(mot_sweep_df, width="stretch", hide_index=True)

    st.subheader("MOT Post-Filter Sweep")
    if mot_postfilter_df.empty:
        st.info("No MOT post-filter sweep report found.")
    else:
        params = mot_postfilter_rec.get("params", {}) if isinstance(mot_postfilter_rec, dict) else {}
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Recommended Age", int(float((params or {}).get("min_track_age", 0) or 0)))
        c2.metric("Recommended Min Conf", f"{float((params or {}).get('min_conf', 0.0) or 0.0):.2f}")
        c3.metric(
            "Recommended Relaxed Conf",
            f"{float((params or {}).get('min_conf_relaxed', -1.0) or -1.0):.2f}",
        )
        c4.metric(
            "Recommended Relax Age",
            int(float((params or {}).get("min_conf_relax_age_start", 0) or 0)),
        )
        c5.metric("Recommended ROI Dwell", int(float((params or {}).get("min_roi_dwell", 0) or 0)))
        c6.metric("Recommended Seq2 MOTA", f"{float((mot_postfilter_rec or {}).get('seq2_mota', 0.0) or 0.0):.4f}")
        st.caption(
            "Recommended class min-conf map: "
            f"`{json.dumps((params or {}).get('class_min_conf_map', {}), ensure_ascii=True)}`"
        )
        st.dataframe(mot_postfilter_df, width="stretch", hide_index=True)

    st.subheader("MOT Profile Comparison (Recall vs Stability)")
    if mot_profile_delta_df.empty and not mot_profile_winners:
        st.info("No MOT profile comparison report found.")
    else:
        w1, w2, w3 = st.columns(3)
        w1.metric(
            "Seq2 Recall Priority",
            str(mot_profile_winners.get("seq2_recall_priority", "n/a")).upper(),
        )
        w2.metric(
            "Seq2 Stability Priority",
            str(mot_profile_winners.get("seq2_stability_priority", "n/a")).upper(),
        )
        w3.metric(
            "Seq2 HOTA Priority",
            str(mot_profile_winners.get("seq2_hota_priority", "n/a")).upper(),
        )
        if not mot_profile_delta_df.empty:
            seq2_df = mot_profile_delta_df[mot_profile_delta_df["sequence"] == "seq2"].copy()
            if not seq2_df.empty:
                seq2_map = {
                    str(r["metric"]): float(r["delta_recall_minus_stability"])
                    for _, r in seq2_df.iterrows()
                }
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Seq2 Delta Recall", f"{seq2_map.get('recall', 0.0):+.4f}")
                d2.metric("Seq2 Delta HOTA", f"{seq2_map.get('hota', 0.0):+.4f}")
                d3.metric("Seq2 Delta MOTA", f"{seq2_map.get('mota', 0.0):+.4f}")
                d4.metric("Seq2 Delta FP", f"{seq2_map.get('fp', 0.0):+.0f}")
                seq2_plot = seq2_df.set_index("metric")[["delta_recall_minus_stability"]]
                st.bar_chart(seq2_plot, height=220)
            st.dataframe(mot_profile_delta_df, width="stretch", hide_index=True)

    st.subheader("MOT Release Profile Gate")
    if not mot_release_gate_summary:
        st.info("No MOT release gate report found.")
    else:
        gate_status = str(mot_release_gate_summary.get("gate_status", "UNKNOWN")).upper()
        selected = str(mot_release_gate_summary.get("selected_profile", "n/a")).upper()
        objective = str(mot_release_gate_summary.get("objective", "n/a")).upper()
        g1, g2, g3 = st.columns(3)
        if gate_status == "PASS":
            g1.success(f"Gate {gate_status}")
        else:
            g1.error(f"Gate {gate_status}")
        g2.metric("Selected MOT_PROFILE", selected)
        g3.metric("Objective", objective)
        rationale = mot_release_gate_summary.get("rationale", [])
        if isinstance(rationale, list) and rationale:
            st.caption("Decision rationale")
            for reason in rationale:
                st.write(f"- {reason}")
        thresholds = mot_release_gate_summary.get("thresholds", {})
        if isinstance(thresholds, dict) and thresholds:
            tdf = pd.DataFrame(
                [{"threshold": k, "value": float(v or 0.0)} for k, v in thresholds.items()]
            )
            st.dataframe(tdf, width="stretch", hide_index=True)
        if not mot_release_gate_checks_df.empty:
            st.dataframe(mot_release_gate_checks_df, width="stretch", hide_index=True)

    st.subheader("MOT Error Slices (Seq2 Gap)")
    if not mot_error_delta:
        st.info("No MOT error slices report found.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Delta MOTA", f"{mot_error_delta.get('mota', 0.0):+.4f}")
        d2.metric("Delta FP", int(mot_error_delta.get("fp", 0.0)))
        d3.metric("Delta FN", int(mot_error_delta.get("fn", 0.0)))
        d4.metric("Delta IDSW", int(mot_error_delta.get("id_switches", 0.0)))
        left, right = st.columns(2)
        with left:
            if mot_error_worst_df.empty:
                st.info("No worst-frame rows.")
            else:
                st.caption("Top problem frames on seq2")
                st.dataframe(mot_error_worst_df, width="stretch", hide_index=True)
        with right:
            if mot_error_tracks_df.empty:
                st.info("No IDSW track rows.")
            else:
                st.caption("Top GT tracks by ID switch count on seq2")
                st.dataframe(mot_error_tracks_df, width="stretch", hide_index=True)

    st.subheader("Run History")
    st.caption(f"DB backend: `{backend}`")
    if runs_df.empty:
        st.info("No runs found in metrics DB.")
    else:
        st.dataframe(runs_df, width="stretch", hide_index=True)
        selected_run = st.selectbox("Inspect run metrics", runs_df["run_id"].tolist())
        run_metrics = metrics_df[metrics_df["run_id"] == selected_run].copy()
        if run_metrics.empty:
            st.info("No metrics for selected run.")
        else:
            pivot = (
                run_metrics.pivot_table(
                    index="condition",
                    columns="metric_name",
                    values="metric_value",
                    aggfunc="mean",
                )
                .reset_index()
                .sort_values("condition")
            )
            st.dataframe(pivot, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
