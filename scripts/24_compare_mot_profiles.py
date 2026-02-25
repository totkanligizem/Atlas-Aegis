#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
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


def _load_mot_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mm = payload.get("mot_metrics", {})
    if not isinstance(mm, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in mm.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def _metric(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0) or 0.0)


def _w_seq2_recall(r: dict[str, float], s: dict[str, float]) -> str:
    rk = (_metric(r, "recall"), _metric(r, "hota"), _metric(r, "mota"), -_metric(r, "fp"))
    sk = (_metric(s, "recall"), _metric(s, "hota"), _metric(s, "mota"), -_metric(s, "fp"))
    if rk > sk:
        return "recall"
    if sk > rk:
        return "stability"
    return "tie"


def _w_seq2_stability(r: dict[str, float], s: dict[str, float]) -> str:
    rk = (_metric(r, "mota"), -_metric(r, "fp"), -_metric(r, "id_switches"), _metric(r, "hota"))
    sk = (_metric(s, "mota"), -_metric(s, "fp"), -_metric(s, "id_switches"), _metric(s, "hota"))
    if rk > sk:
        return "recall"
    if sk > rk:
        return "stability"
    return "tie"


def _w_seq2_hota(r: dict[str, float], s: dict[str, float]) -> str:
    rk = (_metric(r, "hota"), _metric(r, "assa"), _metric(r, "deta"), _metric(r, "mota"))
    sk = (_metric(s, "hota"), _metric(s, "assa"), _metric(s, "deta"), _metric(s, "mota"))
    if rk > sk:
        return "recall"
    if sk > rk:
        return "stability"
    return "tie"


def _to_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MOT Profile Comparison")
    lines.append("")
    w = report.get("winners", {})
    lines.append("## Winners")
    lines.append(f"- seq2_recall_priority: `{w.get('seq2_recall_priority', 'n/a')}`")
    lines.append(f"- seq2_stability_priority: `{w.get('seq2_stability_priority', 'n/a')}`")
    lines.append(f"- seq2_hota_priority: `{w.get('seq2_hota_priority', 'n/a')}`")
    lines.append("")
    lines.append("## Seq2 Snapshot")
    seq2 = report.get("profiles", {}).get("seq2", {})
    rec = seq2.get("recall", {})
    st = seq2.get("stability", {})
    lines.append(
        f"- recall profile: MOTA={rec.get('mota', 0.0):.4f}, HOTA={rec.get('hota', 0.0):.4f}, "
        f"recall={rec.get('recall', 0.0):.4f}, FP={rec.get('fp', 0.0):.0f}, IDSW={rec.get('id_switches', 0.0):.0f}"
    )
    lines.append(
        f"- stability profile: MOTA={st.get('mota', 0.0):.4f}, HOTA={st.get('hota', 0.0):.4f}, "
        f"recall={st.get('recall', 0.0):.4f}, FP={st.get('fp', 0.0):.0f}, IDSW={st.get('id_switches', 0.0):.0f}"
    )
    lines.append("")
    lines.append("## Metrics (recall - stability)")
    lines.append("| sequence | metric | recall | stability | delta |")
    lines.append("|---|---|---:|---:|---:|")
    for seq in ("seq1", "seq2"):
        delta = report.get("delta_recall_minus_stability", {}).get(seq, {})
        rec_m = report.get("profiles", {}).get(seq, {}).get("recall", {})
        st_m = report.get("profiles", {}).get(seq, {}).get("stability", {})
        for k in METRIC_KEYS:
            lines.append(
                f"| {seq} | {k} | {float(rec_m.get(k, 0.0)):.4f} | "
                f"{float(st_m.get(k, 0.0)):.4f} | {float(delta.get(k, 0.0)):+.4f} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recall-seq1", required=True)
    p.add_argument("--recall-seq2", required=True)
    p.add_argument("--stability-seq1", required=True)
    p.add_argument("--stability-seq2", required=True)
    p.add_argument("--out-json", default="reports/mot_profile_comparison_report.json")
    p.add_argument("--out-md", default="reports/mot_profile_comparison_report.md")
    args = p.parse_args()

    recall_seq1 = _load_mot_metrics(Path(args.recall_seq1))
    recall_seq2 = _load_mot_metrics(Path(args.recall_seq2))
    stability_seq1 = _load_mot_metrics(Path(args.stability_seq1))
    stability_seq2 = _load_mot_metrics(Path(args.stability_seq2))

    delta_seq1: dict[str, float] = {}
    delta_seq2: dict[str, float] = {}
    for k in METRIC_KEYS:
        delta_seq1[k] = _metric(recall_seq1, k) - _metric(stability_seq1, k)
        delta_seq2[k] = _metric(recall_seq2, k) - _metric(stability_seq2, k)

    report: dict[str, Any] = {
        "status": "SUCCESS",
        "inputs": {
            "recall_seq1": args.recall_seq1,
            "recall_seq2": args.recall_seq2,
            "stability_seq1": args.stability_seq1,
            "stability_seq2": args.stability_seq2,
        },
        "profiles": {
            "seq1": {"recall": recall_seq1, "stability": stability_seq1},
            "seq2": {"recall": recall_seq2, "stability": stability_seq2},
        },
        "delta_recall_minus_stability": {
            "seq1": delta_seq1,
            "seq2": delta_seq2,
        },
        "winners": {
            "seq2_recall_priority": _w_seq2_recall(recall_seq2, stability_seq2),
            "seq2_stability_priority": _w_seq2_stability(recall_seq2, stability_seq2),
            "seq2_hota_priority": _w_seq2_hota(recall_seq2, stability_seq2),
        },
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
                "winners": report["winners"],
                "out_json": str(out_json),
                "out_md": str(out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
