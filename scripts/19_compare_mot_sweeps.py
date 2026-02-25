#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _key(params: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(params.get("conf", 0.0) or 0.0),
        float(params.get("iou", 0.0) or 0.0),
        int(float(params.get("track_ttl_frames", 0) or 0)),
    )


def _format_params(k: tuple[float, float, int]) -> str:
    return f"conf={k[0]:.2f}, iou={k[1]:.2f}, ttl={k[2]}"


def _rank_row(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("mota_mean", 0.0)),
        float(row.get("precision_mean", 0.0)),
        -float(row.get("idsw_mean", 0.0)),
    )


def _row_from_run(run: dict[str, Any]) -> dict[str, Any]:
    params = run.get("params", {})
    mm = run.get("mot_metrics", {})
    return {
        "params": {
            "conf": float(params.get("conf", 0.0) or 0.0),
            "iou": float(params.get("iou", 0.0) or 0.0),
            "track_ttl_frames": int(float(params.get("track_ttl_frames", 0) or 0)),
        },
        "mota": float(mm.get("mota", 0.0) or 0.0),
        "precision": float(mm.get("precision", 0.0) or 0.0),
        "recall": float(mm.get("recall", 0.0) or 0.0),
        "id_switches": float(mm.get("id_switches", 0.0) or 0.0),
    }


def _build_markdown(report: dict[str, Any]) -> str:
    seq1 = report["seq1"]["best_run"]
    seq2 = report["seq2"]["best_run"]
    joint = report["joint_best"]
    lines: list[str] = []
    lines.append("# MOT Sweep Generalization Comparison")
    lines.append("")
    lines.append("## Sequence Winners")
    lines.append(
        f"- seq1 winner: `{_format_params(_key(seq1['params']))}` "
        f"(MOTA={seq1['metrics']['mota']:.4f}, "
        f"precision={seq1['metrics']['precision']:.4f}, "
        f"recall={seq1['metrics']['recall']:.4f}, "
        f"IDSW={seq1['metrics']['id_switches']:.0f})"
    )
    lines.append(
        f"- seq2 winner: `{_format_params(_key(seq2['params']))}` "
        f"(MOTA={seq2['metrics']['mota']:.4f}, "
        f"precision={seq2['metrics']['precision']:.4f}, "
        f"recall={seq2['metrics']['recall']:.4f}, "
        f"IDSW={seq2['metrics']['id_switches']:.0f})"
    )
    lines.append("")
    lines.append("## Joint Best (mean ranking across seq1+seq2)")
    lines.append(
        f"- params: `{_format_params(_key(joint['params']))}`; "
        f"mota_mean={joint['mota_mean']:.4f}, "
        f"precision_mean={joint['precision_mean']:.4f}, "
        f"recall_mean={joint['recall_mean']:.4f}, "
        f"idsw_mean={joint['idsw_mean']:.1f}"
    )
    lines.append("")
    lines.append("## Top 5 Joint Candidates")
    lines.append("| rank | params | mota_mean | precision_mean | recall_mean | idsw_mean |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(report["joint_top5"], 1):
        lines.append(
            f"| {i} | `{_format_params(_key(r['params']))}` | "
            f"{r['mota_mean']:.4f} | {r['precision_mean']:.4f} | "
            f"{r['recall_mean']:.4f} | {r['idsw_mean']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq1", default="reports/mot_tuning_sweep_visdrone_val_report.json")
    p.add_argument("--seq2", default="reports/mot_tuning_sweep_visdrone_val_seq2_report.json")
    p.add_argument("--out-json", default="reports/mot_sweep_generalization_comparison_report.json")
    p.add_argument("--out-md", default="reports/mot_sweep_generalization_comparison_report.md")
    args = p.parse_args()

    s1 = _load(Path(args.seq1))
    s2 = _load(Path(args.seq2))

    s1_runs = [_row_from_run(r) for r in s1.get("runs", []) if isinstance(r, dict)]
    s2_runs = [_row_from_run(r) for r in s2.get("runs", []) if isinstance(r, dict)]
    s1_map = {_key(r["params"]): r for r in s1_runs}
    s2_map = {_key(r["params"]): r for r in s2_runs}
    common_keys = sorted(set(s1_map.keys()) & set(s2_map.keys()))
    if not common_keys:
        raise RuntimeError("No common parameter sets found between seq1 and seq2 reports.")

    joint_rows: list[dict[str, Any]] = []
    for k in common_keys:
        a = s1_map[k]
        b = s2_map[k]
        row = {
            "params": {
                "conf": k[0],
                "iou": k[1],
                "track_ttl_frames": k[2],
            },
            "seq1": {
                "mota": a["mota"],
                "precision": a["precision"],
                "recall": a["recall"],
                "id_switches": a["id_switches"],
            },
            "seq2": {
                "mota": b["mota"],
                "precision": b["precision"],
                "recall": b["recall"],
                "id_switches": b["id_switches"],
            },
            "mota_mean": (a["mota"] + b["mota"]) / 2.0,
            "precision_mean": (a["precision"] + b["precision"]) / 2.0,
            "recall_mean": (a["recall"] + b["recall"]) / 2.0,
            "idsw_mean": (a["id_switches"] + b["id_switches"]) / 2.0,
        }
        joint_rows.append(row)

    joint_sorted = sorted(joint_rows, key=_rank_row, reverse=True)
    joint_best = joint_sorted[0]

    seq1_best_src = s1.get("best_run", {})
    seq2_best_src = s2.get("best_run", {})
    report = {
        "status": "SUCCESS",
        "inputs": {
            "seq1": args.seq1,
            "seq2": args.seq2,
        },
        "seq1": {
            "best_run": {
                "params": {
                    "conf": float(seq1_best_src.get("params", {}).get("conf", 0.0) or 0.0),
                    "iou": float(seq1_best_src.get("params", {}).get("iou", 0.0) or 0.0),
                    "track_ttl_frames": int(
                        float(seq1_best_src.get("params", {}).get("track_ttl_frames", 0) or 0)
                    ),
                },
                "metrics": {
                    "mota": float(seq1_best_src.get("mot_metrics", {}).get("mota", 0.0) or 0.0),
                    "precision": float(
                        seq1_best_src.get("mot_metrics", {}).get("precision", 0.0) or 0.0
                    ),
                    "recall": float(seq1_best_src.get("mot_metrics", {}).get("recall", 0.0) or 0.0),
                    "id_switches": float(
                        seq1_best_src.get("mot_metrics", {}).get("id_switches", 0.0) or 0.0
                    ),
                },
            }
        },
        "seq2": {
            "best_run": {
                "params": {
                    "conf": float(seq2_best_src.get("params", {}).get("conf", 0.0) or 0.0),
                    "iou": float(seq2_best_src.get("params", {}).get("iou", 0.0) or 0.0),
                    "track_ttl_frames": int(
                        float(seq2_best_src.get("params", {}).get("track_ttl_frames", 0) or 0)
                    ),
                },
                "metrics": {
                    "mota": float(seq2_best_src.get("mot_metrics", {}).get("mota", 0.0) or 0.0),
                    "precision": float(
                        seq2_best_src.get("mot_metrics", {}).get("precision", 0.0) or 0.0
                    ),
                    "recall": float(seq2_best_src.get("mot_metrics", {}).get("recall", 0.0) or 0.0),
                    "id_switches": float(
                        seq2_best_src.get("mot_metrics", {}).get("id_switches", 0.0) or 0.0
                    ),
                },
            }
        },
        "joint_best": joint_best,
        "joint_top5": joint_sorted[:5],
        "joint_all": joint_sorted,
        "ranking_rule": "sort by (mota_mean desc, precision_mean desc, idsw_mean asc)",
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(json.dumps({"joint_best": joint_best, "seq1_best": report["seq1"], "seq2_best": report["seq2"]}, indent=2))
    print(f"wrote report: {out_json}")
    print(f"wrote report: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
