#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_ENV: dict[str, dict[str, str]] = {
    "recall": {
        "MOT_PROFILE": "recall",
        "MOT_MIN_TRACK_AGE": "6",
        "MOT_MIN_CONF": "0.30",
        "MOT_MIN_CONF_RELAXED": "-1",
        "MOT_MIN_CONF_RELAX_AGE_START": "0",
        "MOT_MIN_ROI_DWELL": "0",
        "MOT_CLASS_MIN_CONF_MAP": "1:0.45,4:0.34,9:0.25",
    },
    "stability": {
        "MOT_PROFILE": "stability",
        "MOT_MIN_TRACK_AGE": "6",
        "MOT_MIN_CONF": "0.30",
        "MOT_MIN_CONF_RELAXED": "-1",
        "MOT_MIN_CONF_RELAX_AGE_START": "0",
        "MOT_MIN_ROI_DWELL": "0",
        "MOT_CLASS_MIN_CONF_MAP": "1:0.45,4:0.34,9:0.30",
    },
}


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"comparison report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("comparison report must be a JSON object")
    return payload


def _metrics(payload: dict[str, Any], profile: str) -> dict[str, float]:
    mm = (
        payload.get("profiles", {})
        .get("seq2", {})
        .get(profile, {})
    )
    if not isinstance(mm, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in mm.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def _check_profile(
    metrics: dict[str, float],
    *,
    min_recall: float,
    min_hota: float,
    min_mota: float,
    max_fp: float,
    max_idsw: float,
) -> dict[str, Any]:
    recall = float(metrics.get("recall", 0.0) or 0.0)
    hota = float(metrics.get("hota", 0.0) or 0.0)
    mota = float(metrics.get("mota", 0.0) or 0.0)
    fp = float(metrics.get("fp", 0.0) or 0.0)
    idsw = float(metrics.get("id_switches", 0.0) or 0.0)

    checks = {
        "seq2_recall_min": {
            "value": recall,
            "threshold": float(min_recall),
            "passed": bool(recall >= min_recall),
        },
        "seq2_hota_min": {
            "value": hota,
            "threshold": float(min_hota),
            "passed": bool(hota >= min_hota),
        },
        "seq2_mota_min": {
            "value": mota,
            "threshold": float(min_mota),
            "passed": bool(mota >= min_mota),
        },
    }
    if max_fp >= 0.0:
        checks["seq2_fp_max"] = {
            "value": fp,
            "threshold": float(max_fp),
            "passed": bool(fp <= max_fp),
        }
    if max_idsw >= 0.0:
        checks["seq2_idsw_max"] = {
            "value": idsw,
            "threshold": float(max_idsw),
            "passed": bool(idsw <= max_idsw),
        }

    overall = all(bool(c.get("passed", False)) for c in checks.values())
    return {
        "pass": bool(overall),
        "checks": checks,
    }


def _winner_for_objective(payload: dict[str, Any], objective: str) -> str:
    w = payload.get("winners", {})
    if not isinstance(w, dict):
        return "tie"
    if objective == "recall":
        out = str(w.get("seq2_recall_priority", "tie"))
    elif objective == "stability":
        out = str(w.get("seq2_stability_priority", "tie"))
    else:
        out = str(w.get("seq2_hota_priority", "tie"))
    if out not in ("recall", "stability"):
        return "tie"
    return out


def _fallback_best(metrics_by_profile: dict[str, dict[str, float]]) -> str:
    rk = {}
    for name, mm in metrics_by_profile.items():
        rk[name] = (
            float(mm.get("hota", 0.0) or 0.0),
            float(mm.get("mota", 0.0) or 0.0),
            float(mm.get("recall", 0.0) or 0.0),
            -float(mm.get("fp", 0.0) or 0.0),
            -float(mm.get("id_switches", 0.0) or 0.0),
        )
    return sorted(rk.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def _select_profile(
    payload: dict[str, Any],
    *,
    objective: str,
    min_recall: float,
    min_hota: float,
    min_mota: float,
    max_fp: float,
    max_idsw: float,
) -> dict[str, Any]:
    by_profile = {
        "recall": _metrics(payload, "recall"),
        "stability": _metrics(payload, "stability"),
    }
    checks = {
        name: _check_profile(
            metrics,
            min_recall=min_recall,
            min_hota=min_hota,
            min_mota=min_mota,
            max_fp=max_fp,
            max_idsw=max_idsw,
        )
        for name, metrics in by_profile.items()
    }

    passing = [name for name, c in checks.items() if bool(c.get("pass", False))]
    rationale: list[str] = []
    gate_status = "PASS"

    if len(passing) == 1:
        selected = passing[0]
        rationale.append(f"only `{selected}` satisfies all seq2 release checks")
    elif len(passing) == 2:
        preferred = _winner_for_objective(payload, objective)
        if preferred in ("recall", "stability"):
            selected = preferred
            rationale.append(
                f"both profiles passed; objective=`{objective}` winner is `{preferred}`"
            )
        else:
            selected = _fallback_best(by_profile)
            rationale.append(
                f"both profiles passed; winner was tie, fallback selected `{selected}` by "
                "hota/mota/recall/fp/idsw ranking"
            )
    else:
        selected = _fallback_best(by_profile)
        gate_status = "FAIL"
        rationale.append(
            "no profile satisfies all release checks; fallback selected by "
            "hota/mota/recall/fp/idsw ranking"
        )

    return {
        "objective": objective,
        "gate_status": gate_status,
        "selected_profile": selected,
        "selected_env": PROFILE_ENV[selected],
        "rationale": rationale,
        "profiles": {
            "recall": {
                "metrics_seq2": by_profile["recall"],
                "gate": checks["recall"],
            },
            "stability": {
                "metrics_seq2": by_profile["stability"],
                "gate": checks["stability"],
            },
        },
        "thresholds": {
            "min_seq2_recall": float(min_recall),
            "min_seq2_hota": float(min_hota),
            "min_seq2_mota": float(min_mota),
            "max_seq2_fp": float(max_fp),
            "max_seq2_idsw": float(max_idsw),
        },
    }


def _to_md(report: dict[str, Any]) -> str:
    sel = report.get("selection", {})
    thresholds = sel.get("thresholds", {})
    profiles = sel.get("profiles", {})
    lines: list[str] = []
    lines.append("# MOT Release Profile Gate")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- gate_status: `{sel.get('gate_status', 'UNKNOWN')}`")
    lines.append(f"- objective: `{sel.get('objective', 'hota')}`")
    lines.append(f"- selected_profile: `{sel.get('selected_profile', 'recall')}`")
    lines.append("- rationale:")
    for r in sel.get("rationale", []):
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("## Release Thresholds")
    lines.append(
        "- "
        f"min_seq2_recall={thresholds.get('min_seq2_recall', 0.0):.4f}, "
        f"min_seq2_hota={thresholds.get('min_seq2_hota', 0.0):.4f}, "
        f"min_seq2_mota={thresholds.get('min_seq2_mota', 0.0):.4f}, "
        f"max_seq2_fp={thresholds.get('max_seq2_fp', -1.0):.1f}, "
        f"max_seq2_idsw={thresholds.get('max_seq2_idsw', -1.0):.1f}"
    )
    lines.append("")
    lines.append("## Seq2 Profile Checks")
    lines.append("| profile | metric | value | threshold | passed |")
    lines.append("|---|---|---:|---:|---:|")
    for profile in ("recall", "stability"):
        gate = profiles.get(profile, {}).get("gate", {})
        checks = gate.get("checks", {})
        for name, chk in checks.items():
            if not isinstance(chk, dict):
                continue
            lines.append(
                f"| {profile} | {name} | {float(chk.get('value', 0.0)):.4f} | "
                f"{float(chk.get('threshold', 0.0)):.4f} | "
                f"{'1' if bool(chk.get('passed', False)) else '0'} |"
            )
    lines.append("")
    lines.append("## Selected Env")
    env = sel.get("selected_env", {})
    for k in sorted(env):
        lines.append(f"- `{k}={env[k]}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_env(path: Path, report: dict[str, Any]) -> None:
    sel = report.get("selection", {})
    env = sel.get("selected_env", {})
    lines: list[str] = []
    if isinstance(env, dict):
        for k in sorted(env):
            lines.append(f'{k}="{env[k]}"')
    lines.append(f'MOT_RELEASE_GATE_STATUS="{sel.get("gate_status", "UNKNOWN")}"')
    lines.append(f'MOT_RELEASE_OBJECTIVE="{sel.get("objective", "hota")}"')
    lines.append(f'MOT_RELEASE_SELECTED_PROFILE="{sel.get("selected_profile", "recall")}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", default="reports/mot_profile_comparison_report.json")
    p.add_argument("--objective", choices=["hota", "recall", "stability"], default="hota")
    p.add_argument("--min-seq2-recall", type=float, default=0.333)
    p.add_argument("--min-seq2-hota", type=float, default=0.300)
    p.add_argument("--min-seq2-mota", type=float, default=-0.080)
    p.add_argument("--max-seq2-fp", type=float, default=-1.0)
    p.add_argument("--max-seq2-idsw", type=float, default=-1.0)
    p.add_argument("--fail-on-no-go", action="store_true")
    p.add_argument("--out-json", default="reports/mot_profile_release_gate_report.json")
    p.add_argument("--out-md", default="reports/mot_profile_release_gate_report.md")
    p.add_argument("--out-env", default="reports/mot_profile_release.env")
    args = p.parse_args()

    comparison_path = Path(args.comparison)
    payload = _load_report(comparison_path)
    selection = _select_profile(
        payload,
        objective=str(args.objective),
        min_recall=float(args.min_seq2_recall),
        min_hota=float(args.min_seq2_hota),
        min_mota=float(args.min_seq2_mota),
        max_fp=float(args.max_seq2_fp),
        max_idsw=float(args.max_seq2_idsw),
    )
    selected_profile = str(selection.get("selected_profile", "recall"))
    gate_status = str(selection.get("gate_status", "FAIL"))

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_env = Path(args.out_env)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_env.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "SUCCESS",
        "inputs": {
            "comparison": str(comparison_path),
            "objective": str(args.objective),
            "min_seq2_recall": float(args.min_seq2_recall),
            "min_seq2_hota": float(args.min_seq2_hota),
            "min_seq2_mota": float(args.min_seq2_mota),
            "max_seq2_fp": float(args.max_seq2_fp),
            "max_seq2_idsw": float(args.max_seq2_idsw),
            "fail_on_no_go": bool(args.fail_on_no_go),
        },
        "selection": selection,
        "release_command": (
            f"MOT_PROFILE={selected_profile} make build-mot-visdrone-val "
            "build-mot-visdrone-val-seq2 mot-eval-visdrone-val mot-eval-visdrone-val-seq2"
        ),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    _write_env(out_env, report)

    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "gate_status": gate_status,
                "selected_profile": selected_profile,
                "out_json": str(out_json),
                "out_md": str(out_md),
                "out_env": str(out_env),
            },
            indent=2,
        )
    )
    if bool(args.fail_on_no_go) and gate_status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
