#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_lines(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def _run_step(name: str, cmd: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = "\n".join([x for x in [out, err] if x]).strip()
    return {
        "name": name,
        "command": cmd,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "return_code": int(proc.returncode),
        "duration_sec": float(round(dt, 3)),
        "output_tail": _tail_lines(combined, n=60),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _latency_gate_info(path: Path) -> tuple[str, bool]:
    obj = _read_json(path)
    if not obj:
        return "UNKNOWN", False
    gate = str(obj.get("gate_status", "UNKNOWN"))
    profiles = obj.get("profiles", [])
    if not isinstance(profiles, list):
        profiles = []
    required_mps = False
    for row in profiles:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("required", False)):
            continue
        req = str(row.get("device_requested", "")).strip().lower()
        eff = str(row.get("device_effective", "")).strip().lower()
        if req == "mps" or eff == "mps":
            required_mps = True
            break
    return gate, required_mps


def _bool_val(obj: Any) -> bool:
    return bool(obj)


def _check_files(files: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in files:
        out.append(
            {
                "path": str(p),
                "exists": bool(p.exists()),
                "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            }
        )
    return out


def _mps_status() -> dict[str, Any]:
    script = (
        "import json, torch;"
        "print(json.dumps({'mps_available': bool(torch.backends.mps.is_available()),"
        "'mps_built': bool(torch.backends.mps.is_built())}))"
    )
    proc = subprocess.run(
        f".venv/bin/python -c \"{script}\"",
        shell=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "status": "UNKNOWN",
            "error": (proc.stderr or proc.stdout or "").strip(),
        }
    try:
        obj = json.loads((proc.stdout or "").strip())
        return {
            "status": "OK",
            "mps_available": bool(obj.get("mps_available", False)),
            "mps_built": bool(obj.get("mps_built", False)),
        }
    except Exception:
        return {
            "status": "UNKNOWN",
            "error": (proc.stdout or "").strip(),
        }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Pre-Release Quality Check")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- status: `{report.get('status', 'FAIL')}`")
    lines.append(f"- started_utc: `{report.get('started_utc', '')}`")
    lines.append(f"- ended_utc: `{report.get('ended_utc', '')}`")
    lines.append(f"- duration_sec: `{report.get('duration_sec', 0.0):.3f}`")
    lines.append("")
    lines.append("## Step Results")
    lines.append("| step | status | duration_sec | command |")
    lines.append("|---|---|---:|---|")
    for s in report.get("steps", []):
        lines.append(
            f"| {s.get('name', '')} | {s.get('status', 'FAIL')} | "
            f"{float(s.get('duration_sec', 0.0)):.3f} | `{s.get('command', '')}` |"
        )
    lines.append("")
    lines.append("## Artifact Checks")
    lines.append("| path | exists | size_bytes |")
    lines.append("|---|---:|---:|")
    for f in report.get("artifacts", []):
        lines.append(
            f"| `{f.get('path', '')}` | {1 if f.get('exists', False) else 0} | "
            f"{int(f.get('size_bytes', 0))} |"
        )
    lines.append("")
    gate = report.get("release_gate", {})
    lines.append("## Release Gate")
    lines.append(f"- status: `{gate.get('status', 'UNKNOWN')}`")
    lines.append(f"- selected_profile: `{gate.get('selected_profile', 'n/a')}`")
    lines.append("")
    mps = report.get("mps", {})
    lines.append("## Device Check")
    lines.append(
        f"- mps_available: `{mps.get('mps_available', False)}`; "
        f"mps_built: `{mps.get('mps_built', False)}`"
    )
    warns = report.get("warnings", [])
    if isinstance(warns, list) and warns:
        lines.append("")
        lines.append("## Warnings")
        for w in warns:
            lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", default="reports/pre_release_check_report.json")
    p.add_argument("--out-md", default="reports/pre_release_check_report.md")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failure.")
    args = p.parse_args()

    start_iso = _now_iso()
    t0 = time.perf_counter()
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []

    commands: list[tuple[str, str]] = [
        ("env-check", "make env-check"),
        ("dataset-status", "make dataset-status"),
        ("smoke", "make smoke"),
        ("bench-fast-dry", "make bench-fast-dry"),
        ("latency-bench-dry", "make latency-bench-dry"),
        ("export-bench-dry", "make export-bench-dry"),
        ("eval-det-dry", "make eval-det-dry"),
        ("mot-eval-dry", "make mot-eval-dry"),
        ("mot-release-run", "make mot-release-run"),
        ("compare-mot-sweeps", "make compare-mot-sweeps"),
        ("mot-error-slices", "make mot-error-slices"),
    ]

    overall_ok = True
    for name, cmd in commands:
        step = _run_step(name, cmd)
        steps.append(step)
        print(f"[pre-release] {name}: {step['status']} ({step['duration_sec']:.3f}s)")
        if step["status"] != "PASS":
            overall_ok = False
            break

    artifact_paths = [
        ROOT / "reports/smoke_run.json",
        ROOT / "reports/benchmark_fast_dry_run_report.json",
        ROOT / "reports/latency_benchmark_dry_run_report.json",
        ROOT / "reports/latency_benchmark_dry_run_report.md",
        ROOT / "reports/export_benchmark_dry_run_report.json",
        ROOT / "reports/export_benchmark_dry_run_report.md",
        ROOT / "reports/detector_eval_report.json",
        ROOT / "reports/mot_eval_report.json",
        ROOT / "reports/mot_profile_release_gate_report.json",
        ROOT / "reports/mot_profile_release.env",
        ROOT / "reports/mot_eval_visdrone_val_report.json",
        ROOT / "reports/mot_eval_visdrone_val_seq2_report.json",
    ]
    artifacts = _check_files(artifact_paths)
    missing = [a["path"] for a in artifacts if not _bool_val(a.get("exists", False))]
    if missing:
        overall_ok = False
        warnings.append("Missing required artifacts: " + ", ".join(missing))

    gate_report = _read_json(ROOT / "reports/mot_profile_release_gate_report.json")
    gate_status = "UNKNOWN"
    selected_profile = "n/a"
    if gate_report and isinstance(gate_report.get("selection"), dict):
        sel = gate_report["selection"]
        gate_status = str(sel.get("gate_status", "UNKNOWN"))
        selected_profile = str(sel.get("selected_profile", "n/a"))
        if gate_status != "PASS":
            overall_ok = False
            warnings.append(f"Release gate is {gate_status}")
    else:
        overall_ok = False
        warnings.append("Release gate report missing or unreadable")

    latency_gate, latency_requires_mps = _latency_gate_info(
        ROOT / "reports/latency_benchmark_report.json"
    )
    if latency_gate != "PASS":
        overall_ok = False
        warnings.append(f"Latency benchmark gate is {latency_gate}")

    mps = _mps_status()
    if (
        bool(mps.get("mps_built", False))
        and not bool(mps.get("mps_available", False))
        and bool(latency_requires_mps)
    ):
        warnings.append("torch MPS built but not available; training/eval will run slower on CPU.")

    if args.strict and warnings:
        overall_ok = False

    end_iso = _now_iso()
    duration = time.perf_counter() - t0
    report: dict[str, Any] = {
        "status": "PASS" if overall_ok else "FAIL",
        "started_utc": start_iso,
        "ended_utc": end_iso,
        "duration_sec": float(round(duration, 3)),
        "strict_mode": bool(args.strict),
        "steps": steps,
        "artifacts": artifacts,
        "release_gate": {
            "status": gate_status,
            "selected_profile": selected_profile,
        },
        "mps": mps,
        "warnings": warnings,
    }

    out_json = ROOT / str(args.out_json)
    out_md = ROOT / str(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(json.dumps({"status": report["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
