#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _string_path(v: object) -> str:
    s = str(v or "").strip()
    return s


def _candidate_paths(obj: dict[str, object]) -> list[str]:
    out: list[str] = []

    artifacts = obj.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key in ("best", "last"):
            p = _string_path(artifacts.get(key))
            if p:
                out.append(p)

    cfg = obj.get("config", {})
    if isinstance(cfg, dict):
        p = _string_path(cfg.get("model"))
        if p:
            out.append(p)

    p = _string_path(obj.get("model"))
    if p:
        out.append(p)

    # Keep stable order while dropping duplicates.
    seen: set[str] = set()
    dedup: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _resolve_existing(candidates: list[str], *, report_path: Path) -> Path | None:
    for raw in candidates:
        p = Path(raw).expanduser()
        if p.exists():
            return p
        # If path is relative, resolve relative to report directory as fallback.
        if not p.is_absolute():
            pp = (report_path.parent / p).resolve()
            if pp.exists():
                return pp
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--report",
        required=True,
        help="Path to a train/eval report JSON that contains model artifact information.",
    )
    args = p.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    obj = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Report must be a JSON object: {report_path}")

    candidates = _candidate_paths(obj)
    if not candidates:
        raise RuntimeError(
            f"No model path found in report: {report_path}. "
            "Expected one of: artifacts.best, artifacts.last, config.model, model"
        )

    resolved = _resolve_existing(candidates, report_path=report_path)
    if resolved is None:
        raise FileNotFoundError(
            "Model path candidates were found but none exist on disk: "
            + ", ".join(candidates)
        )

    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
