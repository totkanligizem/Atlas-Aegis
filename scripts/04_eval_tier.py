#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.config import load_yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tier", choices=["fast", "candidate", "full"], required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tiers = cfg.get("loop_tiers", {})
    item = tiers.get(args.tier)
    if not item:
        raise SystemExit(f"tier not found: {args.tier}")

    print(json.dumps({"tier": args.tier, **item}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
