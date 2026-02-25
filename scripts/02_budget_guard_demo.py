#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.budget_guard import BudgetGuard, BudgetThresholds
from aerial_stack.config import load_yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    guard = BudgetGuard(
        BudgetThresholds(
            soft_warning_usd=float(cfg.get("soft_warning_usd", 8.0)),
            high_warning_usd=float(cfg.get("high_warning_usd", 12.0)),
            kill_switch_usd=float(cfg.get("kill_switch_usd", 14.0)),
            hard_cap_usd=float(cfg.get("hard_cap_usd", 15.0)),
        )
    )

    simulated_spend = [1.2, 2.7, 3.3, 1.1, 2.9, 2.0]
    events = [guard.add_spend(v, source="demo") for v in simulated_spend]
    print(json.dumps(events, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
