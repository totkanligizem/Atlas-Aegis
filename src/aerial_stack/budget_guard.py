from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetThresholds:
    soft_warning_usd: float = 8.0
    high_warning_usd: float = 12.0
    kill_switch_usd: float = 14.0
    hard_cap_usd: float = 15.0


class BudgetGuard:
    def __init__(self, thresholds: BudgetThresholds) -> None:
        self.thresholds = thresholds
        self.spent_usd = 0.0

    def add_spend(self, amount_usd: float, source: str = "unknown") -> dict[str, str | float | bool]:
        if amount_usd < 0:
            raise ValueError("amount_usd must be non-negative")
        self.spent_usd += amount_usd

        status = "ok"
        stop_api = False

        if self.spent_usd >= self.thresholds.hard_cap_usd:
            status = "hard_cap_exceeded"
            stop_api = True
        elif self.spent_usd >= self.thresholds.kill_switch_usd:
            status = "kill_switch"
            stop_api = True
        elif self.spent_usd >= self.thresholds.high_warning_usd:
            status = "high_warning"
        elif self.spent_usd >= self.thresholds.soft_warning_usd:
            status = "soft_warning"

        return {
            "source": source,
            "spent_usd": round(self.spent_usd, 4),
            "status": status,
            "stop_api": stop_api,
        }
