from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .budget_guard import BudgetGuard, BudgetThresholds
from .config import load_yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _thresholds_from_cfg(cfg: dict[str, Any]) -> BudgetThresholds:
    return BudgetThresholds(
        soft_warning_usd=float(cfg.get("soft_warning_usd", 8.0)),
        high_warning_usd=float(cfg.get("high_warning_usd", 12.0)),
        kill_switch_usd=float(cfg.get("kill_switch_usd", 14.0)),
        hard_cap_usd=float(cfg.get("hard_cap_usd", 15.0)),
    )


def _read_ledger_spent(ledger_path: Path) -> float:
    if not ledger_path.exists():
        return 0.0
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    try:
        return max(0.0, float(payload.get("spent_usd", 0.0)))
    except Exception:
        return 0.0


def _write_ledger(
    *,
    ledger_path: Path,
    spent_usd: float,
    status: str,
    source: str,
    budget_config_path: Path,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spent_usd": round(float(spent_usd), 6),
        "status": str(status),
        "last_source": str(source),
        "budget_config_path": str(budget_config_path),
        "updated_at_utc": _now_iso(),
    }
    ledger_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_event(event_path: Path, event: dict[str, Any]) -> None:
    event_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"))
    with event_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def apply_budget_guard(
    *,
    budget_config_path: str | Path,
    budget_ledger_path: str | Path,
    budget_events_path: str | Path,
    source: str,
    requested_spend_usd: float = 0.0,
    is_api_call: bool = False,
) -> dict[str, Any]:
    spend = float(requested_spend_usd)
    if spend < 0.0:
        raise ValueError("requested_spend_usd must be non-negative.")

    cfg_path = Path(budget_config_path)
    cfg = load_yaml(cfg_path)
    guard = BudgetGuard(_thresholds_from_cfg(cfg))

    ledger_path = Path(budget_ledger_path)
    existing_spent = _read_ledger_spent(ledger_path)
    guard.spent_usd = existing_spent

    event = guard.add_spend(amount_usd=spend, source=source)
    policy = cfg.get("policy", {})
    block_api = bool(policy.get("block_api_when_kill_switch_reached", True))
    blocked = bool(event.get("stop_api", False)) and bool(is_api_call) and block_api

    out_event: dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "source": str(source),
        "requested_spend_usd": round(spend, 6),
        "spent_usd": float(event.get("spent_usd", 0.0)),
        "status": str(event.get("status", "ok")),
        "stop_api": bool(event.get("stop_api", False)),
        "is_api_call": bool(is_api_call),
        "blocked": blocked,
        "budget_config_path": str(cfg_path),
    }

    _write_ledger(
        ledger_path=ledger_path,
        spent_usd=float(event.get("spent_usd", existing_spent)),
        status=str(event.get("status", "ok")),
        source=source,
        budget_config_path=cfg_path,
    )
    _append_event(Path(budget_events_path), out_event)
    return out_event
