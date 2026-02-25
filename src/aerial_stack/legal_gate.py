from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_yaml


class LegalGateError(RuntimeError):
    """Raised when legal gate checks fail."""


def check_dataset_usage(
    *,
    legal_config_path: str | Path,
    dataset_key: str,
    usage_purpose: str,
) -> dict[str, Any]:
    key = str(dataset_key).strip()
    if not key:
        raise LegalGateError("dataset_key is required for legal gate checks.")

    purpose = str(usage_purpose).strip().lower()
    if purpose not in {"research", "commercial"}:
        raise LegalGateError("usage_purpose must be 'research' or 'commercial'.")

    cfg = load_yaml(legal_config_path)
    datasets = cfg.get("datasets", {})
    if not isinstance(datasets, dict):
        raise LegalGateError("Invalid legal_status file: 'datasets' must be a mapping.")

    if key not in datasets:
        available = ", ".join(sorted(str(k) for k in datasets.keys()))
        raise LegalGateError(
            f"Dataset key not found in legal status: {key}. Available: {available}"
        )

    entry = datasets[key]
    if not isinstance(entry, dict):
        raise LegalGateError(f"Invalid dataset entry in legal status: {key}")

    allow_field = "allowed_research" if purpose == "research" else "allowed_commercial"
    allowed = bool(entry.get(allow_field, False))
    if not allowed:
        note = str(entry.get("notes", "")).strip()
        raise PermissionError(
            f"Legal gate failed: dataset '{key}' is not allowed for {purpose}. "
            f"{note}".strip()
        )

    return {
        "status": "PASS",
        "dataset_key": key,
        "usage_purpose": purpose,
        "allowed_research": bool(entry.get("allowed_research", False)),
        "allowed_commercial": bool(entry.get("allowed_commercial", False)),
        "attribution_required": bool(entry.get("attribution_required", True)),
        "notes": str(entry.get("notes", "")),
        "legal_config_path": str(Path(legal_config_path)),
    }
