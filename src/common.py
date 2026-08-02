"""
Shared utility helpers used across LotteryAI.

Only pure helper functions belong in this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Read a configuration value while tolerating missing or None configs.
    """
    if not config:
        return default
    return config.get(key, default)


def normalize_float(value: Any, default: float = 0.0) -> float:
    """
    Convert values safely to float.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
