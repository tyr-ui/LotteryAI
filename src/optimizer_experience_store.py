"""
Persistence helpers for Optimizer Experience.

This module owns filesystem I/O and the outer Experience-store schema.
Game-specific normalization and statistics remain in optimizer_experience.py
until later Phase 4 refactoring stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from storage import load_json, save_json_atomic


def empty_experience_store(
    schema_version: str,
) -> dict[str, object]:
    """Return an empty Experience store using the requested schema version."""
    return {
        "schema_version": str(schema_version),
        "games": {},
    }


def load_experience_store(
    path: Path,
    schema_version: str,
) -> dict[str, object]:
    """
    Load and normalize the outer Optimizer Experience store.

    Missing, unreadable, malformed, or non-object JSON is treated as an empty
    store. The returned schema_version is always the current version supplied
    by the caller, while existing game data and updated_at are preserved.
    """
    loaded = load_json(path, default=None)

    if not isinstance(loaded, Mapping):
        return empty_experience_store(
            schema_version
        )

    games = loaded.get("games")
    if not isinstance(games, Mapping):
        games = {}

    return {
        "schema_version": str(
            schema_version
        ),
        "updated_at": loaded.get(
            "updated_at"
        ),
        "games": dict(games),
    }


def save_experience_store(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    """
    Atomically save an Optimizer Experience store as UTF-8 JSON.
    """
    save_json_atomic(
        path,
        dict(data),
    )


__all__ = [
    "empty_experience_store",
    "load_experience_store",
    "save_experience_store",
]
