"""
JSON storage helpers for LotteryAI.

This module centralizes filesystem-based JSON input/output.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    """
    Load UTF-8 JSON from ``path``.

    Return ``default`` when the file does not exist, cannot be read,
    contains invalid JSON, or otherwise cannot be decoded.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    """
    Save ``data`` as indented UTF-8 JSON.

    Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_json_atomic(path: Path, data: Any) -> None:
    """
    Save JSON atomically by writing a temporary file and replacing ``path``.

    This prevents a partially written JSON file from remaining when a process
    is interrupted during the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(
                data,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
