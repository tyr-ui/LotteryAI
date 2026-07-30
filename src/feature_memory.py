from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping, Sequence


MEMORY_PATH = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "feature_memory.json"
)


def _load_memory() -> dict:
    if MEMORY_PATH.exists():
        try:
            with open(
                MEMORY_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "schema_version": "1.0",
        "history": {},
    }


def save_feature_memory(
    game_name: str,
    feature_ablation: Sequence[Mapping],
) -> None:
    """
    アブレーション結果を履歴として保存する。

    Optimizerには影響しない。
    """

    memory = _load_memory()

    history = memory.setdefault(
        "history",
        {},
    )

    game_history = history.setdefault(
        game_name,
        [],
    )

    game_history.append(
        {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "features": list(feature_ablation),
        }
    )

    MEMORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MEMORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            memory,
            f,
            ensure_ascii=False,
            indent=2,
        )