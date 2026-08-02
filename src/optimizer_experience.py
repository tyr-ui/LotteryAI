from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from optimizer_experience_store import load_experience_store, save_experience_store
from optimizer_experience_stats import (
    _build_config_statistics,
    _build_search_source_statistics,
    _entry_sort_key,
    _resolve_search_source,
)
from optimizer_adaptation import (
    build_evolution_adaptation,
    build_search_allocation,
)


OUTPUT_DIR = Path("output")
EXPERIENCE_PATH = (
    OUTPUT_DIR / "optimizer_experience.json"
)

SCHEMA_VERSION = "1.3"
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_EXPERIENCE_LIMIT = 3

DEFAULT_SEARCH_ALLOCATION = {
    "experience": 3,
    "random": 4,
    "local": 6,
    "evolution": 4,
}
MIN_SEARCH_ALLOCATION_SAMPLES = 5

DEFAULT_EVOLUTION_COUNT = 4
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_MUTATION_SCALE = 0.08
MIN_ADAPTATION_SAMPLES = 5


def _load_store() -> dict[str, object]:
    """
    Optimizer Experienceの保存データを読み込む。

    外側のJSON読込・破損時フォールバック・schema更新は
    optimizer_experience_storeへ委譲する。
    """
    return load_experience_store(
        EXPERIENCE_PATH,
        SCHEMA_VERSION,
    )


def _normalize_json_value(
    value: object,
) -> object:
    """
    Config等に含まれる値を
    JSONへ安全に保存できる形式へ変換する。
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, Sequence):
        if isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return str(value)

        return [
            _normalize_json_value(item)
            for item in value
        ]

    return str(value)


def _normalize_float(
    value: object,
    *,
    default: float = 0.0,
) -> float:
    """
    数値を有限floatへ正規化する。
    """
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return float(default)

    if not math.isfinite(normalized):
        return float(default)

    return normalized


def _normalize_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """
    Optimizer Configを保存可能な形式へ正規化する。

    Experienceから再利用するため、
    wとfを必ず保持する。
    """
    weights = config.get("w")
    filters = config.get("f", {})

    if not isinstance(weights, Mapping):
        raise ValueError(
            "Optimizer config must contain "
            "a mapping named 'w'."
        )

    if not isinstance(filters, Mapping):
        filters = {}

    normalized_weights = {
        str(key): _normalize_json_value(
            value
        )
        for key, value in weights.items()
    }

    if not normalized_weights:
        raise ValueError(
            "Optimizer config weights "
            "must not be empty."
        )

    normalized_filters = {
        str(key): _normalize_json_value(
            value
        )
        for key, value in filters.items()
    }

    return {
        "w": normalized_weights,
        "f": normalized_filters,
    }


def _config_signature(
    config: Mapping[str, object],
) -> str:
    """
    Configの重複判定に使う安定した文字列を返す。
    """
    normalized = _normalize_config(
        config
    )

    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )




def _normalize_history_entry(
    entry: Mapping[str, object],
) -> dict[str, object] | None:
    """
    保存済みExperience履歴を正規化する。
    """
    config = entry.get("config")

    if not isinstance(config, Mapping):
        return None

    try:
        normalized_config = _normalize_config(
            config
        )
    except ValueError:
        return None

    return {
        "evaluated_at": entry.get(
            "evaluated_at"
        ),
        "config_name": str(
            entry.get(
                "config_name",
                "unknown",
            )
        ),
        "selection_score": round(
            _normalize_float(
                entry.get(
                    "selection_score"
                )
            ),
            6,
        ),
        "avg_matches": round(
            _normalize_float(
                entry.get("avg_matches")
            ),
            6,
        ),
        "average_matches_per_ticket": (
            round(
                _normalize_float(
                    entry.get(
                        "average_matches_per_ticket"
                    )
                ),
                6,
            )
        ),
        "hit_rate_2match": round(
            _normalize_float(
                entry.get(
                    "hit_rate_2match"
                )
            ),
            6,
        ),
        "hit_rate_3match": round(
            _normalize_float(
                entry.get(
                    "hit_rate_3match"
                )
            ),
            6,
        ),
        "hit_rate_4match": round(
            _normalize_float(
                entry.get(
                    "hit_rate_4match"
                )
            ),
            6,
        ),
        "avg_matches_std": round(
            _normalize_float(
                entry.get(
                    "avg_matches_std"
                )
            ),
            6,
        ),
        "random_uplift": round(
            _normalize_float(
                entry.get(
                    "random_uplift"
                )
            ),
            6,
        ),
        "learning_strength": round(
            _normalize_float(
                entry.get(
                    "learning_strength"
                )
            ),
            6,
        ),
        "config": normalized_config,
        "prediction_weights": (
            _normalize_json_value(
                entry.get(
                    "prediction_weights",
                    {},
                )
            )
        ),
    }

def _load_normalized_history(
    game_name: str,
) -> list[dict[str, object]]:
    """
    指定ゲームのExperience履歴を読み込み、判定用形式へ正規化する。
    """
    store = _load_store()
    games = store.get("games")

    if not isinstance(games, Mapping):
        return []

    game_store = games.get(
        str(game_name)
    )

    if not isinstance(
        game_store,
        Mapping,
    ):
        return []

    history = game_store.get("history")

    if not isinstance(history, list):
        return []

    normalized_history: list[
        dict[str, object]
    ] = []

    for item in history:
        if not isinstance(
            item,
            Mapping,
        ):
            continue

        normalized = (
            _normalize_history_entry(
                item
            )
        )

        if normalized is not None:
            normalized_history.append(
                normalized
            )

    return normalized_history


def load_evolution_adaptation(
    game_name: str,
) -> dict[str, object]:
    """
    保存済み履歴を読み込み、Evolution適応判定を返す。
    """
    return build_evolution_adaptation(
        _load_normalized_history(
            game_name
        )
    )


def load_search_allocation(
    game_name: str,
) -> dict[str, object]:
    """
    保存済み履歴を読み込み、探索元ごとの配分を返す。
    """
    return build_search_allocation(
        _load_normalized_history(
            game_name
        )
    )



def load_experience_configs(
    game_name: str,
    *,
    limit: int = DEFAULT_EXPERIENCE_LIMIT,
) -> list[dict[str, object]]:
    """
    過去成績が高いConfigを読み込み、
    Optimizer探索候補として返す。

    同一Configは1件にまとめる。
    """
    normalized_limit = max(
        0,
        int(limit),
    )

    if normalized_limit == 0:
        return []

    store = _load_store()
    games = store.get("games")

    if not isinstance(games, Mapping):
        return []

    game_store = games.get(
        str(game_name)
    )

    if not isinstance(
        game_store,
        Mapping,
    ):
        return []

    history = game_store.get("history")

    if not isinstance(history, list):
        return []

    normalized_history: list[
        dict[str, object]
    ] = []

    for item in history:
        if not isinstance(item, Mapping):
            continue

        normalized = (
            _normalize_history_entry(
                item
            )
        )

        if normalized is not None:
            normalized_history.append(
                normalized
            )

    config_statistics = (
        _build_config_statistics(
            normalized_history
        )
    )

    selected: list[
        dict[str, object]
    ] = []

    for stats in config_statistics:
        config = stats.get("config")

        if not isinstance(
            config,
            Mapping,
        ):
            continue

        experience_index = (
            len(selected) + 1
        )

        selected.append({
            "name": (
                f"experience_"
                f"{game_name}_"
                f"{experience_index}"
            ),
            "w": dict(
                config.get(
                    "w",
                    {},
                )
            ),
            "f": dict(
                config.get(
                    "f",
                    {},
                )
            ),
        })

        if len(selected) >= normalized_limit:
            break

    return selected


def save_optimizer_experience(
    game_name: str,
    config_name: str,
    config: Mapping[str, object],
    evaluation: Mapping[str, object],
    prediction_weights: Mapping[
        str,
        object,
    ],
    learning_strength: float,
    *,
    history_limit: int = (
        DEFAULT_HISTORY_LIMIT
    ),
) -> dict[str, object]:
    """
    今回のOptimizer最終勝者を
    Experience履歴へ保存する。

    同一Configが複数回勝った場合も、
    実行ごとの成績として履歴を保持する。
    """
    normalized_history_limit = max(
        1,
        int(history_limit),
    )

    normalized_config = _normalize_config(
        config
    )

    normalized_learning_strength = (
        _normalize_float(
            learning_strength
        )
    )

    if not (
        0.0
        <= normalized_learning_strength
        <= 1.0
    ):
        raise ValueError(
            "learning_strength must be "
            "between 0.0 and 1.0."
        )

    evaluated_at = datetime.now(
        timezone.utc
    ).isoformat()

    new_entry = {
        "evaluated_at": evaluated_at,
        "config_name": str(
            config_name
        ),
        "selection_score": round(
            _normalize_float(
                evaluation.get(
                    "selection_score"
                )
            ),
            6,
        ),
        "avg_matches": round(
            _normalize_float(
                evaluation.get(
                    "avg_matches"
                )
            ),
            6,
        ),
        "average_matches_per_ticket": (
            round(
                _normalize_float(
                    evaluation.get(
                        "average_matches_per_ticket"
                    )
                ),
                6,
            )
        ),
        "hit_rate_2match": round(
            _normalize_float(
                evaluation.get(
                    "hit_rate_2match"
                )
            ),
            6,
        ),
        "hit_rate_3match": round(
            _normalize_float(
                evaluation.get(
                    "hit_rate_3match"
                )
            ),
            6,
        ),
        "hit_rate_4match": round(
            _normalize_float(
                evaluation.get(
                    "hit_rate_4match"
                )
            ),
            6,
        ),
        "avg_matches_std": round(
            _normalize_float(
                evaluation.get(
                    "avg_matches_std"
                )
            ),
            6,
        ),
        "random_uplift": round(
            _normalize_float(
                evaluation.get(
                    "random_uplift"
                )
            ),
            6,
        ),
        "learning_strength": round(
            normalized_learning_strength,
            6,
        ),
        "config": normalized_config,
        "prediction_weights": (
            _normalize_json_value(
                prediction_weights
            )
        ),
    }

    store = _load_store()
    games = store.get("games")

    if not isinstance(games, dict):
        games = {}

    existing_game = games.get(
        str(game_name)
    )

    if not isinstance(
        existing_game,
        Mapping,
    ):
        existing_game = {}

    existing_history = (
        existing_game.get("history")
    )

    normalized_history: list[
        dict[str, object]
    ] = []

    if isinstance(existing_history, list):
        for item in existing_history:
            if not isinstance(
                item,
                Mapping,
            ):
                continue

            normalized = (
                _normalize_history_entry(
                    item
                )
            )

            if normalized is not None:
                normalized_history.append(
                    normalized
                )

    normalized_history.append(
        new_entry
    )

    ranked_history = sorted(
        normalized_history,
        key=_entry_sort_key,
        reverse=True,
    )

    normalized_history = ranked_history[
        :normalized_history_limit
    ]

    unique_config_count = len({
        _config_signature(
            entry["config"]
        )
        for entry in normalized_history
        if isinstance(
            entry.get("config"),
            Mapping,
        )
    })

    best_entry = (
        ranked_history[0]
        if ranked_history
        else new_entry
    )

    score_sum = sum(
        _normalize_float(
            entry.get(
                "selection_score"
            )
        )
        for entry in normalized_history
    )

    average_selection_score = round(
        score_sum
        / len(normalized_history),
        6,
    )

    config_statistics = (
        _build_config_statistics(
            normalized_history
        )
    )

    public_config_statistics = [
        {
            key: value
            for key, value in stats.items()
            if key not in {
                "signature",
                "config",
            }
        }
        for stats in config_statistics
    ]

    best_experience = (
        config_statistics[0]
        if config_statistics
        else None
    )

    game_output = {
        "updated_at": evaluated_at,
        "history_count": len(
            normalized_history
        ),
        "history_limit": (
            normalized_history_limit
        ),
        "unique_config_count": (
            unique_config_count
        ),
        "average_selection_score": (
            average_selection_score
        ),
        "config_statistics": (
            public_config_statistics
        ),
        "search_source_statistics": (
            _build_search_source_statistics(
                normalized_history
            )
        ),
        "best_experience_config_name": (
            best_experience.get(
                "config_name"
            )
            if best_experience
            is not None
            else None
        ),
        "best_experience_score": (
            best_experience.get(
                "experience_score"
            )
            if best_experience
            is not None
            else None
        ),
        "best_config_name": (
            best_entry.get(
                "config_name"
            )
        ),
        "best_selection_score": (
            best_entry.get(
                "selection_score"
            )
        ),
        "latest": new_entry,
        "history": normalized_history,
    }

    games[str(game_name)] = game_output

    output = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": evaluated_at,
        "games": games,
    }

    save_experience_store(
        EXPERIENCE_PATH,
        output,
    )

    return {
        "history_count": len(
            normalized_history
        ),
        "history_limit": (
            normalized_history_limit
        ),
        "unique_config_count": (
            unique_config_count
        ),
        "best_config_name": (
            best_entry.get(
                "config_name"
            )
        ),
        "best_selection_score": (
            best_entry.get(
                "selection_score"
            )
        ),
        "latest_config_name": str(
            config_name
        ),
        "latest_selection_score": (
            new_entry["selection_score"]
        ),
        "best_experience_config_name": (
            best_experience.get(
                "config_name"
            )
            if best_experience
            is not None
            else None
        ),
        "best_experience_score": (
            best_experience.get(
                "experience_score"
            )
            if best_experience
            is not None
            else None
        ),
    }


__all__ = [
    "load_evolution_adaptation",
    "load_experience_configs",
    "save_optimizer_experience",
]