from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


OUTPUT_DIR = Path("output")
EXPERIENCE_PATH = (
    OUTPUT_DIR / "optimizer_experience.json"
)

SCHEMA_VERSION = "1.2"
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_EXPERIENCE_LIMIT = 3

DEFAULT_EVOLUTION_COUNT = 4
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_MUTATION_SCALE = 0.08
MIN_ADAPTATION_SAMPLES = 5


def _load_store() -> dict[str, object]:
    """
    Optimizer Experienceの保存データを読み込む。

    ファイルが存在しない場合や内容が壊れている場合は、
    空の保存領域を返す。
    """
    if not EXPERIENCE_PATH.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "games": {},
        }

    try:
        loaded = json.loads(
            EXPERIENCE_PATH.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "games": {},
        }

    if not isinstance(loaded, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "games": {},
        }

    games = loaded.get("games")

    if not isinstance(games, dict):
        games = {}

    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": loaded.get(
            "updated_at"
        ),
        "games": games,
    }


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

def _resolve_search_source(
    config_name: object,
) -> str:
    """
    Config名から探索元を判定する。

    既知の接頭辞に一致しない設定は
    baseとして扱う。
    """
    normalized_name = str(
        config_name
        or ""
    ).lower()

    if normalized_name.startswith(
        "evolution_"
    ):
        return "evolution"

    if normalized_name.startswith(
        "local_"
    ):
        return "local"

    if normalized_name.startswith(
        "random_"
    ):
        return "random"

    if normalized_name.startswith(
        "experience_"
    ):
        return "experience"

    return "base"


def _build_search_source_statistics(
    history: Sequence[
        Mapping[str, object]
    ],
) -> dict[str, dict[str, object]]:
    """
    保存履歴を探索元単位で集計する。

    historyには各実行の最終勝者が保存されるため、
    countは探索元ごとの勝者数を表す。
    """
    source_names = (
        "base",
        "experience",
        "random",
        "local",
        "evolution",
    )

    statistics: dict[
        str,
        dict[str, object],
    ] = {
        source: {
            "count": 0,
            "score_sum": 0.0,
            "best_selection_score": 0.0,
            "average_selection_score": 0.0,
            "share": 0.0,
        }
        for source in source_names
    }

    total_count = 0

    for entry in history:
        source = _resolve_search_source(
            entry.get("config_name")
        )
        selection_score = _normalize_float(
            entry.get(
                "selection_score"
            )
        )

        stats = statistics[source]

        stats["count"] = (
            int(stats["count"]) + 1
        )
        stats["score_sum"] = (
            float(stats["score_sum"])
            + selection_score
        )
        stats[
            "best_selection_score"
        ] = max(
            float(
                stats[
                    "best_selection_score"
                ]
            ),
            selection_score,
        )

        total_count += 1

    for stats in statistics.values():
        count = int(stats["count"])

        if count > 0:
            stats[
                "average_selection_score"
            ] = round(
                float(stats["score_sum"])
                / count,
                6,
            )

        if total_count > 0:
            stats["share"] = round(
                count / total_count,
                6,
            )

        stats[
            "best_selection_score"
        ] = round(
            float(
                stats[
                    "best_selection_score"
                ]
            ),
            6,
        )

        del stats["score_sum"]

    return statistics

def _entry_sort_key(
    entry: Mapping[str, object],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Experience履歴の順位付けキーを返す。

    優先順位:
    1. selection_score
    2. random_uplift
    3. avg_matches
    4. avg_matches_stdが小さい
    """
    selection_score = _normalize_float(
        entry.get("selection_score")
    )
    random_uplift = _normalize_float(
        entry.get("random_uplift")
    )
    avg_matches = _normalize_float(
        entry.get("avg_matches")
    )
    avg_matches_std = _normalize_float(
        entry.get("avg_matches_std")
    )

    return (
        selection_score,
        random_uplift,
        avg_matches,
        -avg_matches_std,
    )


def _build_config_statistics(
    history: Sequence[
        Mapping[str, object]
    ],
) -> list[dict[str, object]]:
    """
    Experience履歴をConfig単位で集計する。

    experience_scoreは以下を使用する。

    - 平均selection_score: 50%
    - 最良selection_score: 30%
    - 直近selection_score: 20%
    - 複数回勝利したConfigには最大5%の
      信頼度補正を加える
    """
    statistics_by_signature: dict[
        str,
        dict[str, object],
    ] = {}

    for entry in history:
        config = entry.get("config")

        if not isinstance(
            config,
            Mapping,
        ):
            continue

        signature = _config_signature(
            config
        )
        selection_score = _normalize_float(
            entry.get(
                "selection_score"
            )
        )
        evaluated_at = str(
            entry.get(
                "evaluated_at",
                "",
            )
            or ""
        )

        stats = (
            statistics_by_signature.setdefault(
                signature,
                {
                    "signature": signature,
                    "config_name": str(
                        entry.get(
                            "config_name",
                            "unknown",
                        )
                    ),
                    "wins": 0,
                    "best_selection_score": (
                        selection_score
                    ),
                    "average_selection_score": (
                        0.0
                    ),
                    "latest_selection_score": (
                        selection_score
                    ),
                    "score_sum": 0.0,
                    "last_used": evaluated_at,
                    "config": {
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
                    },
                },
            )
        )

        stats["wins"] = (
            int(stats["wins"]) + 1
        )
        stats["score_sum"] = (
            float(stats["score_sum"])
            + selection_score
        )
        stats[
            "best_selection_score"
        ] = max(
            float(
                stats[
                    "best_selection_score"
                ]
            ),
            selection_score,
        )

        current_last_used = str(
            stats.get(
                "last_used",
                "",
            )
            or ""
        )

        if evaluated_at >= current_last_used:
            stats["last_used"] = (
                evaluated_at
            )
            stats[
                "latest_selection_score"
            ] = selection_score
            stats["config_name"] = str(
                entry.get(
                    "config_name",
                    "unknown",
                )
            )
            stats["config"] = {
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
            }

    results: list[
        dict[str, object]
    ] = []

    for stats in (
        statistics_by_signature.values()
    ):
        wins = int(stats["wins"])

        if wins <= 0:
            continue

        average_score = (
            float(stats["score_sum"])
            / wins
        )
        best_score = float(
            stats[
                "best_selection_score"
            ]
        )
        latest_score = float(
            stats[
                "latest_selection_score"
            ]
        )

        base_experience_score = (
            average_score * 0.50
            + best_score * 0.30
            + latest_score * 0.20
        )

        reliability_multiplier = (
            1.0
            + min(
                max(wins - 1, 0),
                5,
            )
            * 0.01
        )

        stats[
            "average_selection_score"
        ] = round(
            average_score,
            6,
        )
        stats["experience_score"] = round(
            base_experience_score
            * reliability_multiplier,
            6,
        )

        del stats["score_sum"]
        results.append(stats)

    results.sort(
        key=lambda item: (
            _normalize_float(
                item.get(
                    "experience_score"
                )
            ),
            _normalize_float(
                item.get(
                    "best_selection_score"
                )
            ),
            _normalize_float(
                item.get(
                    "average_selection_score"
                )
            ),
            int(
                item.get(
                    "wins",
                    0,
                )
            ),
            str(
                item.get(
                    "last_used",
                    "",
                )
                or ""
            ),
        ),
        reverse=True,
    )

    return results


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

def load_evolution_adaptation(
    game_name: str,
) -> dict[str, object]:
    """
    保存済みの最終勝者履歴から、
    次回Evolutionの探索強度を決定する。

    サンプル不足時:
    - 既定値を使用する

    Evolution勝者率が高い場合:
    - 候補数を増やす
    - 突然変異を弱めて既存の強い領域を深掘りする

    Evolution勝者率が低い場合:
    - 候補数を少し減らす
    - 突然変異を強めて探索範囲を広げる
    """
    default_result = {
        "adaptive": False,
        "reason": "insufficient_history",
        "sample_count": 0,
        "evolution_win_count": 0,
        "evolution_win_rate": 0.0,
        "count": DEFAULT_EVOLUTION_COUNT,
        "mutation_rate": (
            DEFAULT_MUTATION_RATE
        ),
        "mutation_scale": (
            DEFAULT_MUTATION_SCALE
        ),
        "source_statistics": (
            _build_search_source_statistics(
                []
            )
        ),
    }

    store = _load_store()
    games = store.get("games")

    if not isinstance(games, Mapping):
        return default_result

    game_store = games.get(
        str(game_name)
    )

    if not isinstance(
        game_store,
        Mapping,
    ):
        return default_result

    history = game_store.get("history")

    if not isinstance(history, list):
        return default_result

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

    source_statistics = (
        _build_search_source_statistics(
            normalized_history
        )
    )

    sample_count = len(
        normalized_history
    )

    evolution_stats = (
        source_statistics["evolution"]
    )
    evolution_win_count = int(
        evolution_stats["count"]
    )

    evolution_win_rate = (
        evolution_win_count
        / sample_count
        if sample_count > 0
        else 0.0
    )

    if sample_count < MIN_ADAPTATION_SAMPLES:
        return {
            **default_result,
            "sample_count": sample_count,
            "evolution_win_count": (
                evolution_win_count
            ),
            "evolution_win_rate": round(
                evolution_win_rate,
                6,
            ),
            "source_statistics": (
                source_statistics
            ),
        }

    if evolution_win_rate >= 0.40:
        count = 6
        mutation_rate = 0.18
        mutation_scale = 0.06
        reason = "evolution_high_performance"
    elif evolution_win_rate >= 0.20:
        count = DEFAULT_EVOLUTION_COUNT
        mutation_rate = (
            DEFAULT_MUTATION_RATE
        )
        mutation_scale = (
            DEFAULT_MUTATION_SCALE
        )
        reason = "evolution_normal_performance"
    else:
        count = 3
        mutation_rate = 0.35
        mutation_scale = 0.10
        reason = "evolution_low_performance"

    return {
        "adaptive": True,
        "reason": reason,
        "sample_count": sample_count,
        "evolution_win_count": (
            evolution_win_count
        ),
        "evolution_win_rate": round(
            evolution_win_rate,
            6,
        ),
        "count": count,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
        "source_statistics": (
            source_statistics
        ),
    }

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        EXPERIENCE_PATH.with_suffix(
            ".json.tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        EXPERIENCE_PATH
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