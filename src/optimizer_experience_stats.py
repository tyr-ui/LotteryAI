"""
Statistics helpers for Optimizer Experience.

This module contains pure calculations for:
- search-source classification
- search-source statistics
- config statistics
- Experience history ordering

Filesystem I/O and adaptation decisions remain in their dedicated modules.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence


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


def _normalize_json_value(
    value: object,
) -> object:
    """Convert nested Config values into JSON-safe values."""
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
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence):
        if isinstance(value, (str, bytes, bytearray)):
            return str(value)
        return [_normalize_json_value(item) for item in value]

    return str(value)


def _normalize_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Normalize an Optimizer Config for stable statistics and signatures."""
    weights = config.get("w")
    filters = config.get("f", {})

    if not isinstance(weights, Mapping):
        raise ValueError(
            "Optimizer config must contain a mapping named 'w'."
        )

    if not isinstance(filters, Mapping):
        filters = {}

    normalized_weights = {
        str(key): _normalize_json_value(value)
        for key, value in weights.items()
    }
    if not normalized_weights:
        raise ValueError(
            "Optimizer config weights must not be empty."
        )

    normalized_filters = {
        str(key): _normalize_json_value(value)
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

    recent_countとrecent_shareは、
    evaluated_atを基準とした直近10回から算出する。
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
            "latest_selection_score": 0.0,
            "latest_evaluated_at": "",
            "share": 0.0,
            "recent_count": 0,
            "recent_share": 0.0,
            "rank": 0,
            "is_best_source": False,
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
        evaluated_at = str(
            entry.get(
                "evaluated_at",
                "",
            )
            or ""
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

        latest_evaluated_at = str(
            stats.get(
                "latest_evaluated_at",
                "",
            )
            or ""
        )

        if (
            evaluated_at
            >= latest_evaluated_at
        ):
            stats[
                "latest_evaluated_at"
            ] = evaluated_at
            stats[
                "latest_selection_score"
            ] = selection_score

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
        stats[
            "latest_selection_score"
        ] = round(
            float(
                stats[
                    "latest_selection_score"
                ]
            ),
            6,
        )

        del stats["score_sum"]
        del stats["latest_evaluated_at"]

    chronological_history = sorted(
        history,
        key=lambda entry: str(
            entry.get(
                "evaluated_at",
                "",
            )
            or ""
        ),
    )

    recent_history = (
        chronological_history[-10:]
    )
    recent_total = len(
        recent_history
    )

    for entry in recent_history:
        source = _resolve_search_source(
            entry.get(
                "config_name"
            )
        )

        source_stats = statistics[
            source
        ]

        source_stats[
            "recent_count"
        ] = (
            int(
                source_stats[
                    "recent_count"
                ]
            )
            + 1
        )

    if recent_total > 0:
        for stats in (
            statistics.values()
        ):
            stats[
                "recent_share"
            ] = round(
                int(
                    stats[
                        "recent_count"
                    ]
                )
                / recent_total,
                6,
            )

    active_sources = [
        item
        for item in statistics.items()
        if int(
            item[1]["count"]
        ) > 0
    ]

    ranked_sources = sorted(
        active_sources,
        key=lambda item: (
            int(
                item[1]["count"]
            ),
            float(
                item[1][
                    "average_selection_score"
                ]
            ),
            float(
                item[1][
                    "best_selection_score"
                ]
            ),
            float(
                item[1][
                    "recent_share"
                ]
            ),
        ),
        reverse=True,
    )

    for rank, (
        _,
        stats,
    ) in enumerate(
        ranked_sources,
        start=1,
    ):
        stats["rank"] = rank

    if ranked_sources:
        strongest_source = (
            ranked_sources[0][0]
        )

        statistics[
            strongest_source
        ][
            "is_best_source"
        ] = True

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


__all__ = [
    "_resolve_search_source",
    "_build_search_source_statistics",
    "_entry_sort_key",
    "_build_config_statistics",
]
