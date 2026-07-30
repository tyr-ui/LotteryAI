from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from feature_memory import MEMORY_PATH


ANALYSIS_PATH = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "feature_memory_analysis.json"
)


def _load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {
            "schema_version": "1.0",
            "history": {},
        }

    try:
        with open(
            MEMORY_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            loaded = json.load(f)
    except Exception:
        return {
            "schema_version": "1.0",
            "history": {},
        }

    if not isinstance(loaded, dict):
        return {
            "schema_version": "1.0",
            "history": {},
        }

    return loaded


def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _collect_feature_names(
    records: Sequence[Mapping],
) -> list[str]:
    names: set[str] = set()

    for record in records:
        features = record.get(
            "features",
            [],
        )

        if not isinstance(features, list):
            continue

        for feature_result in features:
            if not isinstance(
                feature_result,
                Mapping,
            ):
                continue

            feature_name = feature_result.get(
                "feature"
            )

            if feature_name is not None:
                names.add(str(feature_name))

    return sorted(names)


def _extract_feature_result(
    record: Mapping,
    feature_name: str,
) -> Mapping | None:
    features = record.get(
        "features",
        [],
    )

    if not isinstance(features, list):
        return None

    for feature_result in features:
        if not isinstance(
            feature_result,
            Mapping,
        ):
            continue

        if str(
            feature_result.get("feature")
        ) == feature_name:
            return feature_result

    return None


def _summarize_feature(
    records: Sequence[Mapping],
    feature_name: str,
) -> dict[str, object]:
    score_drops: list[float] = []
    score_drop_percents: list[float] = []
    match_drops: list[float] = []
    match_drop_percents: list[float] = []
    original_weights: list[float] = []

    active_count = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    latest_weight = 0.0
    latest_active = False

    for record in records:
        feature_result = (
            _extract_feature_result(
                record,
                feature_name,
            )
        )

        if feature_result is None:
            continue

        score_drop = _safe_float(
            feature_result.get(
                "selection_score_drop"
            )
        )
        score_drop_percent = _safe_float(
            feature_result.get(
                "selection_score_drop_percent"
            )
        )
        match_drop = _safe_float(
            feature_result.get(
                "avg_matches_drop"
            )
        )
        match_drop_percent = _safe_float(
            feature_result.get(
                "avg_matches_drop_percent"
            )
        )
        original_weight = _safe_float(
            feature_result.get(
                "original_weight"
            )
        )
        active = bool(
            feature_result.get(
                "active",
                False,
            )
        )

        score_drops.append(score_drop)
        score_drop_percents.append(
            score_drop_percent
        )
        match_drops.append(match_drop)
        match_drop_percents.append(
            match_drop_percent
        )
        original_weights.append(
            original_weight
        )

        latest_weight = original_weight
        latest_active = active

        if active:
            active_count += 1

        if score_drop > 0:
            positive_count += 1
        elif score_drop < 0:
            negative_count += 1
        else:
            neutral_count += 1

    sample_count = len(score_drops)

    if sample_count == 0:
        return {
            "feature": feature_name,
            "sample_count": 0,
            "latest_weight": 0.0,
            "latest_active": False,
            "average_original_weight": 0.0,
            "average_selection_score_drop": 0.0,
            (
                "average_selection_score_"
                "drop_percent"
            ): 0.0,
            "average_matches_drop": 0.0,
            (
                "average_matches_drop_"
                "percent"
            ): 0.0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "neutral_rate": 0.0,
            "active_rate": 0.0,
        }

    return {
        "feature": feature_name,
        "sample_count": sample_count,
        "latest_weight": round(
            latest_weight,
            8,
        ),
        "latest_active": latest_active,
        "average_original_weight": round(
            _average(original_weights),
            8,
        ),
        "average_selection_score_drop": (
            round(
                _average(score_drops),
                6,
            )
        ),
        (
            "average_selection_score_"
            "drop_percent"
        ): round(
            _average(score_drop_percents),
            6,
        ),
        "average_matches_drop": round(
            _average(match_drops),
            6,
        ),
        (
            "average_matches_drop_"
            "percent"
        ): round(
            _average(match_drop_percents),
            6,
        ),
        "positive_rate": round(
            positive_count
            / sample_count,
            6,
        ),
        "negative_rate": round(
            negative_count
            / sample_count,
            6,
        ),
        "neutral_rate": round(
            neutral_count
            / sample_count,
            6,
        ),
        "active_rate": round(
            active_count
            / sample_count,
            6,
        ),
    }


def _summarize_records(
    records: Sequence[Mapping],
) -> dict[str, object]:
    feature_names = _collect_feature_names(
        records
    )

    features = [
        _summarize_feature(
            records,
            feature_name,
        )
        for feature_name in feature_names
    ]

    features.sort(
        key=lambda item: float(
            item[
                (
                    "average_selection_score_"
                    "drop_percent"
                )
            ]
        ),
        reverse=True,
    )

    ranking = [
        str(item["feature"])
        for item in features
    ]

    return {
        "run_count": len(records),
        "ranking": ranking,
        "features": features,
    }


def _calculate_trend(
    recent_summary: Mapping,
    previous_summary: Mapping,
) -> list[dict[str, object]]:
    recent_features = {
        str(item["feature"]): item
        for item in recent_summary.get(
            "features",
            [],
        )
        if isinstance(item, Mapping)
    }

    previous_features = {
        str(item["feature"]): item
        for item in previous_summary.get(
            "features",
            [],
        )
        if isinstance(item, Mapping)
    }

    feature_names = sorted(
        set(recent_features)
        | set(previous_features)
    )

    trends: list[dict[str, object]] = []

    for feature_name in feature_names:
        recent_item = recent_features.get(
            feature_name,
            {},
        )
        previous_item = (
            previous_features.get(
                feature_name,
                {},
            )
        )

        recent_value = _safe_float(
            recent_item.get(
                (
                    "average_selection_score_"
                    "drop_percent"
                )
            )
        )
        previous_value = _safe_float(
            previous_item.get(
                (
                    "average_selection_score_"
                    "drop_percent"
                )
            )
        )

        difference = (
            recent_value - previous_value
        )

        if difference > 1.0:
            direction = "up"
        elif difference < -1.0:
            direction = "down"
        else:
            direction = "stable"

        trends.append({
            "feature": feature_name,
            "recent_value": round(
                recent_value,
                6,
            ),
            "previous_value": round(
                previous_value,
                6,
            ),
            "difference": round(
                difference,
                6,
            ),
            "direction": direction,
        })

    trends.sort(
        key=lambda item: abs(
            float(item["difference"])
        ),
        reverse=True,
    )

    return trends


def analyze_game_history(
    records: Sequence[Mapping],
    *,
    short_window: int = 10,
    long_window: int = 30,
) -> dict[str, object]:
    valid_records = [
        record
        for record in records
        if isinstance(record, Mapping)
    ]

    all_summary = _summarize_records(
        valid_records
    )

    short_records = valid_records[
        -short_window:
    ]
    long_records = valid_records[
        -long_window:
    ]

    short_summary = _summarize_records(
        short_records
    )
    long_summary = _summarize_records(
        long_records
    )

    if len(valid_records) >= (
        short_window * 2
    ):
        previous_short_records = (
            valid_records[
                -(short_window * 2):
                -short_window
            ]
        )

        previous_short_summary = (
            _summarize_records(
                previous_short_records
            )
        )

        trend = _calculate_trend(
            short_summary,
            previous_short_summary,
        )
        trend_status = "available"
    else:
        previous_short_summary = {
            "run_count": 0,
            "ranking": [],
            "features": [],
        }
        trend = []
        trend_status = (
            "insufficient_history"
        )

    return {
        "total_run_count": len(
            valid_records
        ),
        "short_window": short_window,
        "long_window": long_window,
        "all_history": all_summary,
        "recent_short": short_summary,
        "recent_long": long_summary,
        "previous_short": (
            previous_short_summary
        ),
        "trend_status": trend_status,
        "trend": trend,
    }


def analyze_feature_memory(
    *,
    short_window: int = 10,
    long_window: int = 30,
) -> dict[str, object]:
    memory = _load_memory()

    history = memory.get(
        "history",
        {},
    )

    if not isinstance(history, Mapping):
        history = {}

    games: dict[str, object] = {}

    for game_name, records in history.items():
        if not isinstance(records, list):
            continue

        games[str(game_name)] = (
            analyze_game_history(
                records,
                short_window=short_window,
                long_window=long_window,
            )
        )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": str(MEMORY_PATH),
        "short_window": short_window,
        "long_window": long_window,
        "games": games,
    }


def save_feature_memory_analysis(
    *,
    short_window: int = 10,
    long_window: int = 30,
) -> dict[str, object]:
    analysis = analyze_feature_memory(
        short_window=short_window,
        long_window=long_window,
    )

    ANALYSIS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        ANALYSIS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            analysis,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return analysis


__all__ = [
    "ANALYSIS_PATH",
    "analyze_feature_memory",
    "analyze_game_history",
    "save_feature_memory_analysis",
]