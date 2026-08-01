from __future__ import annotations

from dataclasses import asdict
from random import Random
from typing import Iterable, Mapping, Sequence

from numbers_backtester import (
    NumbersBacktestSummary,
    run_numbers_backtest,
)
from numbers_features import (
    NumberRow,
    build_numbers_model_context,
)
from numbers_predictor import (
    NumbersPredictionWeights,
    predict_numbers,
)


WEIGHT_FIELDS = tuple(
    field_name
    for field_name in NumbersPredictionWeights.__dataclass_fields__
    if field_name != "diversity"
)


BASE_WEIGHT_CONFIGS: tuple[
    tuple[str, NumbersPredictionWeights],
    ...,
] = (
    (
        "default",
        NumbersPredictionWeights(),
    ),
    (
        "position_recent",
        NumbersPredictionWeights(
            position_frequency=1.45,
            recent_position_frequency=1.85,
            position_delay=0.45,
            overall_frequency=0.25,
            ordered_pair=0.70,
            ordered_triplet=0.20,
            duplicate_pattern=0.45,
            sum_shape=0.35,
            odd_shape=0.25,
            high_shape=0.25,
            adjacent_difference=0.25,
            exact_repeat=0.40,
            unordered_repeat=0.25,
            diversity=0.30,
        ),
    ),
    (
        "position_delay",
        NumbersPredictionWeights(
            position_frequency=1.40,
            recent_position_frequency=1.20,
            position_delay=1.25,
            overall_frequency=0.25,
            ordered_pair=0.65,
            ordered_triplet=0.20,
            duplicate_pattern=0.45,
            sum_shape=0.35,
            odd_shape=0.25,
            high_shape=0.25,
            adjacent_difference=0.25,
            exact_repeat=0.40,
            unordered_repeat=0.25,
            diversity=0.30,
        ),
    ),
    (
        "ordered_patterns",
        NumbersPredictionWeights(
            position_frequency=1.10,
            recent_position_frequency=1.25,
            position_delay=0.45,
            overall_frequency=0.25,
            ordered_pair=1.60,
            ordered_triplet=0.85,
            duplicate_pattern=0.45,
            sum_shape=0.30,
            odd_shape=0.20,
            high_shape=0.20,
            adjacent_difference=0.30,
            exact_repeat=0.35,
            unordered_repeat=0.20,
            diversity=0.30,
        ),
    ),
    (
        "shape_balanced",
        NumbersPredictionWeights(
            position_frequency=1.00,
            recent_position_frequency=1.10,
            position_delay=0.45,
            overall_frequency=0.25,
            ordered_pair=0.55,
            ordered_triplet=0.15,
            duplicate_pattern=0.95,
            sum_shape=0.90,
            odd_shape=0.65,
            high_shape=0.65,
            adjacent_difference=0.75,
            exact_repeat=0.35,
            unordered_repeat=0.25,
            diversity=0.35,
        ),
    ),
    (
        "repeat_balanced",
        NumbersPredictionWeights(
            position_frequency=1.05,
            recent_position_frequency=1.20,
            position_delay=0.55,
            overall_frequency=0.30,
            ordered_pair=0.60,
            ordered_triplet=0.20,
            duplicate_pattern=0.50,
            sum_shape=0.40,
            odd_shape=0.30,
            high_shape=0.30,
            adjacent_difference=0.35,
            exact_repeat=1.15,
            unordered_repeat=0.85,
            diversity=0.35,
        ),
    ),
)


def _config_value(
    config: Mapping[str, object] | object,
    name: str,
    default: object,
) -> object:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _weights_to_dict(
    weights: NumbersPredictionWeights,
) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in asdict(weights).items()
    }


def _normalize_history(
    history: Iterable[Sequence[int]],
) -> tuple[NumberRow, ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in history
    )


def _build_box_prediction(
    ranked_candidates: Sequence,
    *,
    top_k: int,
    model_name: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_box_keys: set[tuple[int, ...]] = set()

    for item in ranked_candidates:
        box_digits = tuple(
            sorted(int(value) for value in item.candidate)
        )

        if box_digits in seen_box_keys:
            continue

        seen_box_keys.add(box_digits)
        selected.append({
            "pattern_id": f"B{len(selected) + 1}",
            "numbers": list(box_digits),
            "digits": list(box_digits),
            "number": "".join(
                str(value) for value in box_digits
            ),
            "representative_straight": item.number,
            "score": round(float(item.total_score), 6),
            "model": model_name,
            "components": dict(item.components),
            "exact_repeat_count": item.exact_repeat_count,
            "unordered_repeat_count": (
                item.unordered_repeat_count
            ),
        })

        if len(selected) >= top_k:
            break

    return selected


def _summary_sort_key(
    summary: NumbersBacktestSummary,
) -> tuple[float, float, float, float, float]:
    return (
        float(summary.selection_score or 0.0),
        float(summary.straight_hit_rate or 0.0),
        float(summary.box_hit_rate or 0.0),
        float(summary.average_best_position_matches or 0.0),
        float(summary.average_best_unordered_matches or 0.0),
    )


def _clamp_weight(value: float) -> float:
    return round(
        min(3.0, max(0.05, float(value))),
        6,
    )


def _mutate_weights(
    base: NumbersPredictionWeights,
    *,
    rng: Random,
    mutation_scale: float,
) -> NumbersPredictionWeights:
    values = _weights_to_dict(base)

    for field_name in WEIGHT_FIELDS:
        multiplier = rng.uniform(
            1.0 - mutation_scale,
            1.0 + mutation_scale,
        )
        values[field_name] = _clamp_weight(
            values[field_name] * multiplier
        )

    values["diversity"] = _clamp_weight(
        values["diversity"]
        * rng.uniform(0.85, 1.15)
    )

    return NumbersPredictionWeights(**values)


def _evaluation_periods(
    *,
    digit_count: int,
    configured_periods: int,
    history_size: int,
    train_window: int,
) -> int:
    default_periods = 24 if digit_count == 3 else 12
    requested = (
        configured_periods
        if configured_periods > 0
        else default_periods
    )
    available = max(0, history_size - train_window)

    if available <= 0:
        return 1

    return max(
        1,
        min(requested, default_periods, available),
    )


def evaluate_numbers_weights(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    weights: NumbersPredictionWeights,
    tested_periods: int,
    top_k: int,
) -> NumbersBacktestSummary:
    return run_numbers_backtest(
        history,
        config,
        train_window=int(
            _config_value(config, "train_window", 500)
        ),
        tested_periods=tested_periods,
        top_k=top_k,
        weights=weights,
    )


def optimize_numbers(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    seed: int = 2025,
) -> dict[str, object]:
    normalized_history = _normalize_history(history)

    if not normalized_history:
        raise ValueError(
            "history must contain at least one draw."
        )

    digit_count = len(normalized_history[0])
    train_window = int(
        _config_value(config, "train_window", 500)
    )
    full_tested_periods = int(
        _config_value(config, "tested_periods", 90)
    )
    top_k = int(
        _config_value(config, "top_k", 10)
    )
    configured_search_periods = int(
        _config_value(
            config,
            "numbers_optimizer_periods",
            0,
        )
        or 0
    )
    local_config_count = int(
        _config_value(
            config,
            "numbers_optimizer_local_configs",
            2,
        )
        or 0
    )
    mutation_scale = float(
        _config_value(
            config,
            "numbers_optimizer_mutation_scale",
            0.20,
        )
        or 0.20
    )

    search_periods = _evaluation_periods(
        digit_count=digit_count,
        configured_periods=configured_search_periods,
        history_size=len(normalized_history),
        train_window=train_window,
    )

    ranked: list[dict[str, object]] = []

    for config_name, weights in BASE_WEIGHT_CONFIGS:
        summary = evaluate_numbers_weights(
            normalized_history,
            config,
            weights=weights,
            tested_periods=search_periods,
            top_k=top_k,
        )
        ranked.append({
            "config": config_name,
            "source": "base",
            "weights": _weights_to_dict(weights),
            **summary.to_dict(),
            "_weights_object": weights,
            "_summary_object": summary,
        })

    ranked.sort(
        key=lambda item: _summary_sort_key(
            item["_summary_object"]
        ),
        reverse=True,
    )

    rng = Random(seed)
    parent_weights = ranked[0]["_weights_object"]

    for index in range(max(0, local_config_count)):
        weights = _mutate_weights(
            parent_weights,
            rng=rng,
            mutation_scale=mutation_scale,
        )
        summary = evaluate_numbers_weights(
            normalized_history,
            config,
            weights=weights,
            tested_periods=search_periods,
            top_k=top_k,
        )
        ranked.append({
            "config": f"local_{index + 1}",
            "source": "local",
            "weights": _weights_to_dict(weights),
            **summary.to_dict(),
            "_weights_object": weights,
            "_summary_object": summary,
        })

    ranked.sort(
        key=lambda item: _summary_sort_key(
            item["_summary_object"]
        ),
        reverse=True,
    )

    selected = ranked[0]
    selected_weights = selected["_weights_object"]

    full_backtest = run_numbers_backtest(
        normalized_history,
        config,
        train_window=train_window,
        tested_periods=full_tested_periods,
        top_k=top_k,
        weights=selected_weights,
    )

    context = build_numbers_model_context(
        normalized_history,
        config,
    )
    prediction_result = predict_numbers(
        context,
        top_k=top_k,
        weights=selected_weights,
    )

    prediction = [
        {
            "pattern_id": f"P{index}",
            "numbers": list(item.candidate),
            "digits": list(item.candidate),
            "number": item.number,
            "score": round(float(item.total_score), 6),
            "model": selected["config"],
            "components": dict(item.components),
            "exact_repeat_count": item.exact_repeat_count,
            "unordered_repeat_count": item.unordered_repeat_count,
        }
        for index, item in enumerate(
            prediction_result.selected,
            start=1,
        )
    ]

    box_prediction = _build_box_prediction(
        prediction_result.ranked,
        top_k=top_k,
        model_name=str(selected["config"]),
    )

    public_ranked = [
        {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }
        for item in ranked
    ]

    default_result = next(
        (
            item
            for item in public_ranked
            if item["config"] == "default"
        ),
        None,
    )

    default_weights = _weights_to_dict(
        NumbersPredictionWeights()
    )
    applied_weights = _weights_to_dict(
        selected_weights
    )

    return {
        "random_baseline": default_result or {},
        "selected_random_filtered_baseline": {},
        "ranked_configs": public_ranked,
        "selected_config": selected["config"],
        "selected_weights": applied_weights,
        "selected_filters": {},
        "learning_summary": {
            "strength": 1.0,
            "strength_optimized": True,
            "tested_strengths": [],
            "loaded_weights": {},
            "base_prediction_weights": default_weights,
            "applied_prediction_weights": applied_weights,
            "weight_diff": {
                key: round(
                    applied_weights[key]
                    - default_weights[key],
                    6,
                )
                for key in applied_weights
            },
        },
        "search_metadata": {
            "algorithm": (
                "numbers_base_plus_local_search"
            ),
            "optimizer_connected": True,
            "candidate_space_size": (
                prediction_result.generated_count
            ),
            "box_prediction_enabled": True,
            "box_prediction_version": 1,
            "box_prediction_count": len(
                box_prediction
            ),
            "base_config_count": len(BASE_WEIGHT_CONFIGS),
            "local_config_count": max(
                0,
                local_config_count,
            ),
            "evaluated_config_count": len(public_ranked),
            "search_tested_periods": search_periods,
            "full_tested_periods": (
                full_backtest.tested_periods
            ),
            "seed": seed,
        },
        "feature_ablation": [],
        "optimizer_experience": {},
        "numbers_backtest": full_backtest.to_dict(),
        "prediction": prediction,
        "box_prediction": box_prediction,
    }


__all__ = [
    "BASE_WEIGHT_CONFIGS",
    "evaluate_numbers_weights",
    "optimize_numbers",
]
