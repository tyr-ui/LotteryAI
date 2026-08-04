from __future__ import annotations

from dataclasses import asdict
from random import Random
from typing import Iterable, Mapping, Sequence

from numbers_backtester import (
    NumbersBacktestSummary,
    run_numbers_backtest,
    run_numbers_box_random_backtest,
    run_numbers_uniform_random_backtest,
)
from numbers_features import (
    NumberRow,
    build_numbers_model_context,
)
from numbers_predictor import (
    NumbersPredictionWeights,
    predict_numbers,
)
from optimizer_experience import (
    load_evolution_adaptation,
    load_experience_configs,
    load_search_allocation,
)


WEIGHT_FIELDS = tuple(
    field_name
    for field_name in NumbersPredictionWeights.__dataclass_fields__
    if field_name != "diversity"
)




DEFAULT_NUMBERS_HOLDOUT_PERIODS = 60
MIN_NUMBERS_HOLDOUT_PERIODS = 10


def _resolve_holdout_periods(
    *,
    history_size: int,
    train_window: int,
    requested_periods: int,
) -> int:
    """Return a safe independent holdout size for Numbers evaluation.

    The selection history must still contain at least one evaluable draw after
    the configured training window. Very small residual holdouts are disabled
    because they are too unstable to present as an independent evaluation.
    """
    available = max(0, int(history_size) - int(train_window) - 1)
    resolved = min(max(0, int(requested_periods)), available)
    if resolved < MIN_NUMBERS_HOLDOUT_PERIODS:
        return 0
    return resolved


def _numbers_holdout_result(
    model_summary: NumbersBacktestSummary,
    random_baseline: Mapping[str, object],
    *,
    holdout_periods: int,
    selection_history_draws: int,
) -> dict[str, object]:
    model = model_summary.to_dict(include_records=True)
    model_score = float(model.get("selection_score") or 0.0)
    random_score = float(random_baseline.get("selection_score") or 0.0)
    model_position = float(
        model.get("average_best_position_matches") or 0.0
    )
    random_position = float(
        random_baseline.get("average_best_position_matches") or 0.0
    )
    return {
        "evaluation_type": "independent_rolling_holdout",
        "holdout_periods": int(holdout_periods),
        "tested_periods": int(model.get("tested_periods") or 0),
        "selection_history_draws": int(selection_history_draws),
        **model,
        "random_baseline": dict(random_baseline),
        "selection_score_uplift": round(model_score - random_score, 6),
        "random_uplift": round(model_position - random_position, 6),
    }


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



def _weights_from_mapping(
    values: Mapping[str, object],
) -> NumbersPredictionWeights | None:
    """Experienceの重み辞書をNumbers用重みへ安全に復元する。"""
    defaults = _weights_to_dict(NumbersPredictionWeights())
    normalized: dict[str, float] = {}

    for field_name, default_value in defaults.items():
        raw_value = values.get(field_name, default_value)
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return None
        normalized[field_name] = _clamp_weight(numeric)

    return NumbersPredictionWeights(**normalized)


def _weights_signature(
    weights: NumbersPredictionWeights,
) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(_weights_to_dict(weights).items()))


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


def _random_weights(rng: Random) -> NumbersPredictionWeights:
    values = _weights_to_dict(NumbersPredictionWeights())
    for field_name in WEIGHT_FIELDS:
        values[field_name] = _clamp_weight(rng.uniform(0.15, 2.25))
    values["diversity"] = _clamp_weight(rng.uniform(0.15, 0.80))
    return NumbersPredictionWeights(**values)


def _crossover_weights(
    first: NumbersPredictionWeights,
    second: NumbersPredictionWeights,
    *,
    rng: Random,
    mutation_rate: float,
    mutation_scale: float,
) -> NumbersPredictionWeights:
    first_values = _weights_to_dict(first)
    second_values = _weights_to_dict(second)
    values: dict[str, float] = {}

    for field_name in first_values:
        value = (
            first_values[field_name]
            if rng.random() < 0.5
            else second_values[field_name]
        )
        if rng.random() < mutation_rate:
            value *= rng.uniform(
                1.0 - mutation_scale,
                1.0 + mutation_scale,
            )
        values[field_name] = _clamp_weight(value)

    return NumbersPredictionWeights(**values)


def _aggregate_uniform_random_summaries(
    summaries: Sequence[NumbersBacktestSummary],
) -> dict[str, object]:
    if not summaries:
        raise ValueError("summaries must not be empty.")

    metric_names = (
        "average_best_position_matches",
        "average_position_matches_per_ticket",
        "average_best_unordered_matches",
        "average_unordered_matches_per_ticket",
        "straight_hit_rate",
        "box_hit_rate",
        "hit_rate_1_position",
        "hit_rate_2_position",
        "hit_rate_3_position",
        "hit_rate_4_position",
        "selection_score",
    )
    result: dict[str, object] = {
        "config": "uniform_random",
        "source": "uniform_random",
        "tested_periods": summaries[0].tested_periods,
        "digit_count": summaries[0].digit_count,
        "evaluated_seeds": len(summaries),
    }
    for name in metric_names:
        values = [
            float(value)
            for summary in summaries
            if (value := getattr(summary, name)) is not None
        ]
        result[name] = (
            round(sum(values) / len(values), 6)
            if values else None
        )
    return result


def _aggregate_box_random_summaries(summaries: Sequence[object]) -> dict[str, object]:
    if not summaries:
        raise ValueError("summaries must not be empty.")
    metrics = (
        "average_best_unordered_matches",
        "average_unordered_matches_per_ticket",
        "box_hit_rate",
    )
    result: dict[str, object] = {
        "evaluation_type": "box_unique_random_baseline",
        "tested_periods": int(getattr(summaries[0], "tested_periods", 0) or 0),
        "evaluated_seeds": len(summaries),
    }
    for key in metrics:
        values = [float(getattr(item, key) or 0.0) for item in summaries]
        result[key] = round(sum(values) / len(values), 6)
    result["seed_records"] = {
        str(index): item.to_dict(include_records=True).get("records", [])
        for index, item in enumerate(summaries)
    }
    return result


def optimize_numbers(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    seed: int = 2025,
    draw_numbers: Sequence[int] | None = None,
) -> dict[str, object]:
    normalized_history = _normalize_history(history)

    if not normalized_history:
        raise ValueError("history must contain at least one draw.")

    digit_count = len(normalized_history[0])
    train_window = int(_config_value(config, "train_window", 500))
    full_tested_periods = int(_config_value(config, "tested_periods", 90))
    top_k = int(_config_value(config, "top_k", 10))
    configured_search_periods = int(
        _config_value(config, "numbers_optimizer_periods", 0) or 0
    )
    mutation_scale = float(
        _config_value(config, "numbers_optimizer_mutation_scale", 0.20)
        or 0.20
    )
    game_key = str(
        _config_value(config, "key", f"numbers{digit_count}")
    )
    requested_holdout_periods = int(
        _config_value(
            config,
            "numbers_holdout_periods",
            DEFAULT_NUMBERS_HOLDOUT_PERIODS,
        )
        or DEFAULT_NUMBERS_HOLDOUT_PERIODS
    )
    holdout_periods = _resolve_holdout_periods(
        history_size=len(normalized_history),
        train_window=train_window,
        requested_periods=requested_holdout_periods,
    )
    selection_history = (
        normalized_history[:-holdout_periods]
        if holdout_periods > 0
        else normalized_history
    )
    trained_through_draw_no: int | None = None
    if draw_numbers is not None:
        normalized_draw_numbers = sorted(int(value) for value in draw_numbers)
        if len(normalized_draw_numbers) != len(normalized_history):
            raise ValueError(
                "draw_numbers length must match normalized history length."
            )
        if selection_history:
            trained_through_draw_no = normalized_draw_numbers[
                len(selection_history) - 1
            ]

    search_periods = _evaluation_periods(
        digit_count=digit_count,
        configured_periods=configured_search_periods,
        history_size=len(selection_history),
        train_window=train_window,
    )

    allocation = load_search_allocation(
        game_key,
        max_trained_through_draw_no=trained_through_draw_no,
    )
    allocation_counts = allocation.get("counts", {})
    if not isinstance(allocation_counts, Mapping):
        allocation_counts = {}

    requested_experience = max(0, int(allocation_counts.get("experience", 3)))
    requested_random = max(0, int(allocation_counts.get("random", 4)))
    requested_local = max(0, int(allocation_counts.get("local", 6)))
    requested_evolution = max(0, int(allocation_counts.get("evolution", 4)))

    evolution_adaptation = load_evolution_adaptation(
        game_key,
        max_trained_through_draw_no=trained_through_draw_no,
    )
    evolution_mutation_rate = float(
        evolution_adaptation.get("mutation_rate", 0.25) or 0.25
    )
    evolution_mutation_scale = float(
        evolution_adaptation.get("mutation_scale", 0.08) or 0.08
    )

    ranked: list[dict[str, object]] = []
    seen_weight_signatures: set[tuple[tuple[str, float], ...]] = set()

    def evaluate_candidate(
        config_name: str,
        source: str,
        weights: NumbersPredictionWeights,
    ) -> bool:
        signature = _weights_signature(weights)
        if signature in seen_weight_signatures:
            return False
        seen_weight_signatures.add(signature)
        summary = evaluate_numbers_weights(
            selection_history,
            config,
            weights=weights,
            tested_periods=search_periods,
            top_k=top_k,
        )
        ranked.append({
            "config": config_name,
            "source": source,
            "weights": _weights_to_dict(weights),
            **summary.to_dict(),
            "_weights_object": weights,
            "_summary_object": summary,
        })
        return True

    for config_name, weights in BASE_WEIGHT_CONFIGS:
        evaluate_candidate(config_name, "base", weights)

    loaded_experience = load_experience_configs(
        game_key,
        limit=requested_experience,
        max_trained_through_draw_no=trained_through_draw_no,
    )
    restored_experience_count = 0
    for item in loaded_experience:
        raw_weights = item.get("w", {})
        if not isinstance(raw_weights, Mapping):
            continue
        weights = _weights_from_mapping(raw_weights)
        if weights is None:
            continue
        if evaluate_candidate(
            str(item.get("name", "experience")),
            "experience",
            weights,
        ):
            restored_experience_count += 1

    rng = Random(seed)
    experience_shortfall = max(
        0,
        requested_experience - restored_experience_count,
    )
    effective_random_target = requested_random + experience_shortfall
    generated_random_count = 0
    random_attempts = 0
    while generated_random_count < effective_random_target and random_attempts < effective_random_target * 20 + 20:
        random_attempts += 1
        if evaluate_candidate(
            f"random_{generated_random_count + 1:02d}",
            "random",
            _random_weights(rng),
        ):
            generated_random_count += 1

    ranked.sort(
        key=lambda item: _summary_sort_key(item["_summary_object"]),
        reverse=True,
    )

    parent_pool = [item["_weights_object"] for item in ranked[: max(2, min(5, len(ranked)))]]
    generated_local_count = 0
    local_attempts = 0
    while generated_local_count < requested_local and local_attempts < requested_local * 20 + 20:
        local_attempts += 1
        parent = parent_pool[generated_local_count % len(parent_pool)]
        weights = _mutate_weights(
            parent,
            rng=rng,
            mutation_scale=mutation_scale,
        )
        if evaluate_candidate(
            f"local_{generated_local_count + 1:02d}",
            "local",
            weights,
        ):
            generated_local_count += 1

    ranked.sort(
        key=lambda item: _summary_sort_key(item["_summary_object"]),
        reverse=True,
    )
    evolution_parents = [item["_weights_object"] for item in ranked[: max(2, min(6, len(ranked)))]]
    generated_evolution_count = 0
    evolution_attempts = 0
    while generated_evolution_count < requested_evolution and evolution_attempts < requested_evolution * 30 + 30:
        evolution_attempts += 1
        first = evolution_parents[rng.randrange(len(evolution_parents))]
        second = evolution_parents[rng.randrange(len(evolution_parents))]
        weights = _crossover_weights(
            first,
            second,
            rng=rng,
            mutation_rate=evolution_mutation_rate,
            mutation_scale=evolution_mutation_scale,
        )
        if evaluate_candidate(
            f"evolution_{generated_evolution_count + 1:02d}",
            "evolution",
            weights,
        ):
            generated_evolution_count += 1

    ranked.sort(
        key=lambda item: _summary_sort_key(item["_summary_object"]),
        reverse=True,
    )

    selected = ranked[0]
    selected_weights = selected["_weights_object"]

    selection_backtest = run_numbers_backtest(
        selection_history,
        config,
        train_window=train_window,
        tested_periods=full_tested_periods,
        top_k=top_k,
        weights=selected_weights,
    )
    selection_random_baseline = _aggregate_uniform_random_summaries([
        run_numbers_uniform_random_backtest(
            selection_history,
            config,
            train_window=train_window,
            tested_periods=full_tested_periods,
            top_k=top_k,
            seed=baseline_seed,
        )
        for baseline_seed in (seed, seed + 1, seed + 2)
    ])
    selection_box_random_baseline = _aggregate_box_random_summaries([
        run_numbers_box_random_backtest(
            selection_history, config, train_window=train_window,
            tested_periods=full_tested_periods, top_k=top_k, seed=baseline_seed,
            include_records=False,
        )
        for baseline_seed in (seed, seed + 1, seed + 2)
    ])

    holdout_evaluation: dict[str, object] = {}
    if holdout_periods > 0:
        holdout_summary = run_numbers_backtest(
            normalized_history,
            config,
            train_window=train_window,
            tested_periods=holdout_periods,
            top_k=top_k,
            weights=selected_weights,
            include_records=True,
        )
        holdout_random_baseline = _aggregate_uniform_random_summaries([
            run_numbers_uniform_random_backtest(
                normalized_history,
                config,
                train_window=train_window,
                tested_periods=holdout_periods,
                top_k=top_k,
                seed=baseline_seed,
            )
            for baseline_seed in (seed, seed + 1, seed + 2)
        ])
        holdout_box_random_baseline = _aggregate_box_random_summaries([
            run_numbers_box_random_backtest(
                normalized_history, config, train_window=train_window,
                tested_periods=holdout_periods, top_k=top_k, seed=baseline_seed,
                include_records=True,
            )
            for baseline_seed in (seed, seed + 1, seed + 2)
        ])
        holdout_evaluation = _numbers_holdout_result(
            holdout_summary,
            holdout_random_baseline,
            holdout_periods=holdout_periods,
            selection_history_draws=len(selection_history),
        )
        holdout_evaluation["box_dedicated_random_baseline"] = holdout_box_random_baseline
        box_eval = holdout_evaluation.get("box_dedicated_evaluation", {})
        if isinstance(box_eval, dict):
            box_eval["random_baseline"] = holdout_box_random_baseline
            box_eval["random_uplift"] = round(
                float(box_eval.get("box_hit_rate") or 0.0)
                - float(holdout_box_random_baseline.get("box_hit_rate") or 0.0), 6
            )
            model_records = model.get("records", []) if isinstance((model := holdout_summary.to_dict(include_records=True)), dict) else []
            random_records = holdout_box_random_baseline.get("seed_records", {}).get("0", [])
            box_eval["paired_draw_results"] = [
                {
                    "draw_index": record.get("draw_index"),
                    "actual": record.get("actual"),
                    "model_box_hit": bool(record.get("box_dedicated_hit")),
                    "random_box_hit": bool(random.get("box_hit")),
                    "model_best_unordered_matches": record.get("box_dedicated_best_unordered_matches"),
                    "random_best_unordered_matches": random.get("best_unordered_digit_matches"),
                }
                for record, random in zip(model_records, random_records)
            ]

    context = build_numbers_model_context(normalized_history, config)
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
        for index, item in enumerate(prediction_result.selected, start=1)
    ]

    box_prediction = _build_box_prediction(
        prediction_result.ranked,
        top_k=top_k,
        model_name=str(selected["config"]),
    )

    public_ranked = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in ranked
    ]
    default_result = next(
        (item for item in public_ranked if item["config"] == "default"),
        None,
    )
    default_weights = _weights_to_dict(NumbersPredictionWeights())
    applied_weights = _weights_to_dict(selected_weights)

    effective_counts = {
        "experience": restored_experience_count,
        "random": generated_random_count,
        "local": generated_local_count,
        "evolution": generated_evolution_count,
    }

    return {
        "random_baseline": selection_random_baseline,
        "default_model_baseline": default_result or {},
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
                key: round(applied_weights[key] - default_weights[key], 6)
                for key in applied_weights
            },
        },
        "search_metadata": {
            "algorithm": "numbers_adaptive_multi_source_search",
            "optimizer_connected": True,
            "candidate_space_size": prediction_result.generated_count,
            "box_prediction_enabled": True,
            "box_prediction_version": 1,
            "box_prediction_count": len(box_prediction),
            "base_config_count": len(BASE_WEIGHT_CONFIGS),
            "requested_allocation": {
                "experience": requested_experience,
                "random": requested_random,
                "local": requested_local,
                "evolution": requested_evolution,
            },
            "effective_allocation": effective_counts,
            "experience_shortfall_to_random": experience_shortfall,
            "allocation_reason": allocation.get("reason"),
            "allocation_receiver": allocation.get("receiver"),
            "allocation_donor": allocation.get("donor"),
            "allocation_sample_count": allocation.get("sample_count", 0),
            "evolution_adaptation": evolution_adaptation,
            "experience_loaded_count": len(loaded_experience),
            "experience_restored_count": restored_experience_count,
            "evaluated_config_count": len(public_ranked),
            "search_tested_periods": search_periods,
            "full_tested_periods": selection_backtest.tested_periods,
            "numbers_holdout_enabled": bool(holdout_evaluation),
            "numbers_holdout_periods": holdout_periods,
            "selection_history_draws": len(selection_history),
            "trained_through_draw_no": trained_through_draw_no,
            "seed": seed,
        },
        "feature_ablation": [],
        "optimizer_experience": {
            "game_key": game_key,
            "loaded_count": len(loaded_experience),
            "restored_count": restored_experience_count,
            "selected_from_experience": str(selected.get("source", "")) == "experience",
            "selected_source": str(selected.get("source", "")),
        },
        "numbers_backtest": selection_backtest.to_dict(),
        "numbers_selection_backtest": selection_backtest.to_dict(),
        "box_prediction_backtest": {
            **selection_backtest.to_dict().get("box_dedicated_evaluation", {}),
            "random_baseline": selection_box_random_baseline,
            "random_uplift": round(
                float(selection_backtest.to_dict().get("box_dedicated_evaluation", {}).get("box_hit_rate") or 0.0)
                - float(selection_box_random_baseline.get("box_hit_rate") or 0.0), 6
            ),
        },
        "holdout_evaluation": holdout_evaluation,
        "numbers_holdout": holdout_evaluation,
        "box_prediction_holdout": (
            holdout_evaluation.get("box_dedicated_evaluation", {})
            if holdout_evaluation
            else {}
        ),
        "trained_through_draw_no": trained_through_draw_no,
        "prediction": prediction,
        "box_prediction": box_prediction,
    }


__all__ = [
    "BASE_WEIGHT_CONFIGS",
    "DEFAULT_NUMBERS_HOLDOUT_PERIODS",
    "_resolve_holdout_periods",
    "evaluate_numbers_weights",
    "optimize_numbers",
]
