from __future__ import annotations

import math
from random import Random
from statistics import mean, pstdev
from typing import Mapping, Sequence

from backtester import BacktestSummary, run_backtest
from data_loader import dataframe_to_history
from features import build_model_context, build_shape_features
from games import LOTTO_GAMES
from optimizer_search import (
    BASE_CONFIGS,
    CONFIGS,
    build_base_candidates,
    deduplicate_configs,
    find_parent_configs,
    generate_local_candidates,
    generate_random_candidates,
    normalized_weight_dict,
)
from predictor import PredictionResult, PredictionWeights, predict


SEED = 2025

# 探索量。
# GitHub Actionsの実行時間が厳しい場合は、
# RANDOM_SEARCH_COUNTとLOCAL_SEARCH_COUNTを減らす。
RANDOM_SEARCH_COUNT = 4
LOCAL_SEARCH_COUNT = 6
PARENT_COUNT = 3
ROBUST_FINALIST_COUNT = 4
ROBUST_SEEDS = (
    SEED,
    SEED + 1,
    SEED + 2,
)


def _resolve_game_config(
    main_cols: Sequence[str],
    min_num: int,
    max_num: int,
    pick_count: int,
) -> dict[str, object]:
    normalized_main_cols = tuple(
        str(column)
        for column in main_cols
    )

    for game_config in LOTTO_GAMES.values():
        configured_main_cols = tuple(
            str(column)
            for column in game_config.get(
                "main_cols",
                (),
            )
        )

        if (
            int(game_config["min_num"])
            == int(min_num)
            and int(game_config["max_num"])
            == int(max_num)
            and int(game_config["pick_count"])
            == int(pick_count)
            and configured_main_cols
            == normalized_main_cols
        ):
            return dict(game_config)

    for game_config in LOTTO_GAMES.values():
        if (
            int(game_config["min_num"])
            == int(min_num)
            and int(game_config["max_num"])
            == int(max_num)
            and int(game_config["pick_count"])
            == int(pick_count)
        ):
            return dict(game_config)

    raise ValueError(
        "Could not resolve lottery configuration: "
        f"min_num={min_num}, "
        f"max_num={max_num}, "
        f"pick_count={pick_count}, "
        f"main_cols={list(main_cols)}"
    )


def _merge_config(
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(game_config)

    if optimizer_config is None:
        return merged

    filters = optimizer_config.get(
        "f",
        {},
    )

    if isinstance(filters, Mapping):
        merged.update(filters)

    return merged


def _prediction_weights(
    config: Mapping[str, object],
) -> PredictionWeights:
    raw_weights = config.get(
        "w",
        {},
    )

    if not isinstance(
        raw_weights,
        Mapping,
    ):
        raw_weights = {}

    normalized = normalized_weight_dict(
        raw_weights
    )

    distribution_weight = normalized["dist"]
    shape_weight = distribution_weight / 6.0

    return PredictionWeights(
        global_frequency=normalized["freq"],
        recent_frequency=normalized["recent"],
        delay=normalized["delay"],
        pair=normalized["pair"],
        triplet=normalized["triplet"],
        repeat=normalized["repeat"],
        sum_shape=shape_weight,
        odd_shape=shape_weight,
        low_shape=shape_weight,
        consecutive_shape=shape_weight,
        span_shape=shape_weight,
        block_shape=shape_weight,
        diversity=0.35,
    )


def _random_weights() -> PredictionWeights:
    return PredictionWeights(
        global_frequency=0.0,
        recent_frequency=0.0,
        delay=0.0,
        pair=0.0,
        triplet=0.0,
        repeat=0.0,
        sum_shape=0.0,
        odd_shape=0.0,
        low_shape=0.0,
        consecutive_shape=0.0,
        span_shape=0.0,
        block_shape=0.0,
        diversity=0.0,
    )


def _summary_to_result(
    summary: BacktestSummary,
    *,
    config_name: str,
) -> dict[str, object]:
    return {
        "config": config_name,
        "tested_periods": summary.tested_periods,
        "avg_matches": (
            summary.average_best_matches
        ),
        "average_matches_per_ticket": (
            summary.average_matches_per_ticket
        ),
        "hit_rate_1match": (
            summary.hit_rate_1match
        ),
        "hit_rate_2match": (
            summary.hit_rate_2match
        ),
        "hit_rate_3match": (
            summary.hit_rate_3match
        ),
        "hit_rate_4match": (
            summary.hit_rate_4match
        ),
        "hit_rate_5match": (
            summary.hit_rate_5match
        ),
        "hit_rate_6match": (
            summary.hit_rate_6match
        ),
        "hit_rate_7match": (
            summary.hit_rate_7match
        ),
    }


def _run_backtest_result(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    *,
    config_name: str,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    weights: PredictionWeights,
    seed: int,
) -> dict[str, object]:
    summary = run_backtest(
        history,
        game_config,
        train_window=train_window,
        tested_periods=tested_periods,
        candidate_count=candidate_count,
        top_k=1,
        weights=weights,
        seed=seed,
        include_records=False,
    )

    return _summary_to_result(
        summary,
        config_name=config_name,
    )


def _aggregate_seed_results(
    results: Sequence[Mapping[str, object]],
    *,
    config_name: str,
) -> dict[str, object]:
    if not results:
        raise ValueError(
            "results must not be empty."
        )

    metric_keys = (
        "avg_matches",
        "average_matches_per_ticket",
        "hit_rate_1match",
        "hit_rate_2match",
        "hit_rate_3match",
        "hit_rate_4match",
        "hit_rate_5match",
        "hit_rate_6match",
        "hit_rate_7match",
    )

    aggregated: dict[str, object] = {
        "config": config_name,
        "tested_periods": int(
            results[0].get(
                "tested_periods"
            )
            or 0
        ),
        "evaluated_seeds": len(results),
    }

    for key in metric_keys:
        values = [
            float(item.get(key) or 0.0)
            for item in results
        ]

        aggregated[key] = round(
            float(mean(values)),
            6,
        )

    avg_values = [
        float(
            item.get("avg_matches")
            or 0.0
        )
        for item in results
    ]

    aggregated["avg_matches_std"] = round(
        (
            float(pstdev(avg_values))
            if len(avg_values) > 1
            else 0.0
        ),
        6,
    )

    aggregated["seed_avg_matches"] = [
        round(value, 6)
        for value in avg_values
    ]

    return aggregated


def _prediction_to_legacy(
    prediction: PredictionResult,
    *,
    context,
    model_name: str,
) -> list[dict[str, object]]:
    estimated_probability = 1 / math.comb(
        context.max_num
        - context.min_num
        + 1,
        context.pick_count,
    )

    converted: list[
        dict[str, object]
    ] = []

    for index, item in enumerate(
        prediction.selected,
        start=1,
    ):
        shape = build_shape_features(
            item.candidate,
            min_num=context.min_num,
            max_num=context.max_num,
            ranges=context.block_ranges,
        )

        repeat_count = (
            int(item.repeat_counts[0])
            if item.repeat_counts
            else 0
        )

        converted.append({
            "pattern_id": f"P{index}",
            "numbers": list(
                item.candidate
            ),
            "score": round(
                float(item.total_score),
                6,
            ),
            "model": model_name,
            "block_counts": list(
                shape.block_counts
            ),
            "consecutive_count": int(
                shape.consecutive_pairs
            ),
            "repeat_count": repeat_count,
            "estimated_probability": (
                estimated_probability
            ),
        })

    return converted


def selection_score(
    result: Mapping[str, object],
    random_avg: float | None,
) -> float:
    """
    設定名に依存しない評価関数。

    平均一致数と高一致率を加点し、
    ランダム比の改善を加点し、
    seed間の不安定さを減点する。
    """
    avg = float(
        result.get("avg_matches")
        or 0.0
    )

    uplift = (
        0.0
        if random_avg is None
        else avg - float(random_avg)
    )

    stability_std = float(
        result.get("avg_matches_std")
        or 0.0
    )

    score = (
        avg
        + 0.30
        * float(
            result.get(
                "hit_rate_2match"
            )
            or 0.0
        )
        + 0.80
        * float(
            result.get(
                "hit_rate_3match"
            )
            or 0.0
        )
        + 1.20
        * float(
            result.get(
                "hit_rate_4match"
            )
            or 0.0
        )
        + 1.60
        * float(
            result.get(
                "hit_rate_5match"
            )
            or 0.0
        )
        + 2.00
        * float(
            result.get(
                "hit_rate_6match"
            )
            or 0.0
        )
        + 2.40
        * float(
            result.get(
                "hit_rate_7match"
            )
            or 0.0
        )
        + 0.35 * uplift
        - 0.20 * stability_std
    )

    if (
        random_avg is not None
        and avg < float(random_avg) - 0.05
    ):
        score -= 0.25

    return round(
        float(score),
        6,
    )


def _evaluate_config(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
    random_baselines: Mapping[
        int,
        Mapping[str, object],
    ],
) -> dict[str, object]:
    merged_config = _merge_config(
        game_config,
        optimizer_config,
    )

    weights = _prediction_weights(
        optimizer_config
    )

    name = str(
        optimizer_config["name"]
    )

    per_seed = [
        _run_backtest_result(
            history,
            merged_config,
            config_name=name,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            weights=weights,
            seed=seed,
        )
        for seed in seeds
    ]

    result = _aggregate_seed_results(
        per_seed,
        config_name=name,
    )

    random_results = [
        random_baselines[seed]
        for seed in seeds
    ]

    random_aggregate = (
        _aggregate_seed_results(
            random_results,
            config_name="random",
        )
    )

    random_avg = float(
        random_aggregate.get(
            "avg_matches"
        )
        or 0.0
    )

    result["selection_score"] = (
        selection_score(
            result,
            random_avg,
        )
    )

    result["random_unfiltered_avg"] = (
        random_avg
    )

    result["random_filtered_avg"] = (
        random_avg
    )

    result["random_uplift"] = round(
        float(
            result.get("avg_matches")
            or 0.0
        )
        - random_avg,
        6,
    )

    result[
        "random_filtered_baseline"
    ] = random_aggregate

    raw_weights = optimizer_config.get(
        "w",
        {},
    )

    result["weights"] = (
        dict(raw_weights)
        if isinstance(
            raw_weights,
            Mapping,
        )
        else {}
    )

    filters = optimizer_config.get(
        "f",
        {},
    )

    result["filters"] = (
        dict(filters)
        if isinstance(
            filters,
            Mapping,
        )
        else {}
    )

    result["search_origin"] = (
        optimizer_config.get(
            "search_origin"
        )
    )

    result["parent"] = (
        optimizer_config.get(
            "parent"
        )
    )

    return result


def _build_random_baselines(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
) -> dict[int, dict[str, object]]:
    random_weights = _random_weights()

    return {
        seed: _run_backtest_result(
            history,
            game_config,
            config_name="random",
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            weights=random_weights,
            seed=seed,
        )
        for seed in ROBUST_SEEDS
    }


def _evaluate_configs(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    configs: Sequence[Mapping[str, object]],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
    random_baselines: Mapping[
        int,
        Mapping[str, object],
    ],
) -> list[dict[str, object]]:
    results = [
        _evaluate_config(
            history,
            game_config,
            config,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            seeds=seeds,
            random_baselines=random_baselines,
        )
        for config in configs
    ]

    results.sort(
        key=lambda item: float(
            item["selection_score"]
        ),
        reverse=True,
    )

    return results


def _replace_with_robust_results(
    preliminary_results: Sequence[
        Mapping[str, object]
    ],
    robust_results: Mapping[
        str,
        Mapping[str, object],
    ],
) -> list[dict[str, object]]:
    ranked_results = [
        dict(
            robust_results.get(
                str(result["config"]),
                result,
            )
        )
        for result in preliminary_results
    ]

    ranked_results.sort(
        key=lambda item: float(
            item["selection_score"]
        ),
        reverse=True,
    )

    return ranked_results


def optimize(
    df,
    main_cols,
    min_num,
    max_num,
    pick_count,
    train_window,
    tested_periods,
    bt_candidates,
    final_candidates,
):
    """
    固定設定、ランダム探索、局所探索を段階的に実行する。

    最終候補は複数seedで再評価し、
    安定性を確認してから次回候補を生成する。

    引数および主要な戻り値は、
    既存のmain.pyとrun_pipeline.pyとの互換性を維持する。
    """
    train_window = int(train_window)
    tested_periods = int(tested_periods)
    bt_candidates = int(bt_candidates)
    final_candidates = int(
        final_candidates
    )

    game_config = _resolve_game_config(
        main_cols,
        min_num,
        max_num,
        pick_count,
    )

    game_config.update({
        "main_cols": tuple(main_cols),
        "min_num": int(min_num),
        "max_num": int(max_num),
        "pick_count": int(pick_count),
        "train_window": train_window,
        "tested_periods": tested_periods,
        "backtest_candidates": (
            bt_candidates
        ),
        "final_candidates": (
            final_candidates
        ),
    })

    history = dataframe_to_history(
        df,
        game_config,
    )

    random_baselines = (
        _build_random_baselines(
            history,
            game_config,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=bt_candidates,
        )
    )

    rng = Random(
        SEED
        + int(max_num) * 100
        + int(pick_count)
    )

    base_candidates = (
        build_base_candidates()
    )

    inherited_filters = dict(
        base_candidates[0].get(
            "f",
            {},
        )
    )

    random_candidates = (
        generate_random_candidates(
            count=RANDOM_SEARCH_COUNT,
            rng=rng,
            inherited_filters=(
                inherited_filters
            ),
        )
    )

    stage_one_configs = (
        deduplicate_configs([
            *base_candidates,
            *random_candidates,
        ])
    )

    stage_one_results = (
        _evaluate_configs(
            history,
            game_config,
            stage_one_configs,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=bt_candidates,
            seeds=(SEED,),
            random_baselines=(
                random_baselines
            ),
        )
    )

    parent_configs = find_parent_configs(
        stage_one_results,
        stage_one_configs,
        parent_count=PARENT_COUNT,
    )

    local_candidates = (
        generate_local_candidates(
            parent_configs,
            count=LOCAL_SEARCH_COUNT,
            rng=rng,
        )
    )

    all_configs = deduplicate_configs([
        *stage_one_configs,
        *local_candidates,
    ])

    evaluated_names = {
        str(result["config"])
        for result in stage_one_results
    }

    unevaluated_configs = [
        config
        for config in all_configs
        if str(config["name"])
        not in evaluated_names
    ]

    local_results = _evaluate_configs(
        history,
        game_config,
        unevaluated_configs,
        train_window=train_window,
        tested_periods=tested_periods,
        candidate_count=bt_candidates,
        seeds=(SEED,),
        random_baselines=random_baselines,
    )

    preliminary_results = [
        *stage_one_results,
        *local_results,
    ]

    preliminary_results.sort(
        key=lambda item: float(
            item["selection_score"]
        ),
        reverse=True,
    )

    all_config_by_name = {
        str(config["name"]): config
        for config in all_configs
    }

    finalist_names = [
        str(result["config"])
        for result in preliminary_results[
            :ROBUST_FINALIST_COUNT
        ]
    ]

    robust_results_by_name = {
        name: _evaluate_config(
            history,
            game_config,
            all_config_by_name[name],
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=bt_candidates,
            seeds=ROBUST_SEEDS,
            random_baselines=(
                random_baselines
            ),
        )
        for name in finalist_names
    }

    ranked_results = (
        _replace_with_robust_results(
            preliminary_results,
            robust_results_by_name,
        )
    )

    if not ranked_results:
        raise RuntimeError(
            "No optimizer configurations "
            "were evaluated."
        )

    best_result = ranked_results[0]
    best_name = str(
        best_result["config"]
    )
    best_config = (
        all_config_by_name[best_name]
    )

    final_config = _merge_config(
        game_config,
        best_config,
    )

    final_weights = _prediction_weights(
        best_config
    )

    final_context = build_model_context(
        history,
        final_config,
    )

    final_prediction = predict(
        final_context,
        final_config,
        candidate_count=final_candidates,
        top_k=5,
        weights=final_weights,
        seed=SEED,
    )

    prediction = _prediction_to_legacy(
        final_prediction,
        context=final_context,
        model_name=best_name,
    )

    selected_random_baseline = (
        best_result[
            "random_filtered_baseline"
        ]
    )

    return {
        "random_baseline": (
            _aggregate_seed_results(
                [
                    random_baselines[seed]
                    for seed in ROBUST_SEEDS
                ],
                config_name="random",
            )
        ),
        "selected_random_filtered_baseline": (
            selected_random_baseline
        ),
        "ranked_configs": ranked_results,
        "selected_config": best_name,
        "selected_weights": dict(
            best_config["w"]
        ),
        "selected_filters": dict(
            best_config.get(
                "f",
                {},
            )
        ),
        "search_metadata": {
            "algorithm": (
                "fixed_random_local_robust"
            ),
            "base_config_count": len(
                base_candidates
            ),
            "random_config_count": len(
                random_candidates
            ),
            "local_config_count": len(
                local_candidates
            ),
            "total_unique_config_count": (
                len(all_configs)
            ),
            "parent_count": PARENT_COUNT,
            "robust_finalist_count": (
                ROBUST_FINALIST_COUNT
            ),
            "robust_seeds": list(
                ROBUST_SEEDS
            ),
            "search_module": (
                "optimizer_search"
            ),
            "note": (
                "Optimizer-specific "
                "max_block/max_first/max_con/"
                "max_common are retained for "
                "output compatibility, but "
                "predictor.py currently does not "
                "consume those keys. Therefore "
                "this version optimizes only "
                "effective prediction weights."
            ),
        },
        "prediction": prediction,
    }