from __future__ import annotations

import math
from random import Random
from statistics import mean, pstdev
from typing import Mapping, Sequence

from backtester import (
    BacktestSummary,
    run_backtest,
    run_filtered_random_backtest,
    run_uniform_random_backtest,
)
from data_loader import dataframe_to_history
from features import build_model_context, build_shape_features
from games import LOTTO_GAMES
from predictor import PredictionResult, PredictionWeights, predict
from optimizer_evolution import generate_evolution_candidates
from optimizer_ablation import (
    run_feature_ablation,
)
from optimizer_experience import (
    load_evolution_adaptation,
    load_experience_configs,
    load_search_allocation,
)


SEED = 2025

# 探索量。GitHub Actionsの実行時間が厳しい場合は、
# RANDOM_SEARCH_COUNTとLOCAL_SEARCH_COUNTを減らす。
RANDOM_SEARCH_COUNT = 4
LOCAL_SEARCH_COUNT = 6
PARENT_COUNT = 3
ROBUST_FINALIST_COUNT = 4
ROBUST_SEEDS = (SEED, SEED + 1, SEED + 2)

WEIGHT_KEYS = (
    "freq",
    "recent",
    "pair",
    "triplet",
    "delay",
    "dist",
    "repeat",
)


# 既存の固定設定。探索の初期点・比較基準として残す。
BASE_CONFIGS = [
    {
        "name": "balanced_strict",
        "w": {
            "freq": 0.22,
            "recent": 0.24,
            "pair": 0.22,
            "triplet": 0.08,
            "delay": 0.08,
            "dist": 0.16,
        },
        "s": {"g": 0.35, "r": 0.40, "d": 0.25},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "balanced_loose",
        "w": {
            "freq": 0.22,
            "recent": 0.24,
            "pair": 0.22,
            "triplet": 0.08,
            "delay": 0.08,
            "dist": 0.16,
        },
        "s": {"g": 0.35, "r": 0.40, "d": 0.25},
        "f": {
            "max_block": 3,
            "max_first": 3,
            "max_con": 2,
            "max_common": 4,
        },
    },
    {
        "name": "no_delay_strict",
        "w": {
            "freq": 0.24,
            "recent": 0.26,
            "pair": 0.24,
            "triplet": 0.08,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {"g": 0.45, "r": 0.55, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "no_delay_loose",
        "w": {
            "freq": 0.24,
            "recent": 0.26,
            "pair": 0.24,
            "triplet": 0.08,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {"g": 0.45, "r": 0.55, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 3,
            "max_con": 2,
            "max_common": 4,
        },
    },
    {
        "name": "freq_pair_strict",
        "w": {
            "freq": 0.30,
            "recent": 0.18,
            "pair": 0.28,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {"g": 0.65, "r": 0.35, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "recent_pair_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.34,
            "pair": 0.26,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.20,
        },
        "s": {"g": 0.25, "r": 0.75, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "delay_light_strict",
        "w": {
            "freq": 0.18,
            "recent": 0.20,
            "pair": 0.20,
            "triplet": 0.05,
            "delay": 0.15,
            "dist": 0.22,
        },
        "s": {"g": 0.30, "r": 0.35, "d": 0.35},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "dist_heavy_strict",
        "w": {
            "freq": 0.18,
            "recent": 0.20,
            "pair": 0.20,
            "triplet": 0.04,
            "delay": 0.04,
            "dist": 0.34,
        },
        "s": {"g": 0.40, "r": 0.45, "d": 0.15},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "repeat_light_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.30,
            "pair": 0.24,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.18,
            "repeat": 0.08,
        },
        "s": {"g": 0.25, "r": 0.75, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "repeat_medium_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.27,
            "pair": 0.22,
            "triplet": 0.05,
            "delay": 0.00,
            "dist": 0.17,
            "repeat": 0.15,
        },
        "s": {"g": 0.25, "r": 0.75, "d": 0.00},
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
]

# 旧コードや外部参照との互換性。
CONFIGS = BASE_CONFIGS


def _resolve_game_config(
    main_cols: Sequence[str],
    min_num: int,
    max_num: int,
    pick_count: int,
) -> dict[str, object]:
    normalized_main_cols = tuple(str(column) for column in main_cols)

    for game_config in LOTTO_GAMES.values():
        configured_main_cols = tuple(
            str(column)
            for column in game_config.get("main_cols", ())
        )
        if (
            int(game_config["min_num"]) == int(min_num)
            and int(game_config["max_num"]) == int(max_num)
            and int(game_config["pick_count"]) == int(pick_count)
            and configured_main_cols == normalized_main_cols
        ):
            return dict(game_config)

    for game_config in LOTTO_GAMES.values():
        if (
            int(game_config["min_num"]) == int(min_num)
            and int(game_config["max_num"]) == int(max_num)
            and int(game_config["pick_count"]) == int(pick_count)
        ):
            return dict(game_config)

    raise ValueError(
        "Could not resolve lottery configuration: "
        f"min_num={min_num}, max_num={max_num}, "
        f"pick_count={pick_count}, main_cols={list(main_cols)}"
    )


def _merge_config(
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(game_config)

    if optimizer_config is None:
        return merged

    filters = optimizer_config.get("f", {})
    if isinstance(filters, Mapping):
        merged.update(filters)

    return merged


def _normalized_weight_dict(
    raw_weights: Mapping[str, object],
) -> dict[str, float]:
    values = {
        key: max(0.0, float(raw_weights.get(key, 0.0)))
        for key in WEIGHT_KEYS
    }
    total = sum(values.values())

    if total <= 0:
        equal = 1.0 / len(WEIGHT_KEYS)
        return {key: equal for key in WEIGHT_KEYS}

    return {
        key: round(value / total, 8)
        for key, value in values.items()
    }


def _config_signature(
    config: Mapping[str, object],
) -> tuple[float, ...]:
    raw_weights = config.get("w", {})
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    normalized = _normalized_weight_dict(raw_weights)
    return tuple(round(normalized[key], 6) for key in WEIGHT_KEYS)


def _copy_search_config(
    config: Mapping[str, object],
    *,
    name: str,
    origin: str,
    parent: str | None = None,
) -> dict[str, object]:
    raw_weights = config.get("w", {})
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    filters = config.get("f", {})
    if not isinstance(filters, Mapping):
        filters = {}

    scoring = config.get("s", {})
    if not isinstance(scoring, Mapping):
        scoring = {}

    return {
        "name": name,
        "w": _normalized_weight_dict(raw_weights),
        "s": dict(scoring),
        "f": dict(filters),
        "search_origin": origin,
        "parent": parent,
    }


def _build_base_candidates() -> list[dict[str, object]]:
    return [
        _copy_search_config(
            config,
            name=str(config["name"]),
            origin="fixed",
        )
        for config in BASE_CONFIGS
    ]


def _generate_random_candidates(
    *,
    count: int,
    rng: Random,
    inherited_filters: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    単体上のランダム探索。

    指数分布から正の値を作って正規化するため、
    全重みの合計は常に1になる。
    """
    candidates: list[dict[str, object]] = []

    for index in range(1, count + 1):
        raw = {
            key: -math.log(max(rng.random(), 1e-12))
            for key in WEIGHT_KEYS
        }

        # tripletとdelayだけが極端に支配しにくいよう軽く抑える。
        raw["triplet"] *= 0.65
        raw["delay"] *= 0.80

        candidates.append({
            "name": f"random_{index:02d}",
            "w": _normalized_weight_dict(raw),
            "s": {},
            "f": dict(inherited_filters),
            "search_origin": "random",
            "parent": None,
        })

    return candidates


def _mutate_weights(
    weights: Mapping[str, object],
    *,
    rng: Random,
    scale: float,
) -> dict[str, float]:
    normalized = _normalized_weight_dict(weights)
    mutated: dict[str, float] = {}

    for key in WEIGHT_KEYS:
        base = normalized[key]
        additive = rng.gauss(0.0, scale)
        multiplicative = math.exp(rng.gauss(0.0, scale * 0.75))
        mutated[key] = max(0.0, base * multiplicative + additive)

    return _normalized_weight_dict(mutated)


def _generate_local_candidates(
    parents: Sequence[Mapping[str, object]],
    *,
    count: int,
    rng: Random,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    if not parents:
        return candidates

    scales = (0.025, 0.05, 0.08)

    for index in range(1, count + 1):
        parent = parents[(index - 1) % len(parents)]
        parent_weights = parent.get("w", {})
        if not isinstance(parent_weights, Mapping):
            parent_weights = {}

        parent_filters = parent.get("f", {})
        if not isinstance(parent_filters, Mapping):
            parent_filters = {}

        parent_name = str(parent["name"])
        scale = scales[(index - 1) % len(scales)]

        candidates.append({
            "name": f"local_{index:02d}_{parent_name}",
            "w": _mutate_weights(
                parent_weights,
                rng=rng,
                scale=scale,
            ),
            "s": {},
            "f": dict(parent_filters),
            "search_origin": "local",
            "parent": parent_name,
        })

    return candidates


def _deduplicate_configs(
    configs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    signatures: set[tuple[float, ...]] = set()

    for config in configs:
        signature = _config_signature(config)
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(dict(config))

    return unique


def _prediction_weights(
    config: Mapping[str, object],
) -> PredictionWeights:
    raw_weights = config.get("w", {})
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    normalized = _normalized_weight_dict(raw_weights)
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
        "avg_matches": summary.average_best_matches,
        "average_matches_per_ticket": summary.average_matches_per_ticket,
        "hit_rate_1match": summary.hit_rate_1match,
        "hit_rate_2match": summary.hit_rate_2match,
        "hit_rate_3match": summary.hit_rate_3match,
        "hit_rate_4match": summary.hit_rate_4match,
        "hit_rate_5match": summary.hit_rate_5match,
        "hit_rate_6match": summary.hit_rate_6match,
        "hit_rate_7match": summary.hit_rate_7match,
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
    return _summary_to_result(summary, config_name=config_name)


def _run_random_backtest_result(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    *,
    config_name: str,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seed: int,
    filtered: bool,
) -> dict[str, object]:
    if filtered:
        summary = run_filtered_random_backtest(
            history,
            game_config,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            top_k=1,
            seed=seed,
            include_records=False,
        )
    else:
        summary = run_uniform_random_backtest(
            history,
            game_config,
            train_window=train_window,
            tested_periods=tested_periods,
            top_k=1,
            seed=seed,
            include_records=False,
        )
    return _summary_to_result(summary, config_name=config_name)


def _aggregate_seed_results(
    results: Sequence[Mapping[str, object]],
    *,
    config_name: str,
) -> dict[str, object]:
    if not results:
        raise ValueError("results must not be empty.")

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
        "tested_periods": int(results[0].get("tested_periods") or 0),
        "evaluated_seeds": len(results),
    }

    for key in metric_keys:
        values = [float(item.get(key) or 0.0) for item in results]
        aggregated[key] = round(float(mean(values)), 6)

    avg_values = [float(item.get("avg_matches") or 0.0) for item in results]
    aggregated["avg_matches_std"] = round(
        float(pstdev(avg_values)) if len(avg_values) > 1 else 0.0,
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
        context.max_num - context.min_num + 1,
        context.pick_count,
    )

    converted: list[dict[str, object]] = []

    for index, item in enumerate(prediction.selected, start=1):
        shape = build_shape_features(
            item.candidate,
            min_num=context.min_num,
            max_num=context.max_num,
            ranges=context.block_ranges,
        )

        repeat_count = int(item.repeat_counts[0]) if item.repeat_counts else 0

        converted.append({
            "pattern_id": f"P{index}",
            "numbers": list(item.candidate),
            "score": round(float(item.total_score), 6),
            "model": model_name,
            "block_counts": list(shape.block_counts),
            "consecutive_count": int(shape.consecutive_pairs),
            "repeat_count": repeat_count,
            "estimated_probability": estimated_probability,
        })

    return converted


def selection_score(
    result: Mapping[str, object],
    random_avg: float | None,
) -> float:
    """
    設定名に依存しない純粋な評価関数。

    平均一致数と高一致率を加点し、
    ランダム比の改善を加点、
    seed間の不安定さを減点する。
    """
    avg = float(result.get("avg_matches") or 0.0)
    uplift = 0.0 if random_avg is None else avg - float(random_avg)
    stability_std = float(result.get("avg_matches_std") or 0.0)

    score = (
        avg
        + 0.30 * float(result.get("hit_rate_2match") or 0.0)
        + 0.80 * float(result.get("hit_rate_3match") or 0.0)
        + 1.20 * float(result.get("hit_rate_4match") or 0.0)
        + 1.60 * float(result.get("hit_rate_5match") or 0.0)
        + 2.00 * float(result.get("hit_rate_6match") or 0.0)
        + 2.40 * float(result.get("hit_rate_7match") or 0.0)
        + 0.35 * uplift
        - 0.20 * stability_std
    )

    if random_avg is not None and avg < float(random_avg) - 0.05:
        score -= 0.25

    return round(float(score), 6)


def _evaluate_config(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
    random_baselines: Mapping[int, Mapping[str, object]],
    filtered_random_baselines: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    merged_config = _merge_config(game_config, optimizer_config)
    weights = _prediction_weights(optimizer_config)
    name = str(optimizer_config["name"])

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

    result = _aggregate_seed_results(per_seed, config_name=name)

    uniform_results = [random_baselines[seed] for seed in seeds]
    filtered_results = [filtered_random_baselines[seed] for seed in seeds]
    uniform_aggregate = _aggregate_seed_results(
        uniform_results, config_name="uniform_random"
    )
    filtered_aggregate = _aggregate_seed_results(
        filtered_results, config_name="filtered_random"
    )
    uniform_avg = float(uniform_aggregate.get("avg_matches") or 0.0)
    filtered_avg = float(filtered_aggregate.get("avg_matches") or 0.0)

    result["selection_score"] = selection_score(result, uniform_avg)
    result["random_unfiltered_avg"] = uniform_avg
    result["random_filtered_avg"] = filtered_avg
    result["random_uplift"] = round(
        float(result.get("avg_matches") or 0.0) - uniform_avg,
        6,
    )
    result["random_filtered_baseline"] = filtered_aggregate
    result["weights"] = dict(optimizer_config["w"])
    result["filters"] = dict(optimizer_config.get("f", {}))
    result["search_origin"] = optimizer_config.get("search_origin")
    result["parent"] = optimizer_config.get("parent")

    return result


def _rank_robust_finalists(
    robust_results_by_name: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return robust finalist results ordered by selection score."""
    ranked = sorted(
        (dict(result) for result in robust_results_by_name.values()),
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )

    if not ranked:
        raise RuntimeError(
            "No robust optimizer finalists were evaluated."
        )

    return ranked


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
    固定設定、ランダム探索、局所探索を段階的に実行し、
    複数seedで最終候補の安定性を確認してから次回候補を生成する。

    引数と主要な戻り値は旧main.pyとの互換性を維持する。
    """
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
        "train_window": int(train_window),
        "tested_periods": int(tested_periods),
        "backtest_candidates": int(bt_candidates),
        "final_candidates": int(final_candidates),
    })

    history = dataframe_to_history(df, game_config)

    random_baselines: dict[int, dict[str, object]] = {}
    filtered_random_baselines: dict[int, dict[str, object]] = {}
    for seed in ROBUST_SEEDS:
        random_baselines[seed] = _run_random_backtest_result(
            history, game_config, config_name="uniform_random",
            train_window=int(train_window), tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates), seed=seed, filtered=False,
        )
        filtered_random_baselines[seed] = _run_random_backtest_result(
            history, game_config, config_name="filtered_random",
            train_window=int(train_window), tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates), seed=seed, filtered=True,
        )

    rng = Random(SEED + int(max_num) * 100 + int(pick_count))

    game_name = str(
        game_config.get("kind")
        or game_config.get("key")
        or game_config.get("display_name")
        or "unknown"
    ).lower()
    search_allocation = load_search_allocation(game_name)
    allocation_counts = search_allocation.get("counts", {})
    if not isinstance(allocation_counts, Mapping):
        allocation_counts = {}

    experience_count = max(
        0,
        int(allocation_counts.get("experience", 3)),
    )
    random_count = max(
        0,
        int(allocation_counts.get("random", RANDOM_SEARCH_COUNT)),
    )
    local_count = max(
        0,
        int(allocation_counts.get("local", LOCAL_SEARCH_COUNT)),
    )
    evolution_count = max(
        0,
        int(allocation_counts.get("evolution", 4)),
    )

    evolution_adaptation = load_evolution_adaptation(game_name)
    mutation_rate = float(
        evolution_adaptation.get("mutation_rate", 0.25)
    )
    mutation_scale = float(
        evolution_adaptation.get("mutation_scale", 0.08)
    )

    base_candidates = _build_base_candidates()
    inherited_filters = dict(base_candidates[0].get("f", {}))

    raw_experience_candidates = load_experience_configs(
        game_name,
        limit=experience_count,
    )
    experience_candidates = [
        _copy_search_config(
            config,
            name=str(config["name"]),
            origin="experience",
        )
        for config in raw_experience_candidates
    ]
    experience_shortfall = max(
        0,
        experience_count - len(experience_candidates),
    )
    effective_random_count = (
        random_count + experience_shortfall
    )

    random_candidates = _generate_random_candidates(
        count=effective_random_count,
        rng=rng,
        inherited_filters=inherited_filters,
    )

    stage_one_configs = _deduplicate_configs(
        [
            *base_candidates,
            *experience_candidates,
            *random_candidates,
        ]
    )

    stage_one_results = [
        _evaluate_config(
            history,
            game_config,
            config,
            train_window=int(train_window),
            tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates),
            seeds=(SEED,),
            random_baselines=random_baselines,
            filtered_random_baselines=filtered_random_baselines,
        )
        for config in stage_one_configs
    ]
    stage_one_results.sort(
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )

    config_by_name = {
        str(config["name"]): config
        for config in stage_one_configs
    }
    parent_configs = [
        config_by_name[str(result["config"])]
        for result in stage_one_results[:PARENT_COUNT]
    ]

    local_candidates = _generate_local_candidates(
        parent_configs,
        count=local_count,
        rng=rng,
    )

    raw_evolution_candidates = generate_evolution_candidates(
        parent_configs,
        count=evolution_count,
        rng=rng,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
    )
    evolution_candidates = [
        _copy_search_config(
            config,
            name=str(config["name"]),
            origin="evolution",
        )
        for config in raw_evolution_candidates
    ]

    all_configs = _deduplicate_configs(
        [
            *stage_one_configs,
            *local_candidates,
            *evolution_candidates,
        ]
    )
    evaluated_names = {
        str(result["config"])
        for result in stage_one_results
    }

    local_results = [
        _evaluate_config(
            history,
            game_config,
            config,
            train_window=int(train_window),
            tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates),
            seeds=(SEED,),
            random_baselines=random_baselines,
            filtered_random_baselines=filtered_random_baselines,
        )
        for config in all_configs
        if str(config["name"]) not in evaluated_names
    ]

    preliminary_results = [
        *stage_one_results,
        *local_results,
    ]
    preliminary_results.sort(
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )

    all_config_by_name = {
        str(config["name"]): config
        for config in all_configs
    }

    finalist_names = [
        str(result["config"])
        for result in preliminary_results[:ROBUST_FINALIST_COUNT]
    ]

    robust_results_by_name: dict[str, dict[str, object]] = {}
    for name in finalist_names:
        robust_results_by_name[name] = _evaluate_config(
            history,
            game_config,
            all_config_by_name[name],
            train_window=int(train_window),
            tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates),
            seeds=ROBUST_SEEDS,
            random_baselines=random_baselines,
            filtered_random_baselines=filtered_random_baselines,
        )

    robust_ranked_results = _rank_robust_finalists(
        robust_results_by_name
    )

    best_result = robust_ranked_results[0]
    best_name = str(best_result["config"])
    best_config = all_config_by_name[best_name]

    feature_ablation = run_feature_ablation(
        history,
        game_config,
        best_config,
        best_result,
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
        seeds=ROBUST_SEEDS,
        random_baselines=random_baselines,
        filtered_random_baselines=filtered_random_baselines,
    )

    final_config = _merge_config(
    final_weights = _prediction_weights(best_config)
    final_context = build_model_context(history, final_config)

    final_prediction = predict(
        final_context,
        final_config,
        candidate_count=int(final_candidates),
        top_k=5,
        weights=final_weights,
        seed=SEED,
    )

    prediction = _prediction_to_legacy(
        final_prediction,
        context=final_context,
        model_name=best_name,
    )

    selected_random_baseline = best_result["random_filtered_baseline"]

    return {
        "random_baseline": _aggregate_seed_results(
            [random_baselines[seed] for seed in ROBUST_SEEDS],
            config_name="uniform_random",
        ),
        "selected_random_filtered_baseline": selected_random_baseline,
        "ranked_configs": robust_ranked_results,
        "selected_config": best_name,
        "selected_weights": dict(best_config["w"]),
        "selected_filters": dict(best_config.get("f", {})),
        "search_metadata": {
            "algorithm": "adaptive_multi_source_robust",
            "game_name": game_name,
            "base_config_count": len(base_candidates),
            "experience_config_count": len(experience_candidates),
            "random_config_count": len(random_candidates),
            "local_config_count": len(local_candidates),
            "evolution_config_count": len(evolution_candidates),
            "total_unique_config_count": len(all_configs),
            "requested_search_allocation": {
                "experience": experience_count,
                "random": random_count,
                "local": local_count,
                "evolution": evolution_count,
            },
            "effective_search_allocation": {
                "experience": len(experience_candidates),
                "random": len(random_candidates),
                "local": len(local_candidates),
                "evolution": len(evolution_candidates),
            },
            "experience_shortfall_to_random": experience_shortfall,
            "search_allocation": search_allocation,
            "evolution_adaptation": evolution_adaptation,
            "parent_count": PARENT_COUNT,
            "robust_finalist_count": ROBUST_FINALIST_COUNT,
            "robust_seeds": list(ROBUST_SEEDS),
            "note": (
                "Optimizer-specific max_block/max_first/max_con/max_common "
                "are retained for output compatibility, but predictor.py "
                "currently does not consume those keys. Therefore this "
                "version optimizes only effective prediction weights."
            ),
        },
        "prediction": prediction,
        "feature_ablation": feature_ablation,
    }
