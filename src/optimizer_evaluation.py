from __future__ import annotations

from statistics import mean, pstdev
from typing import Mapping, Sequence

from backtester import (
    BacktestSummary,
    run_backtest,
    run_filtered_random_backtest,
    run_uniform_random_backtest,
)
from optimizer_search import normalized_weight_dict
from predictor import PredictionWeights


def merge_config(
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """ゲーム設定へoptimizer固有のフィルター設定を統合する。"""
    merged = dict(game_config)

    if optimizer_config is None:
        return merged

    filters = optimizer_config.get("f", {})

    if isinstance(filters, Mapping):
        merged.update(filters)

    return merged


def prediction_weights(
    config: Mapping[str, object],
) -> PredictionWeights:
    """optimizerの重み設定をpredictor用へ変換する。"""
    raw_weights = config.get("w", {})

    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    normalized = normalized_weight_dict(raw_weights)
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


def random_weights() -> PredictionWeights:
    """ランダム比較用の重み設定を返す。"""
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


def summary_to_result(
    summary: BacktestSummary,
    *,
    config_name: str,
) -> dict[str, object]:
    """BacktestSummaryをoptimizerの評価結果形式へ変換する。"""
    return {
        "config": config_name,
        "tested_periods": summary.tested_periods,
        "avg_matches": summary.average_best_matches,
        "average_matches_per_ticket": (
            summary.average_matches_per_ticket
        ),
        "hit_rate_1match": summary.hit_rate_1match,
        "hit_rate_2match": summary.hit_rate_2match,
        "hit_rate_3match": summary.hit_rate_3match,
        "hit_rate_4match": summary.hit_rate_4match,
        "hit_rate_5match": summary.hit_rate_5match,
        "hit_rate_6match": summary.hit_rate_6match,
        "hit_rate_7match": summary.hit_rate_7match,
    }


def run_backtest_result(
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
    """バックテストを実行して評価結果形式で返す。"""
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

    return summary_to_result(
        summary,
        config_name=config_name,
    )


def aggregate_seed_results(
    results: Sequence[Mapping[str, object]],
    *,
    config_name: str,
) -> dict[str, object]:
    """複数seedの評価結果を集約する。"""
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
        "tested_periods": int(
            results[0].get("tested_periods") or 0
        ),
        "evaluated_seeds": len(results),
    }

    for key in metric_keys:
        values = [
            float(item.get(key) or 0.0)
            for item in results
        ]
        aggregated[key] = round(float(mean(values)), 6)

    avg_values = [
        float(item.get("avg_matches") or 0.0)
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


def selection_score(
    result: Mapping[str, object],
    random_avg: float | None,
) -> float:
    """
    設定名に依存しない評価関数。

    平均一致数と高一致率を加点し、ランダム比の改善を加点し、
    seed間の不安定さを減点する。
    """
    avg = float(result.get("avg_matches") or 0.0)
    uplift = (
        0.0
        if random_avg is None
        else avg - float(random_avg)
    )
    stability_std = float(
        result.get("avg_matches_std") or 0.0
    )

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

    if (
        random_avg is not None
        and avg < float(random_avg) - 0.05
    ):
        score -= 0.25

    return round(float(score), 6)


def evaluate_config(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
    random_baselines: Mapping[int, Mapping[str, object]],
    filtered_random_baselines: (
        Mapping[int, Mapping[str, object]] | None
    ) = None,
    weights_override: PredictionWeights | None = None,
) -> dict[str, object]:
    """1つのoptimizer設定を指定seed群で評価する。"""
    merged_game_config = merge_config(
        game_config,
        optimizer_config,
    )
    weights = (
        weights_override
        if weights_override is not None
        else prediction_weights(
            optimizer_config
        )
    )
    name = str(optimizer_config["name"])

    per_seed = [
        run_backtest_result(
            history,
            merged_game_config,
            config_name=name,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            weights=weights,
            seed=seed,
        )
        for seed in seeds
    ]

    result = aggregate_seed_results(
        per_seed,
        config_name=name,
    )

    uniform_results = [
        random_baselines[seed]
        for seed in seeds
    ]
    resolved_filtered = (
        filtered_random_baselines
        if filtered_random_baselines is not None
        else random_baselines
    )
    filtered_results = [
        resolved_filtered[seed]
        for seed in seeds
    ]
    uniform_aggregate = aggregate_seed_results(
        uniform_results,
        config_name="uniform_random",
    )
    filtered_aggregate = aggregate_seed_results(
        filtered_results,
        config_name="filtered_random",
    )
    uniform_avg = float(
        uniform_aggregate.get("avg_matches") or 0.0
    )
    filtered_avg = float(
        filtered_aggregate.get("avg_matches") or 0.0
    )

    result["selection_score"] = selection_score(
        result,
        uniform_avg,
    )
    result["random_unfiltered_avg"] = uniform_avg
    result["random_filtered_avg"] = filtered_avg
    result["random_uplift"] = round(
        float(result.get("avg_matches") or 0.0) - uniform_avg,
        6,
    )
    result["random_filtered_baseline"] = filtered_aggregate

    raw_weights = optimizer_config.get("w", {})
    result["weights"] = (
        dict(raw_weights)
        if isinstance(raw_weights, Mapping)
        else {}
    )

    filters = optimizer_config.get("f", {})
    result["filters"] = (
        dict(filters)
        if isinstance(filters, Mapping)
        else {}
    )

    result["search_origin"] = optimizer_config.get(
        "search_origin"
    )
    result["parent"] = optimizer_config.get("parent")

    return result


def _random_summary_to_result(
    summary: BacktestSummary,
    *,
    config_name: str,
) -> dict[str, object]:
    return summary_to_result(
        summary,
        config_name=config_name,
    )


def build_random_baselines(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
) -> dict[int, dict[str, object]]:
    """全組合せから一様抽出する無フィルタ基準を作成する。"""
    return {
        seed: _random_summary_to_result(
            run_uniform_random_backtest(
                history,
                game_config,
                train_window=train_window,
                tested_periods=tested_periods,
                top_k=1,
                seed=seed,
            ),
            config_name="uniform_random",
        )
        for seed in seeds
    }


def build_filtered_random_baselines(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
) -> dict[int, dict[str, object]]:
    """形状フィルタ通過候補から一様抽出する基準を作成する。"""
    return {
        seed: _random_summary_to_result(
            run_filtered_random_backtest(
                history,
                game_config,
                train_window=train_window,
                tested_periods=tested_periods,
                candidate_count=candidate_count,
                top_k=1,
                seed=seed,
            ),
            config_name="filtered_random",
        )
        for seed in seeds
    }


def evaluate_configs(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    configs: Sequence[Mapping[str, object]],
    *,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seeds: Sequence[int],
    random_baselines: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    """複数設定を評価し、selection_score降順で返す。"""
    results = [
        evaluate_config(
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
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )

    return results


def replace_with_robust_results(
    preliminary_results: Sequence[Mapping[str, object]],
    robust_results: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """予備評価結果を複数seedの再評価結果で置き換える。"""
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
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )

    return ranked_results
