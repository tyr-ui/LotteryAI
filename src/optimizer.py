from __future__ import annotations

import math
from typing import Mapping, Sequence

from backtester import BacktestSummary, run_backtest
from data_loader import dataframe_to_history
from features import build_model_context, build_shape_features
from games import LOTTO_GAMES
from predictor import PredictionResult, PredictionWeights, predict


SEED = 2025


# 自動探索する設定。
#
# 旧optimizer.pyとの出力互換性を維持するため、
# 設定名・重み・フィルタ情報は従来形式のまま保持する。
CONFIGS = [
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
        "s": {
            "g": 0.35,
            "r": 0.40,
            "d": 0.25,
        },
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
        "s": {
            "g": 0.35,
            "r": 0.40,
            "d": 0.25,
        },
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
        "s": {
            "g": 0.45,
            "r": 0.55,
            "d": 0.00,
        },
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
        "s": {
            "g": 0.45,
            "r": 0.55,
            "d": 0.00,
        },
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
        "s": {
            "g": 0.65,
            "r": 0.35,
            "d": 0.00,
        },
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
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
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
        "s": {
            "g": 0.30,
            "r": 0.35,
            "d": 0.35,
        },
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
        "s": {
            "g": 0.40,
            "r": 0.45,
            "d": 0.15,
        },
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
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
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
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
]


def _resolve_game_config(
    main_cols: Sequence[str],
    min_num: int,
    max_num: int,
    pick_count: int,
) -> dict[str, object]:
    """
    旧optimize()の引数から対応するゲーム設定を取得する。

    main.pyを同時変更しなくても動かせるよう、
    現在のoptimize()呼び出し形式との互換性を維持する。
    """
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
        f"min_num={min_num}, "
        f"max_num={max_num}, "
        f"pick_count={pick_count}, "
        f"main_cols={list(main_cols)}"
    )


def _merge_config(
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    ゲーム固有設定とoptimizer探索設定を統合する。

    現行predictor.pyが直接利用しない旧フィルタ値も、
    出力互換性と将来対応のため設定内に保持する。
    """
    merged = dict(game_config)

    if optimizer_config is None:
        return merged

    filters = optimizer_config.get("f", {})
    if isinstance(filters, Mapping):
        merged.update(filters)

    return merged


def _prediction_weights(
    config: Mapping[str, object],
) -> PredictionWeights:
    """
    旧optimizer形式の重みを新predictor形式へ変換する。
    """
    raw_weights = config.get("w", {})
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    distribution_weight = float(raw_weights.get("dist", 0.0))
    shape_weight = distribution_weight / 6.0

    return PredictionWeights(
        global_frequency=float(raw_weights.get("freq", 0.0)),
        recent_frequency=float(raw_weights.get("recent", 0.0)),
        delay=float(raw_weights.get("delay", 0.0)),
        pair=float(raw_weights.get("pair", 0.0)),
        triplet=float(raw_weights.get("triplet", 0.0)),
        repeat=float(raw_weights.get("repeat", 0.0)),
        sum_shape=shape_weight,
        odd_shape=shape_weight,
        low_shape=shape_weight,
        consecutive_shape=shape_weight,
        span_shape=shape_weight,
        block_shape=shape_weight,
        diversity=0.35,
    )


def _random_weights() -> PredictionWeights:
    """
    数字ごとの生成重みと候補スコアを均一化する。

    predictor.pyの既存APIを変更せず、ランダム基準を作るために使う。
    """
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
    """
    新backtesterの結果を旧optimizer出力形式へ変換する。
    """
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


def _prediction_to_legacy(
    prediction: PredictionResult,
    *,
    context,
    model_name: str,
) -> list[dict[str, object]]:
    """
    新predictorのPredictionResultを既存JSON形式へ変換する。
    """
    estimated_probability = (
        1 / math.comb(context.max_num, context.pick_count)
    )

    converted: list[dict[str, object]] = []

    for index, item in enumerate(prediction.selected, start=1):
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

        converted.append(
            {
                "pattern_id": f"P{index}",
                "numbers": list(item.candidate),
                "score": round(float(item.total_score), 6),
                "model": model_name,
                "block_counts": list(shape.block_counts),
                "consecutive_count": int(
                    shape.consecutive_count
                ),
                "repeat_count": repeat_count,
                "estimated_probability": (
                    estimated_probability
                ),
            }
        )

    return converted


def filter_key(
    config: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    filters = config.get("f", {})

    if not isinstance(filters, Mapping):
        return ()

    return tuple(
        sorted(
            (str(key), value)
            for key, value in filters.items()
        )
    )


def selection_score(
    result: Mapping[str, object],
    random_avg: float | None,
) -> float:
    avg = float(result.get("avg_matches") or 0.0)
    hit_rate_2 = float(
        result.get("hit_rate_2match") or 0.0
    )
    hit_rate_3 = float(
        result.get("hit_rate_3match") or 0.0
    )
    hit_rate_4 = float(
        result.get("hit_rate_4match") or 0.0
    )

    uplift = (
        0.0
        if random_avg is None
        else avg - float(random_avg)
    )

    score = (
        avg
        + 0.30 * hit_rate_2
        + 0.80 * hit_rate_3
        + 1.20 * hit_rate_4
        + 0.35 * uplift
    )

    name = str(result.get("config", ""))

    if "no_delay" in name or "balanced" in name:
        score += 0.015

    if "loose" in name:
        score -= 0.010

    if (
        random_avg is not None
        and avg < float(random_avg) - 0.05
    ):
        score -= 0.25

    return round(float(score), 6)


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
    複数の重み設定をバックテストし、
    最良設定を使って次回候補を生成する。

    引数と戻り値は旧main.pyとの互換性を維持する。
    """
    game_config = _resolve_game_config(
        main_cols,
        min_num,
        max_num,
        pick_count,
    )

    game_config.update(
        {
            "main_cols": tuple(main_cols),
            "min_num": int(min_num),
            "max_num": int(max_num),
            "pick_count": int(pick_count),
            "train_window": int(train_window),
            "tested_periods": int(tested_periods),
            "backtest_candidates": int(bt_candidates),
            "final_candidates": int(final_candidates),
        }
    )

    history = dataframe_to_history(
        df,
        game_config,
    )

    random_weights = _random_weights()

    # 互換キー名はrandom_unfilteredだが、
    # 現在のpredictorが持つゲーム共通フィルタは適用される。
    random_unfiltered_result = _run_backtest_result(
        history,
        game_config,
        config_name="random",
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
        weights=random_weights,
        seed=SEED,
    )

    random_filtered_cache: dict[
        tuple[tuple[str, object], ...],
        dict[str, object],
    ] = {}

    for optimizer_config in CONFIGS:
        key = filter_key(optimizer_config)

        if key in random_filtered_cache:
            continue

        merged_config = _merge_config(
            game_config,
            optimizer_config,
        )

        random_filtered_cache[key] = (
            _run_backtest_result(
                history,
                merged_config,
                config_name="random",
                train_window=int(train_window),
                tested_periods=int(tested_periods),
                candidate_count=int(bt_candidates),
                weights=random_weights,
                seed=SEED + 100000,
            )
        )

    results: list[dict[str, object]] = []

    for optimizer_config in CONFIGS:
        key = filter_key(optimizer_config)
        merged_config = _merge_config(
            game_config,
            optimizer_config,
        )
        weights = _prediction_weights(
            optimizer_config,
        )

        result = _run_backtest_result(
            history,
            merged_config,
            config_name=str(optimizer_config["name"]),
            train_window=int(train_window),
            tested_periods=int(tested_periods),
            candidate_count=int(bt_candidates),
            weights=weights,
            seed=SEED,
        )

        random_filtered_result = (
            random_filtered_cache[key]
        )
        random_filtered_avg = (
            random_filtered_result.get("avg_matches")
        )

        resolved_random_avg = (
            float(random_filtered_avg)
            if random_filtered_avg is not None
            else None
        )

        result["selection_score"] = selection_score(
            result,
            resolved_random_avg,
        )

        result["random_unfiltered_avg"] = (
            random_unfiltered_result.get("avg_matches")
        )
        result["random_filtered_avg"] = (
            random_filtered_avg
        )
        result["random_uplift"] = round(
            float(result.get("avg_matches") or 0.0)
            - float(random_filtered_avg or 0.0),
            4,
        )
        result["random_filtered_baseline"] = (
            random_filtered_result
        )
        result["weights"] = dict(
            optimizer_config["w"]
        )
        result["filters"] = dict(
            optimizer_config["f"]
        )

        results.append(result)

    if not results:
        raise RuntimeError(
            "No optimizer configurations were evaluated."
        )

    results.sort(
        key=lambda item: float(
            item["selection_score"]
        ),
        reverse=True,
    )

    best_result = results[0]
    best_name = str(best_result["config"])

    best_config = next(
        config
        for config in CONFIGS
        if config["name"] == best_name
    )

    final_config = _merge_config(
        game_config,
        best_config,
    )
    final_weights = _prediction_weights(
        best_config,
    )

    final_context = build_model_context(
        history,
        final_config,
    )

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

    return {
        "random_baseline": (
            random_unfiltered_result
        ),
        "selected_random_filtered_baseline": (
            best_result["random_filtered_baseline"]
        ),
        "ranked_configs": results,
        "selected_config": best_name,
        "selected_weights": dict(
            best_config["w"]
        ),
        "selected_filters": dict(
            best_config["f"]
        ),
        "prediction": prediction,
    }