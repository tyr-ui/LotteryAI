from __future__ import annotations

import math
from random import Random
from typing import Mapping, Sequence

from data_loader import dataframe_to_history
from features import build_model_context, build_shape_features
from games import LOTTO_GAMES
from optimizer_evaluation import (
    aggregate_seed_results,
    build_random_baselines,
    evaluate_config,
    evaluate_configs,
    merge_config,
    prediction_weights,
    replace_with_robust_results,
)
from optimizer_search import (
    BASE_CONFIGS,
    CONFIGS,
    build_base_candidates,
    deduplicate_configs,
    find_parent_configs,
    generate_local_candidates,
    generate_random_candidates,
)

from optimizer_ablation import (
    run_feature_ablation,
)

from feature_memory import (
    save_feature_memory,
)

from optimizer_learning import (
    apply_learning_weights,
    load_learning_strength,
    load_learning_weights,
)

from predictor import PredictionResult, predict


SEED = 2025

# GitHub Actionsの実行時間が厳しい場合は、
# RANDOM_SEARCH_COUNTとLOCAL_SEARCH_COUNTを減らす。
RANDOM_SEARCH_COUNT = 4
LOCAL_SEARCH_COUNT = 6
PARENT_COUNT = 3
ROBUST_FINALIST_COUNT = 4
ROBUST_SEEDS = (SEED, SEED + 1, SEED + 2)


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
            int(game_config["min_num"]) == int(min_num)
            and int(game_config["max_num"]) == int(max_num)
            and int(game_config["pick_count"])
            == int(pick_count)
            and configured_main_cols
            == normalized_main_cols
        ):
            return dict(game_config)

    for game_config in LOTTO_GAMES.values():
        if (
            int(game_config["min_num"]) == int(min_num)
            and int(game_config["max_num"]) == int(max_num)
            and int(game_config["pick_count"])
            == int(pick_count)
        ):
            return dict(game_config)

    raise ValueError(
        "Could not resolve lottery configuration: "
        f"min_num={min_num}, max_num={max_num}, "
        f"pick_count={pick_count}, "
        f"main_cols={list(main_cols)}"
    )


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
            "numbers": list(item.candidate),
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
    複数seedで最終候補の安定性を確認してから
    次回候補を生成する。

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
        "backtest_candidates": int(
            bt_candidates
        ),
        "final_candidates": int(
            final_candidates
        ),
    })

    learning_weights = load_learning_weights(
        str(game_config["kind"])
    )

    history = dataframe_to_history(
        df,
        game_config,
    )

    random_baselines = build_random_baselines(
        history,
        game_config,
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
        seeds=ROBUST_SEEDS,
    )

    rng = Random(
        SEED
        + int(max_num) * 100
        + int(pick_count)
    )

    base_candidates = build_base_candidates()
    inherited_filters = dict(
        base_candidates[0].get("f", {})
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

    stage_one_configs = deduplicate_configs([
        *base_candidates,
        *random_candidates,
    ])

    stage_one_results = evaluate_configs(
        history,
        game_config,
        stage_one_configs,
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
        seeds=(SEED,),
        random_baselines=random_baselines,
    )

    parent_configs = find_parent_configs(
        stage_one_results,
        stage_one_configs,
        parent_count=PARENT_COUNT,
    )

    local_candidates = generate_local_candidates(
        parent_configs,
        count=LOCAL_SEARCH_COUNT,
        rng=rng,
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

    local_results = evaluate_configs(
        history,
        game_config,
        unevaluated_configs,
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
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
        name: evaluate_config(
            history,
            game_config,
            all_config_by_name[name],
            train_window=int(train_window),
            tested_periods=int(
                tested_periods
            ),
            candidate_count=int(
                bt_candidates
            ),
            seeds=ROBUST_SEEDS,
            random_baselines=(
                random_baselines
            ),
        )
        for name in finalist_names
    }

    ranked_results = (
        replace_with_robust_results(
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
    best_name = str(best_result["config"])
    best_config = all_config_by_name[
        best_name
    ]

    ablation_results = run_feature_ablation(
        history,
        game_config,
        best_config,
        best_result,
        train_window=int(train_window),
        tested_periods=int(tested_periods),
        candidate_count=int(bt_candidates),
        seeds=ROBUST_SEEDS,
        random_baselines=random_baselines,
    )

    save_feature_memory(
        str(game_config["kind"]),
        ablation_results,
    )

    final_config = merge_config(
        game_config,
        best_config,
    )

    base_weights = prediction_weights(
        best_config
    )

    learning_strength = load_learning_strength(
        str(game_config["kind"])
    )

    final_weights = apply_learning_weights(
        base_weights,
        learning_weights,
        strength=learning_strength,
    )

    final_context = build_model_context(
        history,
        final_config,
    )
    
    final_prediction = predict(
        final_context,
        final_config,
        candidate_count=int(
            final_candidates
        ),
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
            aggregate_seed_results(
                [
                    random_baselines[seed]
                    for seed in ROBUST_SEEDS
                ],
                config_name="random",
            )
        ),
        (
            "selected_random_filtered_"
            "baseline"
        ): selected_random_baseline,
        "ranked_configs": ranked_results,
        "selected_config": best_name,
        "selected_weights": dict(
            best_config["w"]
        ),
        "selected_filters": dict(
            best_config.get("f", {})
        ),
        "search_metadata": {
            "algorithm": (
                "fixed_random_local_robust"
            ),
            "learning_weights_loaded": (
                learning_weights
            ),
            "learning_applied": True,
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
            "note": (
                "Optimizer-specific max_block/"
                "max_first/max_con/max_common "
                "are retained for output "
                "compatibility, but "
                "predictor.py currently does "
                "not consume those keys. "
                "Therefore this version "
                "optimizes only effective "
                "prediction weights."
            ),
        },
        "feature_ablation": (
            ablation_results
        ),
        "prediction": prediction,
    }


__all__ = [
    "BASE_CONFIGS",
    "CONFIGS",
    "optimize",
]
