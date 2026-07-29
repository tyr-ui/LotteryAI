from __future__ import annotations

from typing import Mapping, Sequence

from optimizer_evaluation import evaluate_config


ABLATION_FEATURES = (
    "freq",
    "recent",
    "pair",
    "triplet",
    "delay",
    "dist",
    "repeat",
)


def _build_ablated_config(
    optimizer_config: Mapping[str, object],
    *,
    feature: str,
) -> dict[str, object]:
    """
    指定した特徴量の重みだけを0にした設定を作成する。

    元の設定は変更しない。
    他の重みはevaluate_config内のprediction_weightsで
    再正規化される。
    """
    if feature not in ABLATION_FEATURES:
        raise ValueError(
            f"Unsupported ablation feature: {feature}"
        )

    original_name = str(
        optimizer_config.get(
            "name",
            "unknown",
        )
    )

    raw_weights = optimizer_config.get(
        "w",
        {},
    )

    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    ablated_weights = dict(raw_weights)
    ablated_weights[feature] = 0.0

    filters = optimizer_config.get(
        "f",
        {},
    )

    if not isinstance(filters, Mapping):
        filters = {}

    scoring = optimizer_config.get(
        "s",
        {},
    )

    if not isinstance(scoring, Mapping):
        scoring = {}

    return {
        "name": (
            f"{original_name}__without_"
            f"{feature}"
        ),
        "w": ablated_weights,
        "s": dict(scoring),
        "f": dict(filters),
        "search_origin": "ablation",
        "parent": original_name,
        "ablated_feature": feature,
    }


def _calculate_drop(
    baseline: float,
    ablated: float,
) -> tuple[float, float]:
    """
    通常結果からアブレーション結果を引き、
    絶対差と低下率を返す。

    正の値:
        特徴量を無効化すると性能が低下した。
        その特徴量が有効である可能性が高い。

    負の値:
        特徴量を無効化すると性能が改善した。
        その特徴量が悪影響を与えている可能性がある。
    """
    drop = baseline - ablated

    if baseline == 0.0:
        drop_percent = 0.0
    else:
        drop_percent = (
            drop / abs(baseline)
        ) * 100.0

    return (
        round(drop, 6),
        round(drop_percent, 6),
    )


def run_feature_ablation(
    history: Sequence[Sequence[int]],
    game_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    baseline_result: Mapping[str, object],
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
    """
    選択済み設定について、特徴量を1つずつ無効化して
    バックテストを実行する。

    baseline_resultには、通常状態で複数seed評価された
    best_resultを渡す。

    結果はavg_matchesの低下量が大きい順に返す。
    """
    if not seeds:
        raise ValueError(
            "seeds must not be empty."
        )

    baseline_avg = float(
        baseline_result.get(
            "avg_matches",
            0.0,
        )
        or 0.0
    )
    baseline_score = float(
        baseline_result.get(
            "selection_score",
            0.0,
        )
        or 0.0
    )

    results: list[dict[str, object]] = []

    for feature in ABLATION_FEATURES:
        ablated_config = (
            _build_ablated_config(
                optimizer_config,
                feature=feature,
            )
        )

        ablated_result = evaluate_config(
            history,
            game_config,
            ablated_config,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            seeds=seeds,
            random_baselines=(
                random_baselines
            ),
        )

        ablated_avg = float(
            ablated_result.get(
                "avg_matches",
                0.0,
            )
            or 0.0
        )
        ablated_score = float(
            ablated_result.get(
                "selection_score",
                0.0,
            )
            or 0.0
        )

        avg_drop, avg_drop_percent = (
            _calculate_drop(
                baseline_avg,
                ablated_avg,
            )
        )
        score_drop, score_drop_percent = (
            _calculate_drop(
                baseline_score,
                ablated_score,
            )
        )

        results.append({
            "feature": feature,
            "baseline_config": str(
                optimizer_config.get(
                    "name",
                    "unknown",
                )
            ),
            "ablated_config": str(
                ablated_config["name"]
            ),
            "baseline_avg_matches": round(
                baseline_avg,
                6,
            ),
            "ablated_avg_matches": round(
                ablated_avg,
                6,
            ),
            "avg_matches_drop": avg_drop,
            "avg_matches_drop_percent": (
                avg_drop_percent
            ),
            "baseline_selection_score": (
                round(
                    baseline_score,
                    6,
                )
            ),
            "ablated_selection_score": (
                round(
                    ablated_score,
                    6,
                )
            ),
            "selection_score_drop": (
                score_drop
            ),
            (
                "selection_score_"
                "drop_percent"
            ): score_drop_percent,
            "ablated_weights": dict(
                ablated_result.get(
                    "weights",
                    {},
                )
            ),
            "tested_periods": int(
                ablated_result.get(
                    "tested_periods",
                    0,
                )
                or 0
            ),
            "evaluated_seeds": int(
                ablated_result.get(
                    "evaluated_seeds",
                    len(seeds),
                )
                or len(seeds)
            ),
        })

    results.sort(
        key=lambda item: (
            float(
                item[
                    "avg_matches_drop"
                ]
            ),
            float(
                item[
                    "selection_score_drop"
                ]
            ),
        ),
        reverse=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

    return results


__all__ = [
    "ABLATION_FEATURES",
    "run_feature_ablation",
]