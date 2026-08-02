from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from numbers_features import NumbersModelContext
from numbers_predictor import (
    NumberRow,
    NumbersCandidateScore,
    NumbersPredictionResult,
    NumbersPredictionWeights,
    generate_all_candidates,
    score_candidate,
    select_diverse,
)


NORMALIZATION_VERSION = "rank_percentile_v2"

COMPONENT_NAMES = (
    "position_frequency",
    "recent_position_frequency",
    "position_delay",
    "overall_frequency",
    "ordered_pair",
    "ordered_triplet",
    "duplicate_pattern",
    "sum_shape",
    "odd_shape",
    "high_shape",
    "adjacent_difference",
    "exact_repeat",
    "unordered_repeat",
)


def _average_percentile_ranks(
    values: Sequence[float],
) -> tuple[float, ...]:
    """
    候補集合内で各値を0.0〜1.0の平均パーセンタイル順位へ変換する。

    同値には同じ平均順位を与えるため、離散特徴量でも候補間の
    順序を壊さず、最大値割りのような1.0付近への密集を避けられる。
    """
    count = len(values)

    if count == 0:
        return ()
    if count == 1:
        return (0.5,)

    ordered = sorted(
        enumerate(float(value) for value in values),
        key=lambda item: item[1],
    )
    result = [0.0] * count
    index = 0

    while index < count:
        end = index + 1
        current_value = ordered[index][1]

        while (
            end < count
            and ordered[end][1] == current_value
        ):
            end += 1

        average_rank = (
            index + end - 1
        ) / 2.0
        percentile = average_rank / (
            count - 1
        )

        for ranked_index in range(
            index,
            end,
        ):
            original_index = ordered[
                ranked_index
            ][0]
            result[original_index] = float(
                percentile
            )

        index = end

    return tuple(result)


def _resolve_weights(
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ),
) -> NumbersPredictionWeights:
    if isinstance(
        weights,
        NumbersPredictionWeights,
    ):
        return weights

    return NumbersPredictionWeights.from_mapping(
        weights
    )


def rank_candidates_rank_v2(
    candidates: Iterable[Sequence[int]],
    context: NumbersModelContext,
    *,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
) -> tuple[NumbersCandidateScore, ...]:
    resolved_weights = _resolve_weights(
        weights
    )

    raw_scores = tuple(
        score_candidate(
            candidate,
            context,
            weights=resolved_weights,
        )
        for candidate in candidates
    )

    if not raw_scores:
        return ()

    normalized_columns = {
        component_name: (
            _average_percentile_ranks(
                tuple(
                    float(
                        item.components[
                            component_name
                        ]
                    )
                    for item in raw_scores
                )
            )
        )
        for component_name in COMPONENT_NAMES
    }

    normalized_scores: list[
        NumbersCandidateScore
    ] = []

    for item_index, item in enumerate(
        raw_scores
    ):
        components = {
            component_name: float(
                normalized_columns[
                    component_name
                ][item_index]
            )
            for component_name in COMPONENT_NAMES
        }

        total_score = sum(
            float(
                getattr(
                    resolved_weights,
                    component_name,
                )
            )
            * components[component_name]
            for component_name in COMPONENT_NAMES
        )

        normalized_scores.append(
            replace(
                item,
                total_score=float(
                    total_score
                ),
                components=components,
            )
        )

    normalized_scores.sort(
        key=lambda item: (
            item.total_score,
            item.candidate,
        ),
        reverse=True,
    )

    return tuple(normalized_scores)


def predict_numbers_rank_v2(
    context: NumbersModelContext,
    *,
    top_k: int = 10,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
) -> NumbersPredictionResult:
    resolved_weights = _resolve_weights(
        weights
    )
    candidates = generate_all_candidates(
        context
    )
    ranked = rank_candidates_rank_v2(
        candidates,
        context,
        weights=resolved_weights,
    )
    selected = select_diverse(
        ranked,
        top_k=top_k,
        diversity_weight=(
            resolved_weights.diversity
        ),
    )

    return NumbersPredictionResult(
        selected=selected,
        ranked=ranked,
        generated_count=len(candidates),
    )


# numbers_backtester / numbers_optimizer から差し替えて使える共通名。
predict_numbers = predict_numbers_rank_v2


__all__ = [
    "COMPONENT_NAMES",
    "NORMALIZATION_VERSION",
    "predict_numbers",
    "predict_numbers_rank_v2",
    "rank_candidates_rank_v2",
]
