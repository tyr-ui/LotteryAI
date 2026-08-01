from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log
from typing import Iterable, Mapping, Sequence

from numbers_features import (
    NumbersModelContext,
    build_numbers_shape_features,
    exact_position_matches,
    gaussian_shape_score,
    unordered_digit_matches,
)


NumberRow = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NumbersPredictionWeights:
    position_frequency: float = 1.20
    recent_position_frequency: float = 1.40
    position_delay: float = 0.70
    overall_frequency: float = 0.40
    ordered_pair: float = 0.80
    ordered_triplet: float = 0.25
    duplicate_pattern: float = 0.60
    sum_shape: float = 0.45
    odd_shape: float = 0.30
    high_shape: float = 0.30
    adjacent_difference: float = 0.35
    exact_repeat: float = 0.55
    unordered_repeat: float = 0.35
    diversity: float = 0.30

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> "NumbersPredictionWeights":
        if values is None:
            return cls()

        aliases = {
            "position": "position_frequency",
            "recent": "recent_position_frequency",
            "delay": "position_delay",
            "overall": "overall_frequency",
            "pair": "ordered_pair",
            "triplet": "ordered_triplet",
            "duplicate": "duplicate_pattern",
            "sum": "sum_shape",
            "odd": "odd_shape",
            "high": "high_shape",
            "adjacent": "adjacent_difference",
            "repeat": "exact_repeat",
            "unordered": "unordered_repeat",
        }

        valid_fields = set(cls.__dataclass_fields__)
        normalized: dict[str, float] = {}

        for key, value in values.items():
            field_name = aliases.get(str(key), str(key))

            if field_name not in valid_fields:
                continue

            try:
                normalized[field_name] = float(value)
            except (TypeError, ValueError):
                continue

        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class NumbersCandidateScore:
    candidate: NumberRow
    number: str
    total_score: float
    components: Mapping[str, float]
    exact_repeat_count: int
    unordered_repeat_count: int


@dataclass(frozen=True, slots=True)
class NumbersPredictionResult:
    selected: tuple[NumbersCandidateScore, ...]
    ranked: tuple[NumbersCandidateScore, ...]
    generated_count: int

    @property
    def numbers(self) -> tuple[NumberRow, ...]:
        return tuple(
            item.candidate
            for item in self.selected
        )


def format_number(
    candidate: Sequence[int],
) -> str:
    return "".join(
        str(int(digit))
        for digit in candidate
    )


def generate_all_candidates(
    context: NumbersModelContext,
) -> tuple[NumberRow, ...]:
    pool = range(
        context.digit_min,
        context.digit_max + 1,
    )

    return tuple(
        tuple(int(value) for value in candidate)
        for candidate in product(
            pool,
            repeat=context.digit_count,
        )
    )


def _safe_log_probability(
    value: float,
) -> float:
    safe = max(float(value), 1e-12)

    return 1.0 / (
        1.0 + abs(log(safe))
    )


def _average_recent_position_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    if not context.recent_windows:
        return 0.0

    window_scores: list[float] = []

    for window in context.recent_windows:
        position_maps = (
            context
            .recent_position_frequency[
                window
            ]
        )

        score = sum(
            float(
                position_maps[position].get(
                    int(digit),
                    0.0,
                )
            )
            for position, digit
            in enumerate(candidate)
        ) / len(candidate)

        window_scores.append(score)

    return (
        sum(window_scores)
        / len(window_scores)
    )


def _position_frequency_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    return sum(
        float(
            context.position_frequency[
                position
            ].get(
                int(digit),
                0.0,
            )
        )
        for position, digit
        in enumerate(candidate)
    ) / len(candidate)


def _position_delay_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    maximum_delay = max(
        1,
        context.history_size,
    )

    normalized_scores = [
        min(
            float(
                context.position_delay[
                    position
                ].get(
                    int(digit),
                    maximum_delay,
                )
            )
            / maximum_delay,
            1.0,
        )
        for position, digit
        in enumerate(candidate)
    ]

    return sum(
        normalized_scores
    ) / len(normalized_scores)


def _overall_frequency_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    return sum(
        float(
            context.overall_frequency.get(
                int(digit),
                0.0,
            )
        )
        for digit in candidate
    ) / len(candidate)


def _ordered_pair_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    values: list[float] = []

    for left_position in range(
        len(candidate)
    ):
        for right_position in range(
            left_position + 1,
            len(candidate),
        ):
            values.append(
                float(
                    context
                    .ordered_pair_frequency
                    .get(
                        (
                            left_position,
                            int(
                                candidate[
                                    left_position
                                ]
                            ),
                            int(
                                candidate[
                                    right_position
                                ]
                            ),
                        ),
                        0.0,
                    )
                )
            )

    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def _ordered_triplet_score(
    candidate: Sequence[int],
    context: NumbersModelContext,
) -> float:
    values: list[float] = []

    for start in range(
        max(
            0,
            len(candidate) - 2,
        )
    ):
        values.append(
            float(
                context
                .ordered_triplet_frequency
                .get(
                    (
                        start,
                        int(candidate[start]),
                        int(candidate[start + 1]),
                        int(candidate[start + 2]),
                    ),
                    0.0,
                )
            )
        )

    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def score_candidate(
    candidate: Sequence[int],
    context: NumbersModelContext,
    *,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
) -> NumbersCandidateScore:
    normalized = tuple(
        int(value)
        for value in candidate
    )

    if len(normalized) != context.digit_count:
        raise ValueError(
            "candidate digit count does not "
            "match context."
        )

    resolved_weights = (
        weights
        if isinstance(
            weights,
            NumbersPredictionWeights,
        )
        else NumbersPredictionWeights.from_mapping(
            weights
        )
    )

    shape = build_numbers_shape_features(
        normalized,
        digit_min=context.digit_min,
        digit_max=context.digit_max,
    )

    position_frequency = (
        _position_frequency_score(
            normalized,
            context,
        )
    )
    recent_position_frequency = (
        _average_recent_position_score(
            normalized,
            context,
        )
    )
    position_delay = (
        _position_delay_score(
            normalized,
            context,
        )
    )
    overall_frequency = (
        _overall_frequency_score(
            normalized,
            context,
        )
    )
    ordered_pair = (
        _ordered_pair_score(
            normalized,
            context,
        )
    )
    ordered_triplet = (
        _ordered_triplet_score(
            normalized,
            context,
        )
    )

    duplicate_pattern = float(
        context
        .duplicate_pattern_frequency
        .get(
            shape.duplicate_pattern,
            0.0,
        )
    )

    sum_shape = gaussian_shape_score(
        shape.sum_value,
        context.sum_distribution,
    )
    odd_shape = gaussian_shape_score(
        shape.odd_count,
        context.odd_distribution,
    )
    high_shape = gaussian_shape_score(
        shape.high_count,
        context.high_distribution,
    )
    adjacent_difference = (
        gaussian_shape_score(
            shape.adjacent_difference_sum,
            context
            .adjacent_difference_distribution,
        )
    )

    exact_repeat_count = (
        exact_position_matches(
            normalized,
            context.latest_draw,
        )
    )
    unordered_repeat_count = (
        unordered_digit_matches(
            normalized,
            context.latest_draw,
        )
    )

    exact_repeat = gaussian_shape_score(
        exact_repeat_count,
        context.exact_repeat_distribution,
    )
    unordered_repeat = (
        gaussian_shape_score(
            unordered_repeat_count,
            context.unordered_repeat_distribution,
        )
    )

    components = {
        "position_frequency": (
            position_frequency
        ),
        "recent_position_frequency": (
            recent_position_frequency
        ),
        "position_delay": position_delay,
        "overall_frequency": (
            overall_frequency
        ),
        "ordered_pair": ordered_pair,
        "ordered_triplet": ordered_triplet,
        "duplicate_pattern": (
            duplicate_pattern
        ),
        "sum_shape": sum_shape,
        "odd_shape": odd_shape,
        "high_shape": high_shape,
        "adjacent_difference": (
            adjacent_difference
        ),
        "exact_repeat": exact_repeat,
        "unordered_repeat": (
            unordered_repeat
        ),
    }

    total_score = (
        resolved_weights.position_frequency
        * position_frequency
        + resolved_weights
        .recent_position_frequency
        * recent_position_frequency
        + resolved_weights.position_delay
        * position_delay
        + resolved_weights.overall_frequency
        * overall_frequency
        + resolved_weights.ordered_pair
        * ordered_pair
        + resolved_weights.ordered_triplet
        * ordered_triplet
        + resolved_weights.duplicate_pattern
        * _safe_log_probability(
            duplicate_pattern
        )
        + resolved_weights.sum_shape
        * sum_shape
        + resolved_weights.odd_shape
        * odd_shape
        + resolved_weights.high_shape
        * high_shape
        + resolved_weights.adjacent_difference
        * adjacent_difference
        + resolved_weights.exact_repeat
        * exact_repeat
        + resolved_weights.unordered_repeat
        * unordered_repeat
    )

    return NumbersCandidateScore(
        candidate=normalized,
        number=format_number(
            normalized
        ),
        total_score=float(
            total_score
        ),
        components=components,
        exact_repeat_count=(
            exact_repeat_count
        ),
        unordered_repeat_count=(
            unordered_repeat_count
        ),
    )


def rank_candidates(
    candidates: Iterable[
        Sequence[int]
    ],
    context: NumbersModelContext,
    *,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
) -> tuple[
    NumbersCandidateScore,
    ...,
]:
    scored = [
        score_candidate(
            candidate,
            context,
            weights=weights,
        )
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: (
            item.total_score,
            item.candidate,
        ),
        reverse=True,
    )

    return tuple(scored)


def candidate_distance(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "candidate lengths must match."
        )

    positional_difference = sum(
        int(left[index])
        != int(right[index])
        for index in range(
            len(left)
        )
    )

    unordered_overlap = (
        unordered_digit_matches(
            left,
            right,
        )
    )

    return (
        positional_difference
        + (
            len(left)
            - unordered_overlap
        )
        * 0.5
    )


def select_diverse(
    ranked: Sequence[
        NumbersCandidateScore
    ],
    *,
    top_k: int,
    diversity_weight: float = 0.30,
) -> tuple[
    NumbersCandidateScore,
    ...,
]:
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if not ranked:
        return ()

    selected: list[
        NumbersCandidateScore
    ] = []
    remaining = list(ranked)

    while (
        remaining
        and len(selected) < top_k
    ):
        best_index = 0
        best_adjusted = float("-inf")

        for index, item in enumerate(
            remaining
        ):
            if not selected:
                adjusted = (
                    item.total_score
                )
            else:
                similarities = [
                    (
                        len(item.candidate)
                        - candidate_distance(
                            item.candidate,
                            chosen.candidate,
                        )
                    )
                    for chosen in selected
                ]

                average_similarity = (
                    sum(similarities)
                    / len(similarities)
                )

                adjusted = (
                    item.total_score
                    - diversity_weight
                    * average_similarity
                )

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index

        selected.append(
            remaining.pop(
                best_index
            )
        )

    selected.sort(
        key=lambda item: (
            item.total_score,
            item.candidate,
        ),
        reverse=True,
    )

    return tuple(selected)


def predict_numbers(
    context: NumbersModelContext,
    *,
    top_k: int = 10,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
) -> NumbersPredictionResult:
    resolved_weights = (
        weights
        if isinstance(
            weights,
            NumbersPredictionWeights,
        )
        else NumbersPredictionWeights.from_mapping(
            weights
        )
    )

    candidates = generate_all_candidates(
        context
    )
    ranked = rank_candidates(
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
        generated_count=len(
            candidates
        ),
    )


__all__ = [
    "NumberRow",
    "NumbersCandidateScore",
    "NumbersPredictionResult",
    "NumbersPredictionWeights",
    "candidate_distance",
    "format_number",
    "generate_all_candidates",
    "predict_numbers",
    "rank_candidates",
    "score_candidate",
    "select_diverse",
]
