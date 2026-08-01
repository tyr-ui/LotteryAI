from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


NumberRow = tuple[int, ...]
DEFAULT_RECENT_WINDOWS = (5, 10, 20, 50)


@dataclass(frozen=True, slots=True)
class DistributionStats:
    minimum: float
    maximum: float
    average: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class NumbersShapeFeatures:
    digits: NumberRow
    digit_count: int
    sum_value: int
    average_value: float
    min_digit: int
    max_digit: int
    span: int
    odd_count: int
    even_count: int
    high_count: int
    low_count: int
    unique_count: int
    duplicate_count: int
    max_multiplicity: int
    duplicate_pattern: str
    ascending_adjacent_count: int
    descending_adjacent_count: int
    equal_adjacent_count: int
    consecutive_pair_count: int
    is_full_ascending_run: bool
    is_full_descending_run: bool
    is_palindrome: bool
    adjacent_differences: tuple[int, ...]
    adjacent_difference_sum: int
    adjacent_difference_average: float
    adjacent_difference_max: int
    sum_mod_2: int
    sum_mod_3: int
    sum_mod_5: int
    entropy: float


@dataclass(frozen=True, slots=True)
class NumbersModelContext:
    digit_count: int
    digit_min: int
    digit_max: int
    history_size: int
    recent_windows: tuple[int, ...]
    position_frequency: tuple[Mapping[int, float], ...]
    recent_position_frequency: Mapping[
        int,
        tuple[Mapping[int, float], ...],
    ]
    position_delay: tuple[Mapping[int, int], ...]
    overall_frequency: Mapping[int, float]
    recent_overall_frequency: Mapping[
        int,
        Mapping[int, float],
    ]
    ordered_pair_frequency: Mapping[
        tuple[int, int, int, int],
        float,
    ]
    ordered_triplet_frequency: Mapping[
        tuple[int, int, int, int],
        float,
    ]
    duplicate_pattern_frequency: Mapping[str, float]
    sum_distribution: DistributionStats
    odd_distribution: DistributionStats
    high_distribution: DistributionStats
    adjacent_difference_distribution: DistributionStats
    exact_repeat_distribution: DistributionStats
    unordered_repeat_distribution: DistributionStats
    latest_draw: NumberRow


def _config_value(
    config: Mapping[str, object] | object,
    name: str,
    default: object,
) -> object:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _normalize_history(
    history: Iterable[Sequence[int]],
    *,
    digit_count: int | None,
    digit_min: int,
    digit_max: int,
) -> tuple[NumberRow, ...]:
    rows: list[NumberRow] = []
    resolved_count = digit_count

    for row in history:
        digits = tuple(int(value) for value in row)

        if resolved_count is None:
            resolved_count = len(digits)

        if len(digits) != resolved_count:
            raise ValueError(
                "All Numbers rows must have the same digit count."
            )

        if any(
            digit < digit_min or digit > digit_max
            for digit in digits
        ):
            raise ValueError(
                "Numbers digits are outside the configured range."
            )

        rows.append(digits)

    if not rows:
        raise ValueError(
            "history must contain at least one draw."
        )

    return tuple(rows)


def _distribution_stats(
    values: Sequence[int | float],
) -> DistributionStats:
    if not values:
        return DistributionStats(0.0, 0.0, 0.0, 0.0)

    normalized = [float(value) for value in values]

    return DistributionStats(
        minimum=min(normalized),
        maximum=max(normalized),
        average=float(mean(normalized)),
        standard_deviation=(
            float(pstdev(normalized))
            if len(normalized) > 1
            else 0.0
        ),
    )


def _frequency_mapping(
    values: Sequence[int],
    *,
    digit_min: int,
    digit_max: int,
) -> dict[int, float]:
    counts = Counter(values)
    total = len(values)

    return {
        digit: (
            counts.get(digit, 0) / total
            if total > 0
            else 0.0
        )
        for digit in range(digit_min, digit_max + 1)
    }


def _position_frequency(
    history: Sequence[NumberRow],
    *,
    digit_min: int,
    digit_max: int,
) -> tuple[Mapping[int, float], ...]:
    return tuple(
        _frequency_mapping(
            [row[position] for row in history],
            digit_min=digit_min,
            digit_max=digit_max,
        )
        for position in range(len(history[0]))
    )


def _position_delay(
    history: Sequence[NumberRow],
    *,
    digit_min: int,
    digit_max: int,
) -> tuple[Mapping[int, int], ...]:
    result: list[Mapping[int, int]] = []
    history_size = len(history)

    for position in range(len(history[0])):
        delays: dict[int, int] = {}

        for digit in range(digit_min, digit_max + 1):
            delay = history_size

            for index, row in enumerate(reversed(history)):
                if row[position] == digit:
                    delay = index
                    break

            delays[digit] = delay

        result.append(delays)

    return tuple(result)


def duplicate_pattern(
    digits: Sequence[int],
) -> str:
    counts = sorted(
        Counter(int(value) for value in digits).values(),
        reverse=True,
    )

    known = {
        (1, 1, 1): "all_unique",
        (2, 1): "one_pair",
        (3,): "all_same",
        (1, 1, 1, 1): "all_unique",
        (2, 1, 1): "one_pair",
        (2, 2): "two_pairs",
        (3, 1): "three_same",
        (4,): "all_same",
    }

    return known.get(
        tuple(counts),
        "-".join(str(value) for value in counts),
    )


def _entropy(
    digits: Sequence[int],
) -> float:
    counts = Counter(digits)
    total = len(digits)

    return round(
        -sum(
            (count / total) * log2(count / total)
            for count in counts.values()
        ),
        6,
    )


def build_numbers_shape_features(
    digits: Sequence[int],
    *,
    digit_min: int = 0,
    digit_max: int = 9,
) -> NumbersShapeFeatures:
    normalized = tuple(int(value) for value in digits)

    if not normalized:
        raise ValueError("digits must not be empty.")

    if any(
        digit < digit_min or digit > digit_max
        for digit in normalized
    ):
        raise ValueError(
            "digits contain values outside the configured range."
        )

    digit_count = len(normalized)
    total = sum(normalized)
    minimum = min(normalized)
    maximum = max(normalized)
    counts = Counter(normalized)

    adjacent_differences = tuple(
        abs(normalized[index + 1] - normalized[index])
        for index in range(digit_count - 1)
    )

    ascending = sum(
        normalized[index + 1] - normalized[index] == 1
        for index in range(digit_count - 1)
    )
    descending = sum(
        normalized[index] - normalized[index + 1] == 1
        for index in range(digit_count - 1)
    )
    equal = sum(
        normalized[index] == normalized[index + 1]
        for index in range(digit_count - 1)
    )

    difference_sum = sum(adjacent_differences)
    odd_count = sum(digit % 2 == 1 for digit in normalized)
    high_count = sum(digit >= 5 for digit in normalized)

    return NumbersShapeFeatures(
        digits=normalized,
        digit_count=digit_count,
        sum_value=total,
        average_value=round(total / digit_count, 6),
        min_digit=minimum,
        max_digit=maximum,
        span=maximum - minimum,
        odd_count=odd_count,
        even_count=digit_count - odd_count,
        high_count=high_count,
        low_count=digit_count - high_count,
        unique_count=len(counts),
        duplicate_count=digit_count - len(counts),
        max_multiplicity=max(counts.values()),
        duplicate_pattern=duplicate_pattern(normalized),
        ascending_adjacent_count=ascending,
        descending_adjacent_count=descending,
        equal_adjacent_count=equal,
        consecutive_pair_count=sum(
            difference == 1
            for difference in adjacent_differences
        ),
        is_full_ascending_run=(
            digit_count > 1
            and ascending == digit_count - 1
        ),
        is_full_descending_run=(
            digit_count > 1
            and descending == digit_count - 1
        ),
        is_palindrome=(
            normalized == tuple(reversed(normalized))
        ),
        adjacent_differences=adjacent_differences,
        adjacent_difference_sum=difference_sum,
        adjacent_difference_average=round(
            difference_sum / len(adjacent_differences),
            6,
        )
        if adjacent_differences
        else 0.0,
        adjacent_difference_max=(
            max(adjacent_differences)
            if adjacent_differences
            else 0
        ),
        sum_mod_2=total % 2,
        sum_mod_3=total % 3,
        sum_mod_5=total % 5,
        entropy=_entropy(normalized),
    )


def exact_position_matches(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    if len(left) != len(right):
        raise ValueError(
            "Numbers rows must have equal lengths."
        )

    return sum(
        int(a) == int(b)
        for a, b in zip(left, right)
    )


def unordered_digit_matches(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    left_counts = Counter(int(value) for value in left)
    right_counts = Counter(int(value) for value in right)

    return sum(
        min(left_counts[digit], right_counts[digit])
        for digit in left_counts.keys() | right_counts.keys()
    )


def _ordered_pair_frequency(
    history: Sequence[NumberRow],
) -> dict[
    tuple[int, int, int, int],
    float,
]:
    counts: Counter[
        tuple[int, int, int, int]
    ] = Counter()
    total = 0

    for row in history:
        for left_position in range(
            len(row)
        ):
            for right_position in range(
                left_position + 1,
                len(row),
            ):
                counts[
                    (
                        left_position,
                        right_position,
                        row[left_position],
                        row[right_position],
                    )
                ] += 1
                total += 1

    return {
        key: count / total
        for key, count in counts.items()
    } if total > 0 else {}


def _ordered_triplet_frequency(
    history: Sequence[NumberRow],
) -> dict[tuple[int, int, int, int], float]:
    counts: Counter[tuple[int, int, int, int]] = Counter()
    total = 0

    for row in history:
        for start in range(max(0, len(row) - 2)):
            counts[
                (
                    start,
                    row[start],
                    row[start + 1],
                    row[start + 2],
                )
            ] += 1
            total += 1

    return {
        key: count / total
        for key, count in counts.items()
    } if total > 0 else {}


def _repeat_distributions(
    history: Sequence[NumberRow],
) -> tuple[DistributionStats, DistributionStats]:
    exact_values: list[int] = []
    unordered_values: list[int] = []

    for index in range(1, len(history)):
        previous = history[index - 1]
        current = history[index]

        exact_values.append(
            exact_position_matches(current, previous)
        )
        unordered_values.append(
            unordered_digit_matches(current, previous)
        )

    return (
        _distribution_stats(exact_values),
        _distribution_stats(unordered_values),
    )


def build_numbers_model_context(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
) -> NumbersModelContext:
    configured_count = int(
        _config_value(config, "digit_count", 0) or 0
    )
    digit_min = int(
        _config_value(config, "digit_min", 0) or 0
    )
    digit_max = int(
        _config_value(config, "digit_max", 9) or 9
    )

    configured_windows = _config_value(
        config,
        "recent_windows",
        DEFAULT_RECENT_WINDOWS,
    )

    recent_windows = tuple(
        sorted(
            {
                max(1, int(window))
                for window in configured_windows
            }
        )
    )

    normalized_history = _normalize_history(
        history,
        digit_count=(
            configured_count
            if configured_count > 0
            else None
        ),
        digit_min=digit_min,
        digit_max=digit_max,
    )

    position_frequency = _position_frequency(
        normalized_history,
        digit_min=digit_min,
        digit_max=digit_max,
    )

    recent_position_frequency = {
        window: _position_frequency(
            normalized_history[-window:],
            digit_min=digit_min,
            digit_max=digit_max,
        )
        for window in recent_windows
    }

    flattened = [
        digit
        for row in normalized_history
        for digit in row
    ]

    recent_overall_frequency = {
        window: _frequency_mapping(
            [
                digit
                for row in normalized_history[-window:]
                for digit in row
            ],
            digit_min=digit_min,
            digit_max=digit_max,
        )
        for window in recent_windows
    }

    shapes = [
        build_numbers_shape_features(
            row,
            digit_min=digit_min,
            digit_max=digit_max,
        )
        for row in normalized_history
    ]

    exact_repeat, unordered_repeat = (
        _repeat_distributions(normalized_history)
    )

    pattern_counts = Counter(
        shape.duplicate_pattern
        for shape in shapes
    )
    history_size = len(normalized_history)

    return NumbersModelContext(
        digit_count=len(normalized_history[0]),
        digit_min=digit_min,
        digit_max=digit_max,
        history_size=history_size,
        recent_windows=recent_windows,
        position_frequency=position_frequency,
        recent_position_frequency=recent_position_frequency,
        position_delay=_position_delay(
            normalized_history,
            digit_min=digit_min,
            digit_max=digit_max,
        ),
        overall_frequency=_frequency_mapping(
            flattened,
            digit_min=digit_min,
            digit_max=digit_max,
        ),
        recent_overall_frequency=recent_overall_frequency,
        ordered_pair_frequency=_ordered_pair_frequency(
            normalized_history
        ),
        ordered_triplet_frequency=_ordered_triplet_frequency(
            normalized_history
        ),
        duplicate_pattern_frequency={
            pattern: count / history_size
            for pattern, count in pattern_counts.items()
        },
        sum_distribution=_distribution_stats(
            [shape.sum_value for shape in shapes]
        ),
        odd_distribution=_distribution_stats(
            [shape.odd_count for shape in shapes]
        ),
        high_distribution=_distribution_stats(
            [shape.high_count for shape in shapes]
        ),
        adjacent_difference_distribution=_distribution_stats(
            [
                shape.adjacent_difference_sum
                for shape in shapes
            ]
        ),
        exact_repeat_distribution=exact_repeat,
        unordered_repeat_distribution=unordered_repeat,
        latest_draw=normalized_history[-1],
    )


def standardized_distance(
    value: float,
    stats: DistributionStats,
) -> float:
    if stats.standard_deviation <= 0.0:
        return abs(float(value) - stats.average)

    return abs(
        float(value) - stats.average
    ) / stats.standard_deviation


def gaussian_shape_score(
    value: float,
    stats: DistributionStats,
) -> float:
    if stats.standard_deviation <= 0.0:
        return 1.0 if float(value) == stats.average else 0.0

    z_score = (
        float(value) - stats.average
    ) / stats.standard_deviation

    return round(
        1.0 / (1.0 + z_score * z_score),
        6,
    )


__all__ = [
    "DEFAULT_RECENT_WINDOWS",
    "DistributionStats",
    "NumberRow",
    "NumbersModelContext",
    "NumbersShapeFeatures",
    "build_numbers_model_context",
    "build_numbers_shape_features",
    "duplicate_pattern",
    "exact_position_matches",
    "gaussian_shape_score",
    "standardized_distance",
    "unordered_digit_matches",
]
