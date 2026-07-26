from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import exp
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


NumberRow = tuple[int, ...]
Pair = tuple[int, int]
Triplet = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DistributionStats:
    """整数値の経験分布。"""

    counts: Mapping[int, int]
    probabilities: Mapping[int, float]
    average: float
    standard_deviation: float
    minimum: int
    maximum: int
    sample_size: int

    def probability(self, value: int, *, smoothing: float = 1.0) -> float:
        """
        Laplace smoothing付き確率。

        未観測値が必ず0点になるのを防ぐため、既定で1.0を加算する。
        """
        if smoothing < 0:
            raise ValueError("smoothing must be zero or greater.")

        support_min = min(self.minimum, value)
        support_max = max(self.maximum, value)
        support_size = support_max - support_min + 1
        denominator = self.sample_size + smoothing * support_size
        if denominator <= 0:
            return 0.0
        return (self.counts.get(value, 0) + smoothing) / denominator


@dataclass(frozen=True, slots=True)
class ShapeFeatures:
    """1候補または1抽選回の形状特徴。"""

    total: int
    odd_count: int
    low_count: int
    consecutive_pairs: int
    max_consecutive_run: int
    block_counts: tuple[int, ...]
    span: int


@dataclass(frozen=True, slots=True)
class ModelContext:
    """
    過去履歴から構築した特徴量コンテキスト。

    predictor.py、backtester.py、optimizer.pyから共通利用する。
    """

    pick_count: int
    min_num: int
    max_num: int
    number_pool: tuple[int, ...]
    history_size: int
    recent_windows: tuple[int, ...]
    block_ranges: tuple[tuple[int, int], ...]

    global_frequency: Mapping[int, float]
    recent_frequency: Mapping[int, Mapping[int, float]]
    delay: Mapping[int, int]
    pair_frequency: Mapping[Pair, float]
    triplet_frequency: Mapping[Triplet, float]

    repeat_sets: tuple[frozenset[int], ...]
    repeat_distribution: Mapping[int, DistributionStats]

    sum_distribution: DistributionStats
    odd_distribution: DistributionStats
    low_distribution: DistributionStats
    consecutive_distribution: DistributionStats
    span_distribution: DistributionStats
    block_distributions: tuple[DistributionStats, ...]

    number_score_components: Mapping[int, Mapping[str, float]]


def _as_int(value: object, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc


def _config_value(
    config: Mapping[str, object] | object,
    *names: str,
    default: object | None = None,
) -> object:
    """
    従来dict形式と将来のGameConfig形式の両方から値を取得する。
    """
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
    else:
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
    return default


def normalize_history(
    history: Iterable[Sequence[int]],
    *,
    pick_count: int,
    min_num: int,
    max_num: int,
) -> tuple[NumberRow, ...]:
    """履歴を検証し、組合せ型ロト用の昇順tupleへ正規化する。"""
    normalized: list[NumberRow] = []

    for row_index, row in enumerate(history, start=1):
        numbers = tuple(sorted(_as_int(number, name="number") for number in row))

        if len(numbers) != pick_count:
            raise ValueError(
                f"history row {row_index} must contain {pick_count} numbers; "
                f"received {len(numbers)}."
            )
        if len(set(numbers)) != len(numbers):
            raise ValueError(
                f"history row {row_index} contains duplicate numbers: {numbers}"
            )

        invalid = [
            number
            for number in numbers
            if number < min_num or number > max_num
        ]
        if invalid:
            raise ValueError(
                f"history row {row_index} contains numbers outside "
                f"{min_num}..{max_num}: {invalid}"
            )

        normalized.append(numbers)

    if not normalized:
        raise ValueError("history must contain at least one draw.")

    return tuple(normalized)


def _normalize_counter(
    counter: Counter[int] | Counter[tuple[int, ...]],
    *,
    denominator: float,
) -> dict:
    if denominator <= 0:
        return {key: 0.0 for key in counter}
    return {
        key: count / denominator
        for key, count in counter.items()
    }


def _distribution(values: Sequence[int]) -> DistributionStats:
    if not values:
        raise ValueError("distribution requires at least one value.")

    counts = Counter(values)
    sample_size = len(values)

    return DistributionStats(
        counts=dict(counts),
        probabilities={
            value: count / sample_size
            for value, count in counts.items()
        },
        average=float(mean(values)),
        standard_deviation=float(pstdev(values)) if sample_size > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        sample_size=sample_size,
    )


def count_consecutive_pairs(numbers: Sequence[int]) -> int:
    """隣接差が1の組数を返す。例: 1,2,3は2組。"""
    ordered = sorted(numbers)
    return sum(
        1
        for left, right in zip(ordered, ordered[1:])
        if right - left == 1
    )


def max_consecutive_run(numbers: Sequence[int]) -> int:
    """最長連番長を返す。連番がなくても1を返す。"""
    ordered = sorted(numbers)
    if not ordered:
        return 0

    best = 1
    current = 1
    for left, right in zip(ordered, ordered[1:]):
        if right - left == 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def block_counts(
    numbers: Sequence[int],
    ranges: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """各ブロックに含まれる数字数を返す。"""
    return tuple(
        sum(1 for number in numbers if start <= number <= end)
        for start, end in ranges
    )


def build_shape_features(
    numbers: Sequence[int],
    *,
    min_num: int,
    max_num: int,
    ranges: Sequence[tuple[int, int]],
) -> ShapeFeatures:
    ordered = tuple(sorted(numbers))
    midpoint = (min_num + max_num) / 2

    return ShapeFeatures(
        total=sum(ordered),
        odd_count=sum(number % 2 for number in ordered),
        low_count=sum(number <= midpoint for number in ordered),
        consecutive_pairs=count_consecutive_pairs(ordered),
        max_consecutive_run=max_consecutive_run(ordered),
        block_counts=block_counts(ordered, ranges),
        span=ordered[-1] - ordered[0] if ordered else 0,
    )


def _compute_delay(
    history: Sequence[NumberRow],
    number_pool: Sequence[int],
) -> dict[int, int]:
    """
    最新回から何回出ていないかを返す。

    最新回に出た数字は0、1回前だけに出た数字は1。
    全履歴で未出現なら履歴長。
    """
    delays: dict[int, int] = {}
    reversed_history = tuple(reversed(history))

    for number in number_pool:
        delay = len(history)
        for index, draw in enumerate(reversed_history):
            if number in draw:
                delay = index
                break
        delays[number] = delay

    return delays


def _minmax(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        return {}

    minimum = min(values.values())
    maximum = max(values.values())
    width = maximum - minimum

    if width <= 0:
        return {key: 0.5 for key in values}

    return {
        key: (value - minimum) / width
        for key, value in values.items()
    }


def _delay_preference(delays: Mapping[int, int]) -> dict[int, float]:
    """
    Delayを単純な長期未出現ほど高得点にはせず、
    全数字の平均付近からやや上を頂点とする滑らかな得点へ変換する。
    """
    if not delays:
        return {}

    values = list(delays.values())
    center = mean(values)
    spread = pstdev(values) if len(values) > 1 else 1.0
    spread = max(spread, 1.0)
    preferred = center + 0.35 * spread
    sigma = max(1.25 * spread, 1.0)

    return {
        number: exp(-0.5 * ((delay - preferred) / sigma) ** 2)
        for number, delay in delays.items()
    }


def _build_number_components(
    number_pool: Sequence[int],
    global_frequency: Mapping[int, float],
    recent_frequency: Mapping[int, Mapping[int, float]],
    delay: Mapping[int, int],
) -> dict[int, dict[str, float]]:
    global_scaled = _minmax(global_frequency)
    delay_scaled = _delay_preference(delay)

    result: dict[int, dict[str, float]] = {}
    for number in number_pool:
        components: dict[str, float] = {
            "global_frequency": global_scaled.get(number, 0.0),
            "delay": delay_scaled.get(number, 0.0),
        }

        recent_values: list[float] = []
        for window, frequencies in recent_frequency.items():
            scaled = _minmax(frequencies)
            value = scaled.get(number, 0.0)
            components[f"recent_{window}"] = value
            recent_values.append(value)

        components["recent_frequency"] = (
            sum(recent_values) / len(recent_values)
            if recent_values
            else 0.0
        )
        result[number] = components

    return result


def build_model_context(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    recent_windows: Sequence[int] | None = None,
    repeat_depth: int = 3,
    triplet_history_limit: int | None = 500,
) -> ModelContext:
    """
    LotteryAI v2の全特徴量を一括構築する。

    Args:
        history:
            古い回から新しい回の順に並んだ本数字履歴。
        config:
            現行LOTTO_GAMESのゲーム別dict、または将来のGameConfig。
        recent_windows:
            Recent Frequency期間。省略時はconfigまたは
            (50, 100, 200, 500)。
        repeat_depth:
            前何回までRepeat特徴として保持するか。
        triplet_history_limit:
            Triplet集計対象の最大直近回数。Noneなら全期間。
    """
    pick_count = _as_int(
        _config_value(config, "pick_count", "main_count"),
        name="pick_count",
    )
    min_num = _as_int(
        _config_value(config, "min_num", "min_number"),
        name="min_num",
    )
    max_num = _as_int(
        _config_value(config, "max_num", "max_number"),
        name="max_num",
    )

    raw_ranges = _config_value(config, "block_ranges", default=())
    ranges = tuple(
        (int(start), int(end))
        for start, end in raw_ranges  # type: ignore[misc]
    )
    if not ranges:
        ranges = ((min_num, max_num),)

    configured_windows = _config_value(
        config,
        "recent_windows",
        "default_recent_windows",
        default=(50, 100, 200, 500),
    )
    windows = tuple(
        sorted({
            int(window)
            for window in (
                recent_windows
                if recent_windows is not None
                else configured_windows  # type: ignore[arg-type]
            )
            if int(window) > 0
        })
    )
    if not windows:
        raise ValueError("recent_windows must contain at least one positive value.")
    if repeat_depth <= 0:
        raise ValueError("repeat_depth must be greater than zero.")

    normalized_history = normalize_history(
        history,
        pick_count=pick_count,
        min_num=min_num,
        max_num=max_num,
    )
    number_pool = tuple(range(min_num, max_num + 1))
    history_size = len(normalized_history)

    global_counter: Counter[int] = Counter()
    pair_counter: Counter[Pair] = Counter()
    triplet_counter: Counter[Triplet] = Counter()

    for draw in normalized_history:
        global_counter.update(draw)
        pair_counter.update(combinations(draw, 2))

    triplet_history = (
        normalized_history
        if triplet_history_limit is None
        else normalized_history[-max(1, triplet_history_limit):]
    )
    for draw in triplet_history:
        triplet_counter.update(combinations(draw, 3))

    global_frequency = {
        number: global_counter[number] / history_size
        for number in number_pool
    }

    recent_frequency: dict[int, dict[int, float]] = {}
    for window in windows:
        selected = normalized_history[-min(window, history_size):]
        counter = Counter(
            number
            for draw in selected
            for number in draw
        )
        denominator = len(selected)
        recent_frequency[window] = {
            number: counter[number] / denominator
            for number in number_pool
        }

    pair_denominator = history_size
    pair_frequency = _normalize_counter(
        pair_counter,
        denominator=pair_denominator,
    )

    triplet_denominator = len(triplet_history)
    triplet_frequency = _normalize_counter(
        triplet_counter,
        denominator=triplet_denominator,
    )

    delay = _compute_delay(normalized_history, number_pool)

    repeat_sets = tuple(
        frozenset(draw)
        for draw in reversed(normalized_history[-repeat_depth:])
    )

    repeat_values_by_depth: dict[int, list[int]] = {
        depth: []
        for depth in range(1, repeat_depth + 1)
    }
    for current_index in range(history_size):
        current = set(normalized_history[current_index])
        for depth in range(1, repeat_depth + 1):
            previous_index = current_index - depth
            if previous_index < 0:
                continue
            repeat_values_by_depth[depth].append(
                len(current.intersection(normalized_history[previous_index]))
            )

    repeat_distribution = {
        depth: _distribution(values)
        for depth, values in repeat_values_by_depth.items()
        if values
    }

    shapes = [
        build_shape_features(
            draw,
            min_num=min_num,
            max_num=max_num,
            ranges=ranges,
        )
        for draw in normalized_history
    ]

    sum_distribution = _distribution([shape.total for shape in shapes])
    odd_distribution = _distribution([shape.odd_count for shape in shapes])
    low_distribution = _distribution([shape.low_count for shape in shapes])
    consecutive_distribution = _distribution(
        [shape.consecutive_pairs for shape in shapes]
    )
    span_distribution = _distribution([shape.span for shape in shapes])

    block_distributions = tuple(
        _distribution([
            shape.block_counts[index]
            for shape in shapes
        ])
        for index in range(len(ranges))
    )

    number_score_components = _build_number_components(
        number_pool,
        global_frequency,
        recent_frequency,
        delay,
    )

    return ModelContext(
        pick_count=pick_count,
        min_num=min_num,
        max_num=max_num,
        number_pool=number_pool,
        history_size=history_size,
        recent_windows=windows,
        block_ranges=ranges,
        global_frequency=global_frequency,
        recent_frequency=recent_frequency,
        delay=delay,
        pair_frequency=pair_frequency,
        triplet_frequency=triplet_frequency,
        repeat_sets=repeat_sets,
        repeat_distribution=repeat_distribution,
        sum_distribution=sum_distribution,
        odd_distribution=odd_distribution,
        low_distribution=low_distribution,
        consecutive_distribution=consecutive_distribution,
        span_distribution=span_distribution,
        block_distributions=block_distributions,
        number_score_components=number_score_components,
    )


def candidate_repeat_counts(
    candidate: Sequence[int],
    context: ModelContext,
) -> tuple[int, ...]:
    """前回、前々回、3回前…との一致数を順番に返す。"""
    candidate_set = set(candidate)
    return tuple(
        len(candidate_set.intersection(previous))
        for previous in context.repeat_sets
    )


def candidate_pair_score(
    candidate: Sequence[int],
    context: ModelContext,
) -> float:
    """候補内Pair頻度の平均。"""
    pairs = tuple(combinations(sorted(candidate), 2))
    if not pairs:
        return 0.0
    return sum(
        context.pair_frequency.get(pair, 0.0)
        for pair in pairs
    ) / len(pairs)


def candidate_triplet_score(
    candidate: Sequence[int],
    context: ModelContext,
) -> float:
    """候補内Triplet頻度の平均。"""
    triplets = tuple(combinations(sorted(candidate), 3))
    if not triplets:
        return 0.0
    return sum(
        context.triplet_frequency.get(triplet, 0.0)
        for triplet in triplets
    ) / len(triplets)


def candidate_shape_likelihood(
    candidate: Sequence[int],
    context: ModelContext,
    *,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """
    候補形状の経験分布上の尤度を要素別に返す。
    """
    shape = build_shape_features(
        candidate,
        min_num=context.min_num,
        max_num=context.max_num,
        ranges=context.block_ranges,
    )

    result = {
        "sum": context.sum_distribution.probability(
            shape.total,
            smoothing=smoothing,
        ),
        "odd": context.odd_distribution.probability(
            shape.odd_count,
            smoothing=smoothing,
        ),
        "low": context.low_distribution.probability(
            shape.low_count,
            smoothing=smoothing,
        ),
        "consecutive": context.consecutive_distribution.probability(
            shape.consecutive_pairs,
            smoothing=smoothing,
        ),
        "span": context.span_distribution.probability(
            shape.span,
            smoothing=smoothing,
        ),
    }

    block_values = [
        distribution.probability(count, smoothing=smoothing)
        for distribution, count in zip(
            context.block_distributions,
            shape.block_counts,
        )
    ]
    result["blocks"] = (
        sum(block_values) / len(block_values)
        if block_values
        else 0.0
    )

    return result


def candidate_repeat_likelihood(
    candidate: Sequence[int],
    context: ModelContext,
    *,
    smoothing: float = 1.0,
) -> float:
    """前数回との一致数が過去分布上どの程度自然かを返す。"""
    counts = candidate_repeat_counts(candidate, context)
    probabilities: list[float] = []

    for depth, count in enumerate(counts, start=1):
        distribution = context.repeat_distribution.get(depth)
        if distribution is None:
            continue
        probabilities.append(
            distribution.probability(count, smoothing=smoothing)
        )

    return (
        sum(probabilities) / len(probabilities)
        if probabilities
        else 0.0
    )


def validate_candidate(
    candidate: Sequence[int],
    context: ModelContext,
) -> NumberRow:
    """候補を検証し、昇順tupleとして返す。"""
    normalized = tuple(sorted(int(number) for number in candidate))

    if len(normalized) != context.pick_count:
        raise ValueError(
            f"candidate must contain {context.pick_count} numbers."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate must not contain duplicate numbers.")
    if normalized[0] < context.min_num or normalized[-1] > context.max_num:
        raise ValueError(
            f"candidate numbers must be within "
            f"{context.min_num}..{context.max_num}."
        )

    return normalized
