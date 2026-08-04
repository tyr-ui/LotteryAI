from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from random import Random
from itertools import combinations_with_replacement
from typing import Iterable, Mapping, Sequence

from numbers_features import (
    NumberRow,
    build_numbers_model_context,
    exact_position_matches,
    unordered_digit_matches,
)
from numbers_predictor import (
    NumbersPredictionWeights,
    predict_numbers,
)


@dataclass(frozen=True, slots=True)
class NumbersBacktestRecord:
    draw_index: int
    actual: NumberRow
    predicted: tuple[NumberRow, ...]
    best_exact_position_matches: int
    average_exact_position_matches: float
    best_unordered_digit_matches: int
    average_unordered_digit_matches: float
    straight_hit: bool
    box_hit: bool
    box_dedicated_predicted: tuple[NumberRow, ...] = ()
    box_dedicated_best_unordered_matches: int = 0
    box_dedicated_average_unordered_matches: float = 0.0
    box_dedicated_hit: bool = False




@dataclass(frozen=True, slots=True)
class NumbersBoxBacktestRecord:
    draw_index: int
    actual: NumberRow
    predicted_boxes: tuple[NumberRow, ...]
    best_unordered_digit_matches: int
    average_unordered_digit_matches: float
    box_hit: bool


@dataclass(frozen=True, slots=True)
class NumbersBoxBacktestSummary:
    tested_periods: int
    digit_count: int
    average_best_unordered_matches: float | None
    average_unordered_matches_per_ticket: float | None
    box_hit_rate: float | None
    records: tuple[NumbersBoxBacktestRecord, ...]

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "evaluation_type": "box_dedicated_walk_forward",
            "tested_periods": self.tested_periods,
            "digit_count": self.digit_count,
            "average_best_unordered_matches": self.average_best_unordered_matches,
            "average_unordered_matches_per_ticket": self.average_unordered_matches_per_ticket,
            "box_hit_rate": self.box_hit_rate,
        }
        if include_records:
            result["records"] = [asdict(record) for record in self.records]
        return result


@dataclass(frozen=True, slots=True)
class NumbersBacktestSummary:
    tested_periods: int
    digit_count: int
    average_best_position_matches: float | None
    average_position_matches_per_ticket: float | None
    average_best_unordered_matches: float | None
    average_unordered_matches_per_ticket: float | None
    straight_hit_rate: float | None
    box_hit_rate: float | None
    hit_rate_1_position: float | None
    hit_rate_2_position: float | None
    hit_rate_3_position: float | None
    hit_rate_4_position: float | None
    selection_score: float | None
    box_dedicated_average_best_unordered_matches: float | None = None
    box_dedicated_average_unordered_matches_per_ticket: float | None = None
    box_dedicated_hit_rate: float | None = None
    records: tuple[NumbersBacktestRecord, ...] = ()

    def to_dict(
        self,
        *,
        include_records: bool = False,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "tested_periods": self.tested_periods,
            "digit_count": self.digit_count,
            "average_best_position_matches": (
                self.average_best_position_matches
            ),
            "average_position_matches_per_ticket": (
                self.average_position_matches_per_ticket
            ),
            "average_best_unordered_matches": (
                self.average_best_unordered_matches
            ),
            "average_unordered_matches_per_ticket": (
                self.average_unordered_matches_per_ticket
            ),
            "straight_hit_rate": self.straight_hit_rate,
            "box_hit_rate": self.box_hit_rate,
            "hit_rate_1_position": self.hit_rate_1_position,
            "hit_rate_2_position": self.hit_rate_2_position,
            "hit_rate_3_position": self.hit_rate_3_position,
            "hit_rate_4_position": self.hit_rate_4_position,
            "selection_score": self.selection_score,
            "box_dedicated_evaluation": {
                "evaluation_type": "box_dedicated_walk_forward",
                "tested_periods": self.tested_periods,
                "digit_count": self.digit_count,
                "average_best_unordered_matches": self.box_dedicated_average_best_unordered_matches,
                "average_unordered_matches_per_ticket": self.box_dedicated_average_unordered_matches_per_ticket,
                "box_hit_rate": self.box_dedicated_hit_rate,
            },
        }

        if include_records:
            result["records"] = [
                asdict(record)
                for record in self.records
            ]

        return result


def _config_value(
    config: Mapping[str, object] | object,
    *names: str,
    default: object | None = None,
) -> object:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
    else:
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)

    return default


def _normalize_history(
    history: Iterable[Sequence[int]],
    *,
    digit_count: int | None = None,
    digit_min: int = 0,
    digit_max: int = 9,
) -> tuple[NumberRow, ...]:
    normalized: list[NumberRow] = []
    resolved_digit_count = digit_count

    for row in history:
        digits = tuple(
            int(value)
            for value in row
        )

        if resolved_digit_count is None:
            resolved_digit_count = len(digits)

        if len(digits) != resolved_digit_count:
            raise ValueError(
                "All Numbers rows must have the same digit count."
            )

        if any(
            digit < digit_min
            or digit > digit_max
            for digit in digits
        ):
            raise ValueError(
                "Numbers digits are outside the configured range."
            )

        normalized.append(digits)

    if not normalized:
        raise ValueError(
            "history must contain at least one draw."
        )

    return tuple(normalized)


def _box_signature(
    digits: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    counts = Counter(
        int(value)
        for value in digits
    )

    return tuple(
        sorted(
            counts.items()
        )
    )


def is_straight_hit(
    predicted: Sequence[int],
    actual: Sequence[int],
) -> bool:
    return tuple(predicted) == tuple(actual)


def is_box_hit(
    predicted: Sequence[int],
    actual: Sequence[int],
) -> bool:
    return _box_signature(
        predicted
    ) == _box_signature(
        actual
    )


def numbers_selection_score(
    *,
    digit_count: int,
    average_best_position_matches: float,
    average_best_unordered_matches: float,
    straight_hit_rate: float,
    box_hit_rate: float,
    hit_rate_2_position: float,
    hit_rate_3_position: float,
    hit_rate_4_position: float,
) -> float:
    """
    Numbers向けの複合評価値。

    完全一致を最重視しつつ、完全一致が少ない期間でも
    学習信号が残るよう、位置一致・数字一致・BOX一致を加点する。
    """
    normalized_position = (
        average_best_position_matches
        / max(1, digit_count)
    )
    normalized_unordered = (
        average_best_unordered_matches
        / max(1, digit_count)
    )

    score = (
        1.80 * normalized_position
        + 1.10 * normalized_unordered
        + 12.00 * straight_hit_rate
        + 4.00 * box_hit_rate
        + 0.60 * hit_rate_2_position
        + 1.20 * hit_rate_3_position
        + 2.00 * hit_rate_4_position
    )

    return round(
        float(score),
        6,
    )


def _empty_summary(
    digit_count: int,
) -> NumbersBacktestSummary:
    return NumbersBacktestSummary(
        tested_periods=0,
        digit_count=digit_count,
        average_best_position_matches=None,
        average_position_matches_per_ticket=None,
        average_best_unordered_matches=None,
        average_unordered_matches_per_ticket=None,
        straight_hit_rate=None,
        box_hit_rate=None,
        hit_rate_1_position=None,
        hit_rate_2_position=None,
        hit_rate_3_position=None,
        hit_rate_4_position=None,
        selection_score=None,
        box_dedicated_average_best_unordered_matches=None,
        box_dedicated_average_unordered_matches_per_ticket=None,
        box_dedicated_hit_rate=None,
        records=(),
    )


def _summarize_numbers_records(
    records: Sequence[NumbersBacktestRecord],
    *,
    digit_count: int,
    include_records: bool,
) -> NumbersBacktestSummary:
    tested = len(records)
    if tested == 0:
        return _empty_summary(digit_count)

    best_position_values = [
        record.best_exact_position_matches
        for record in records
    ]
    average_position_values = [
        record.average_exact_position_matches
        for record in records
    ]
    best_unordered_values = [
        record.best_unordered_digit_matches
        for record in records
    ]
    average_unordered_values = [
        record.average_unordered_digit_matches
        for record in records
    ]

    def position_hit_rate(threshold: int) -> float:
        if threshold > digit_count:
            return 0.0
        return round(
            sum(value >= threshold for value in best_position_values)
            / tested,
            6,
        )

    straight_hit_rate = round(
        sum(record.straight_hit for record in records) / tested,
        6,
    )
    box_hit_rate = round(
        sum(record.box_hit for record in records) / tested,
        6,
    )
    average_best_position_matches = round(
        float(mean(best_position_values)), 6
    )
    average_position_matches_per_ticket = round(
        float(mean(average_position_values)), 6
    )
    average_best_unordered_matches = round(
        float(mean(best_unordered_values)), 6
    )
    average_unordered_matches_per_ticket = round(
        float(mean(average_unordered_values)), 6
    )

    hit_rate_1 = position_hit_rate(1)
    hit_rate_2 = position_hit_rate(2)
    hit_rate_3 = position_hit_rate(3)
    hit_rate_4 = position_hit_rate(4)
    dedicated_best_values = [
        record.box_dedicated_best_unordered_matches
        for record in records
    ]
    dedicated_average_values = [
        record.box_dedicated_average_unordered_matches
        for record in records
    ]
    dedicated_hit_rate = round(
        sum(record.box_dedicated_hit for record in records) / tested,
        6,
    )
    dedicated_average_best = round(
        float(mean(dedicated_best_values)), 6
    )
    dedicated_average_per_ticket = round(
        float(mean(dedicated_average_values)), 6
    )

    selection_score = numbers_selection_score(
        digit_count=digit_count,
        average_best_position_matches=average_best_position_matches,
        average_best_unordered_matches=average_best_unordered_matches,
        straight_hit_rate=straight_hit_rate,
        box_hit_rate=box_hit_rate,
        hit_rate_2_position=hit_rate_2,
        hit_rate_3_position=hit_rate_3,
        hit_rate_4_position=hit_rate_4,
    )

    return NumbersBacktestSummary(
        tested_periods=tested,
        digit_count=digit_count,
        average_best_position_matches=average_best_position_matches,
        average_position_matches_per_ticket=average_position_matches_per_ticket,
        average_best_unordered_matches=average_best_unordered_matches,
        average_unordered_matches_per_ticket=average_unordered_matches_per_ticket,
        straight_hit_rate=straight_hit_rate,
        box_hit_rate=box_hit_rate,
        hit_rate_1_position=hit_rate_1,
        hit_rate_2_position=hit_rate_2,
        hit_rate_3_position=hit_rate_3,
        hit_rate_4_position=hit_rate_4,
        selection_score=selection_score,
        box_dedicated_average_best_unordered_matches=dedicated_average_best,
        box_dedicated_average_unordered_matches_per_ticket=dedicated_average_per_ticket,
        box_dedicated_hit_rate=dedicated_hit_rate,
        records=tuple(records) if include_records else (),
    )


def _integer_to_digits(value: int, digit_count: int) -> NumberRow:
    return tuple(int(character) for character in f"{value:0{digit_count}d}")


def run_numbers_uniform_random_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    top_k: int | None = None,
    seed: int = 2025,
    include_records: bool = False,
) -> NumbersBacktestSummary:
    """000..999/9999の全候補空間から一様に抽出する比較基準。"""
    configured_digit_count = int(
        _config_value(config, "digit_count", default=0) or 0
    )
    normalized_history = _normalize_history(
        history,
        digit_count=configured_digit_count or None,
        digit_min=0,
        digit_max=9,
    )
    digit_count = len(normalized_history[0])
    total_draws = len(normalized_history)
    resolved_train_window = int(
        train_window if train_window is not None
        else _config_value(config, "train_window", default=500)
    )
    resolved_tested_periods = int(
        tested_periods if tested_periods is not None
        else _config_value(config, "tested_periods", default=90)
    )
    resolved_top_k = int(
        top_k if top_k is not None
        else _config_value(config, "top_k", default=10)
    )
    if min(resolved_train_window, resolved_tested_periods, resolved_top_k) <= 0:
        raise ValueError("backtest parameters must be positive.")

    candidate_space_size = 10 ** digit_count
    if resolved_top_k > candidate_space_size:
        raise ValueError("top_k exceeds the Numbers candidate space.")

    start_index = max(
        resolved_train_window,
        total_draws - resolved_tested_periods,
    )
    records: list[NumbersBacktestRecord] = []
    for test_index in range(start_index, total_draws):
        rng = Random(seed + test_index)
        predicted = tuple(
            _integer_to_digits(value, digit_count)
            for value in rng.sample(
                range(candidate_space_size),
                resolved_top_k,
            )
        )
        actual = normalized_history[test_index]
        position_matches = [
            exact_position_matches(candidate, actual)
            for candidate in predicted
        ]
        unordered_matches = [
            unordered_digit_matches(candidate, actual)
            for candidate in predicted
        ]
        records.append(NumbersBacktestRecord(
            draw_index=test_index,
            actual=actual,
            predicted=predicted,
            best_exact_position_matches=max(position_matches),
            average_exact_position_matches=float(mean(position_matches)),
            best_unordered_digit_matches=max(unordered_matches),
            average_unordered_digit_matches=float(mean(unordered_matches)),
            straight_hit=any(is_straight_hit(candidate, actual) for candidate in predicted),
            box_hit=any(is_box_hit(candidate, actual) for candidate in predicted),
        ))

    return _summarize_numbers_records(
        records, digit_count=digit_count, include_records=include_records
    )


def run_numbers_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    top_k: int | None = None,
    weights: (
        NumbersPredictionWeights
        | Mapping[str, object]
        | None
    ) = None,
    include_records: bool = False,
) -> NumbersBacktestSummary:
    """
    Numbers3/4向けウォークフォワード検証。

    各検証回では、その回より前の履歴だけを使用して
    コンテキストを構築し、候補を予測する。
    """
    configured_digit_count = int(
        _config_value(
            config,
            "digit_count",
            default=0,
        )
        or 0
    )
    digit_min = int(
        _config_value(
            config,
            "digit_min",
            "min_num",
            default=0,
        )
        or 0
    )
    digit_max = int(
        _config_value(
            config,
            "digit_max",
            "max_num",
            default=9,
        )
        or 9
    )

    normalized_history = _normalize_history(
        history,
        digit_count=(
            configured_digit_count
            if configured_digit_count > 0
            else None
        ),
        digit_min=digit_min,
        digit_max=digit_max,
    )

    digit_count = len(
        normalized_history[0]
    )
    total_draws = len(
        normalized_history
    )

    resolved_train_window = (
        int(train_window)
        if train_window is not None
        else int(
            _config_value(
                config,
                "train_window",
                default=500,
            )
        )
    )
    resolved_tested_periods = (
        int(tested_periods)
        if tested_periods is not None
        else int(
            _config_value(
                config,
                "tested_periods",
                default=90,
            )
        )
    )
    resolved_top_k = (
        int(top_k)
        if top_k is not None
        else int(
            _config_value(
                config,
                "top_k",
                default=10,
            )
        )
    )

    if resolved_train_window <= 0:
        raise ValueError(
            "train_window must be greater than zero."
        )
    if resolved_tested_periods <= 0:
        raise ValueError(
            "tested_periods must be greater than zero."
        )
    if resolved_top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if total_draws <= resolved_train_window:
        return _empty_summary(
            digit_count
        )

    start_index = max(
        resolved_train_window,
        total_draws
        - resolved_tested_periods,
    )

    records: list[
        NumbersBacktestRecord
    ] = []

    for test_index in range(
        start_index,
        total_draws,
    ):
        training_history = (
            normalized_history[
                max(
                    0,
                    test_index
                    - resolved_train_window,
                ):test_index
            ]
        )
        actual = normalized_history[
            test_index
        ]

        context = (
            build_numbers_model_context(
                training_history,
                config,
            )
        )
        prediction = predict_numbers(
            context,
            top_k=resolved_top_k,
            weights=weights,
        )

        predicted = prediction.numbers
        box_dedicated_predicted = _select_box_candidates_from_ranked(
            prediction.ranked,
            top_k=resolved_top_k,
        )
        box_dedicated_unordered = [
            unordered_digit_matches(candidate, actual)
            for candidate in box_dedicated_predicted
        ]

        position_matches = [
            exact_position_matches(
                candidate,
                actual,
            )
            for candidate in predicted
        ]
        unordered_matches = [
            unordered_digit_matches(
                candidate,
                actual,
            )
            for candidate in predicted
        ]

        best_position = (
            max(position_matches)
            if position_matches
            else 0
        )
        average_position = (
            float(
                mean(position_matches)
            )
            if position_matches
            else 0.0
        )
        best_unordered = (
            max(unordered_matches)
            if unordered_matches
            else 0
        )
        average_unordered = (
            float(
                mean(unordered_matches)
            )
            if unordered_matches
            else 0.0
        )

        straight_hit = any(
            is_straight_hit(
                candidate,
                actual,
            )
            for candidate in predicted
        )
        box_hit = any(
            is_box_hit(
                candidate,
                actual,
            )
            for candidate in predicted
        )

        records.append(
            NumbersBacktestRecord(
                draw_index=test_index,
                actual=actual,
                predicted=predicted,
                best_exact_position_matches=(
                    best_position
                ),
                average_exact_position_matches=(
                    average_position
                ),
                best_unordered_digit_matches=(
                    best_unordered
                ),
                average_unordered_digit_matches=(
                    average_unordered
                ),
                straight_hit=straight_hit,
                box_hit=box_hit,
                box_dedicated_predicted=box_dedicated_predicted,
                box_dedicated_best_unordered_matches=(
                    max(box_dedicated_unordered)
                    if box_dedicated_unordered
                    else 0
                ),
                box_dedicated_average_unordered_matches=(
                    float(mean(box_dedicated_unordered))
                    if box_dedicated_unordered
                    else 0.0
                ),
                box_dedicated_hit=any(
                    is_box_hit(candidate, actual)
                    for candidate in box_dedicated_predicted
                ),
            )
        )

    return _summarize_numbers_records(
        records,
        digit_count=digit_count,
        include_records=include_records,
    )


def _select_box_candidates_from_ranked(
    ranked_candidates: Sequence[object],
    *,
    top_k: int,
) -> tuple[NumberRow, ...]:
    selected: list[NumberRow] = []
    seen: set[tuple[int, ...]] = set()
    for item in ranked_candidates:
        candidate = tuple(int(value) for value in getattr(item, "candidate"))
        box_digits = tuple(sorted(candidate))
        if box_digits in seen:
            continue
        seen.add(box_digits)
        selected.append(box_digits)
        if len(selected) >= top_k:
            break
    return tuple(selected)


def run_numbers_box_random_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    top_k: int | None = None,
    seed: int = 2025,
    include_records: bool = False,
) -> NumbersBoxBacktestSummary:
    """Random baseline using unique BOX signatures under the model budget.

    Each ticket is an unordered digit multiset, so permutations of the same
    BOX are never counted twice. This matches the production BOX candidate
    contract more closely than sampling arbitrary ordered Numbers tickets.
    """
    configured_digit_count = int(_config_value(config, "digit_count", default=0) or 0)
    normalized_history = _normalize_history(
        history, digit_count=configured_digit_count or None, digit_min=0, digit_max=9
    )
    digit_count = len(normalized_history[0])
    total_draws = len(normalized_history)
    resolved_train_window = int(train_window if train_window is not None else _config_value(config, "train_window", default=500))
    resolved_tested_periods = int(tested_periods if tested_periods is not None else _config_value(config, "tested_periods", default=90))
    resolved_top_k = int(top_k if top_k is not None else _config_value(config, "top_k", default=10))
    if min(resolved_train_window, resolved_tested_periods, resolved_top_k) <= 0:
        raise ValueError("backtest parameters must be positive.")
    box_space = tuple(combinations_with_replacement(range(10), digit_count))
    if resolved_top_k > len(box_space):
        raise ValueError("top_k exceeds the unique BOX candidate space.")
    if total_draws <= resolved_train_window:
        return NumbersBoxBacktestSummary(0, digit_count, None, None, None, ())

    start_index = max(resolved_train_window, total_draws - resolved_tested_periods)
    records: list[NumbersBoxBacktestRecord] = []
    for test_index in range(start_index, total_draws):
        rng = Random(seed + test_index)
        predicted_boxes = tuple(rng.sample(box_space, resolved_top_k))
        actual = normalized_history[test_index]
        unordered = [unordered_digit_matches(candidate, actual) for candidate in predicted_boxes]
        records.append(NumbersBoxBacktestRecord(
            draw_index=test_index,
            actual=actual,
            predicted_boxes=predicted_boxes,
            best_unordered_digit_matches=max(unordered) if unordered else 0,
            average_unordered_digit_matches=float(mean(unordered)) if unordered else 0.0,
            box_hit=any(is_box_hit(candidate, actual) for candidate in predicted_boxes),
        ))
    tested = len(records)
    return NumbersBoxBacktestSummary(
        tested_periods=tested,
        digit_count=digit_count,
        average_best_unordered_matches=round(float(mean(r.best_unordered_digit_matches for r in records)), 6) if records else None,
        average_unordered_matches_per_ticket=round(float(mean(r.average_unordered_digit_matches for r in records)), 6) if records else None,
        box_hit_rate=round(sum(r.box_hit for r in records) / tested, 6) if tested else None,
        records=tuple(records) if include_records else (),
    )


def run_numbers_box_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    top_k: int | None = None,
    weights: NumbersPredictionWeights | Mapping[str, object] | None = None,
    include_records: bool = False,
) -> NumbersBoxBacktestSummary:
    """Evaluate the actual BOX-dedicated candidate selection.

    Each test draw rebuilds the model only from earlier draws, ranks the full
    candidate space, then selects unique BOX signatures using the same rule as
    the production BOX output.
    """
    configured_digit_count = int(_config_value(config, "digit_count", default=0) or 0)
    normalized_history = _normalize_history(
        history,
        digit_count=configured_digit_count or None,
        digit_min=int(_config_value(config, "digit_min", "min_num", default=0) or 0),
        digit_max=int(_config_value(config, "digit_max", "max_num", default=9) or 9),
    )
    digit_count = len(normalized_history[0])
    total_draws = len(normalized_history)
    resolved_train_window = int(train_window if train_window is not None else _config_value(config, "train_window", default=500))
    resolved_tested_periods = int(tested_periods if tested_periods is not None else _config_value(config, "tested_periods", default=90))
    resolved_top_k = int(top_k if top_k is not None else _config_value(config, "top_k", default=10))
    if min(resolved_train_window, resolved_tested_periods, resolved_top_k) <= 0:
        raise ValueError("backtest parameters must be positive.")
    if total_draws <= resolved_train_window:
        return NumbersBoxBacktestSummary(0, digit_count, None, None, None, ())

    start_index = max(resolved_train_window, total_draws - resolved_tested_periods)
    records: list[NumbersBoxBacktestRecord] = []
    for test_index in range(start_index, total_draws):
        training_history = normalized_history[max(0, test_index - resolved_train_window):test_index]
        actual = normalized_history[test_index]
        context = build_numbers_model_context(training_history, config)
        prediction = predict_numbers(context, top_k=resolved_top_k, weights=weights)
        predicted_boxes = _select_box_candidates_from_ranked(
            prediction.ranked, top_k=resolved_top_k
        )
        unordered = [unordered_digit_matches(candidate, actual) for candidate in predicted_boxes]
        records.append(NumbersBoxBacktestRecord(
            draw_index=test_index,
            actual=actual,
            predicted_boxes=predicted_boxes,
            best_unordered_digit_matches=max(unordered) if unordered else 0,
            average_unordered_digit_matches=float(mean(unordered)) if unordered else 0.0,
            box_hit=any(is_box_hit(candidate, actual) for candidate in predicted_boxes),
        ))

    tested = len(records)
    return NumbersBoxBacktestSummary(
        tested_periods=tested,
        digit_count=digit_count,
        average_best_unordered_matches=round(float(mean(r.best_unordered_digit_matches for r in records)), 6) if records else None,
        average_unordered_matches_per_ticket=round(float(mean(r.average_unordered_digit_matches for r in records)), 6) if records else None,
        box_hit_rate=round(sum(r.box_hit for r in records) / tested, 6) if tested else None,
        records=tuple(records) if include_records else (),
    )


__all__ = [
    "NumbersBacktestRecord",
    "NumbersBacktestSummary",
    "NumbersBoxBacktestRecord",
    "NumbersBoxBacktestSummary",
    "is_box_hit",
    "is_straight_hit",
    "numbers_selection_score",
    "run_numbers_backtest",
    "run_numbers_box_backtest",
    "run_numbers_box_random_backtest",
    "run_numbers_uniform_random_backtest",
]
