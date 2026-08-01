from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
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
    records: tuple[NumbersBacktestRecord, ...]

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
        records=(),
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
            )
        )

    tested = len(records)

    if tested == 0:
        return _empty_summary(
            digit_count
        )

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

    def position_hit_rate(
        threshold: int,
    ) -> float:
        if threshold > digit_count:
            return 0.0

        return round(
            sum(
                value >= threshold
                for value
                in best_position_values
            )
            / tested,
            6,
        )

    straight_hit_rate = round(
        sum(
            record.straight_hit
            for record in records
        )
        / tested,
        6,
    )
    box_hit_rate = round(
        sum(
            record.box_hit
            for record in records
        )
        / tested,
        6,
    )

    average_best_position_matches = round(
        float(
            mean(
                best_position_values
            )
        ),
        6,
    )
    average_position_matches_per_ticket = round(
        float(
            mean(
                average_position_values
            )
        ),
        6,
    )
    average_best_unordered_matches = round(
        float(
            mean(
                best_unordered_values
            )
        ),
        6,
    )
    average_unordered_matches_per_ticket = round(
        float(
            mean(
                average_unordered_values
            )
        ),
        6,
    )

    hit_rate_1 = position_hit_rate(1)
    hit_rate_2 = position_hit_rate(2)
    hit_rate_3 = position_hit_rate(3)
    hit_rate_4 = position_hit_rate(4)

    selection_score = numbers_selection_score(
        digit_count=digit_count,
        average_best_position_matches=(
            average_best_position_matches
        ),
        average_best_unordered_matches=(
            average_best_unordered_matches
        ),
        straight_hit_rate=(
            straight_hit_rate
        ),
        box_hit_rate=(
            box_hit_rate
        ),
        hit_rate_2_position=(
            hit_rate_2
        ),
        hit_rate_3_position=(
            hit_rate_3
        ),
        hit_rate_4_position=(
            hit_rate_4
        ),
    )

    return NumbersBacktestSummary(
        tested_periods=tested,
        digit_count=digit_count,
        average_best_position_matches=(
            average_best_position_matches
        ),
        average_position_matches_per_ticket=(
            average_position_matches_per_ticket
        ),
        average_best_unordered_matches=(
            average_best_unordered_matches
        ),
        average_unordered_matches_per_ticket=(
            average_unordered_matches_per_ticket
        ),
        straight_hit_rate=(
            straight_hit_rate
        ),
        box_hit_rate=(
            box_hit_rate
        ),
        hit_rate_1_position=(
            hit_rate_1
        ),
        hit_rate_2_position=(
            hit_rate_2
        ),
        hit_rate_3_position=(
            hit_rate_3
        ),
        hit_rate_4_position=(
            hit_rate_4
        ),
        selection_score=(
            selection_score
        ),
        records=(
            tuple(records)
            if include_records
            else ()
        ),
    )


__all__ = [
    "NumbersBacktestRecord",
    "NumbersBacktestSummary",
    "is_box_hit",
    "is_straight_hit",
    "numbers_selection_score",
    "run_numbers_backtest",
]
