from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from statistics import mean
from typing import Iterable, Mapping, Sequence

from features import build_model_context
from predictor import (
    PredictionWeights,
    generate_candidates,
    predict,
)


NumberRow = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BacktestRecord:
    draw_index: int
    actual: NumberRow
    predicted: tuple[NumberRow, ...]
    best_match_count: int
    average_match_count: float


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    tested_periods: int
    average_best_matches: float | None
    average_matches_per_ticket: float | None
    hit_rate_1match: float | None
    hit_rate_2match: float | None
    hit_rate_3match: float | None
    hit_rate_4match: float | None
    hit_rate_5match: float | None
    hit_rate_6match: float | None
    hit_rate_7match: float | None
    records: tuple[BacktestRecord, ...]

    def to_dict(self, *, include_records: bool = False) -> dict:
        result = {
            "tested_periods": self.tested_periods,
            "average_best_matches": self.average_best_matches,
            "average_matches_per_ticket": self.average_matches_per_ticket,
            "hit_rate_1match": self.hit_rate_1match,
            "hit_rate_2match": self.hit_rate_2match,
            "hit_rate_3match": self.hit_rate_3match,
            "hit_rate_4match": self.hit_rate_4match,
            "hit_rate_5match": self.hit_rate_5match,
            "hit_rate_6match": self.hit_rate_6match,
            "hit_rate_7match": self.hit_rate_7match,
        }

        if include_records:
            result["records"] = [asdict(record) for record in self.records]

        return result


def _normalize_history(
    history: Iterable[Sequence[int]],
) -> tuple[NumberRow, ...]:
    normalized: list[NumberRow] = []

    for row in history:
        normalized.append(tuple(sorted(int(number) for number in row)))

    if not normalized:
        raise ValueError("history must contain at least one draw.")

    return tuple(normalized)


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


def _match_count(
    predicted: Sequence[int],
    actual: Sequence[int],
) -> int:
    return len(set(predicted).intersection(actual))



def _uniform_combination(
    population: Sequence[int],
    pick_count: int,
    rng,
) -> NumberRow:
    return tuple(sorted(rng.sample(list(population), pick_count)))


def _uniform_unique_combinations(
    population: Sequence[int],
    pick_count: int,
    count: int,
    rng,
) -> tuple[NumberRow, ...]:
    limit = comb(len(population), pick_count)
    if count > limit:
        raise ValueError("count exceeds the available combination space.")

    selected: dict[NumberRow, None] = {}
    while len(selected) < count:
        selected[_uniform_combination(population, pick_count, rng)] = None
    return tuple(selected)


def _summarize_records(
    records: Sequence[BacktestRecord],
    *,
    include_records: bool,
) -> BacktestSummary:
    tested = len(records)
    if tested == 0:
        return BacktestSummary(
            tested_periods=0,
            average_best_matches=None,
            average_matches_per_ticket=None,
            hit_rate_1match=None,
            hit_rate_2match=None,
            hit_rate_3match=None,
            hit_rate_4match=None,
            hit_rate_5match=None,
            hit_rate_6match=None,
            hit_rate_7match=None,
            records=(),
        )

    best_matches = [record.best_match_count for record in records]
    average_per_ticket = [record.average_match_count for record in records]

    def hit_rate(threshold: int) -> float:
        return round(
            sum(match >= threshold for match in best_matches) / tested,
            4,
        )

    return BacktestSummary(
        tested_periods=tested,
        average_best_matches=round(float(mean(best_matches)), 4),
        average_matches_per_ticket=round(float(mean(average_per_ticket)), 4),
        hit_rate_1match=hit_rate(1),
        hit_rate_2match=hit_rate(2),
        hit_rate_3match=hit_rate(3),
        hit_rate_4match=hit_rate(4),
        hit_rate_5match=hit_rate(5),
        hit_rate_6match=hit_rate(6),
        hit_rate_7match=hit_rate(7),
        records=tuple(records) if include_records else (),
    )


def run_uniform_random_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    top_k: int = 1,
    seed: int = 2025,
    include_records: bool = False,
) -> BacktestSummary:
    """全組合せから一様に抽出する、無フィルタの比較基準。"""
    from random import Random

    normalized_history = _normalize_history(history)
    total_draws = len(normalized_history)
    resolved_train_window = int(
        train_window if train_window is not None
        else _config_value(config, "train_window", default=100)
    )
    resolved_tested_periods = int(
        tested_periods if tested_periods is not None
        else _config_value(config, "tested_periods", default=30)
    )
    if resolved_train_window <= 0 or resolved_tested_periods <= 0 or top_k <= 0:
        raise ValueError("train_window, tested_periods and top_k must be positive.")

    min_num = int(_config_value(config, "min_num", default=1))
    max_num = int(_config_value(config, "max_num", default=43))
    pick_count = int(_config_value(config, "pick_count", default=len(normalized_history[0])))
    population = tuple(range(min_num, max_num + 1))
    if top_k > 1_000_000:
        raise ValueError("top_k is unreasonably large.")

    start_index = max(resolved_train_window, total_draws - resolved_tested_periods)
    records: list[BacktestRecord] = []
    for test_index in range(start_index, total_draws):
        rng = Random(seed + test_index)
        predicted = _uniform_unique_combinations(
            population, pick_count, top_k, rng
        )
        actual = normalized_history[test_index]
        matches = [_match_count(candidate, actual) for candidate in predicted]
        records.append(BacktestRecord(
            draw_index=test_index,
            actual=actual,
            predicted=predicted,
            best_match_count=max(matches) if matches else 0,
            average_match_count=float(mean(matches)) if matches else 0.0,
        ))
    return _summarize_records(records, include_records=include_records)


def run_filtered_random_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    candidate_count: int | None = None,
    top_k: int = 1,
    seed: int = 2025,
    include_records: bool = False,
) -> BacktestSummary:
    """形状・repeatフィルタ通過候補から一様に抽出する比較基準。"""
    from random import Random

    normalized_history = _normalize_history(history)
    total_draws = len(normalized_history)
    resolved_train_window = int(
        train_window if train_window is not None
        else _config_value(config, "train_window", default=100)
    )
    resolved_tested_periods = int(
        tested_periods if tested_periods is not None
        else _config_value(config, "tested_periods", default=30)
    )
    resolved_candidate_count = int(
        candidate_count if candidate_count is not None
        else _config_value(config, "backtest_candidates", default=300)
    )
    if min(resolved_train_window, resolved_tested_periods, resolved_candidate_count, top_k) <= 0:
        raise ValueError("backtest parameters must be positive.")

    zero_weights = PredictionWeights(
        global_frequency=0.0, recent_frequency=0.0, delay=0.0,
        pair=0.0, triplet=0.0, repeat=0.0,
        sum_shape=0.0, odd_shape=0.0, low_shape=0.0,
        consecutive_shape=0.0, span_shape=0.0, block_shape=0.0,
        diversity=0.0,
    )
    start_index = max(resolved_train_window, total_draws - resolved_tested_periods)
    records: list[BacktestRecord] = []
    for test_index in range(start_index, total_draws):
        training_history = normalized_history[max(0, test_index-resolved_train_window):test_index]
        context = build_model_context(training_history, config)
        candidates = generate_candidates(
            context, config, candidate_count=max(resolved_candidate_count, top_k),
            weights=zero_weights, seed=seed + test_index,
        )
        rng = Random(seed + 1_000_000 + test_index)
        predicted = tuple(rng.sample(list(candidates), min(top_k, len(candidates))))
        actual = normalized_history[test_index]
        matches = [_match_count(candidate, actual) for candidate in predicted]
        records.append(BacktestRecord(
            draw_index=test_index, actual=actual, predicted=predicted,
            best_match_count=max(matches) if matches else 0,
            average_match_count=float(mean(matches)) if matches else 0.0,
        ))
    return _summarize_records(records, include_records=include_records)


def run_backtest(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    train_window: int | None = None,
    tested_periods: int | None = None,
    candidate_count: int | None = None,
    top_k: int = 5,
    weights: PredictionWeights | Mapping[str, object] | None = None,
    seed: int = 2025,
    include_records: bool = False,
) -> BacktestSummary:
    """
    時系列順にウォークフォワード検証を行う。

    各検証回では、その回より前の履歴だけを使って特徴量を構築し、
    次回候補を生成する。未来情報は使用しない。
    """
    normalized_history = _normalize_history(history)
    total_draws = len(normalized_history)

    resolved_train_window = (
        int(train_window)
        if train_window is not None
        else int(_config_value(config, "train_window", default=100))
    )
    resolved_tested_periods = (
        int(tested_periods)
        if tested_periods is not None
        else int(_config_value(config, "tested_periods", default=30))
    )
    resolved_candidate_count = (
        int(candidate_count)
        if candidate_count is not None
        else int(_config_value(config, "backtest_candidates", default=300))
    )

    if resolved_train_window <= 0:
        raise ValueError("train_window must be greater than zero.")
    if resolved_tested_periods <= 0:
        raise ValueError("tested_periods must be greater than zero.")
    if resolved_candidate_count <= 0:
        raise ValueError("candidate_count must be greater than zero.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if total_draws <= resolved_train_window:
        return BacktestSummary(
            tested_periods=0,
            average_best_matches=None,
            average_matches_per_ticket=None,
            hit_rate_1match=None,
            hit_rate_2match=None,
            hit_rate_3match=None,
            hit_rate_4match=None,
            hit_rate_5match=None,
            hit_rate_6match=None,
            hit_rate_7match=None,
            records=(),
        )

    start_index = max(
        resolved_train_window,
        total_draws - resolved_tested_periods,
    )

    records: list[BacktestRecord] = []

    for test_index in range(start_index, total_draws):
        training_history = normalized_history[
            max(0, test_index - resolved_train_window):test_index
        ]
        actual = normalized_history[test_index]

        context = build_model_context(
            training_history,
            config,
        )

        prediction = predict(
            context,
            config,
            candidate_count=resolved_candidate_count,
            top_k=top_k,
            weights=weights,
            seed=seed + test_index,
        )

        predicted = prediction.numbers
        match_counts = [
            _match_count(candidate, actual)
            for candidate in predicted
        ]

        best_match = max(match_counts) if match_counts else 0
        average_match = mean(match_counts) if match_counts else 0.0

        records.append(
            BacktestRecord(
                draw_index=test_index,
                actual=actual,
                predicted=predicted,
                best_match_count=best_match,
                average_match_count=float(average_match),
            )
        )

    return _summarize_records(records, include_records=include_records)
