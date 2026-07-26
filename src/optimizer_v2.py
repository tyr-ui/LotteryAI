from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Iterable, Mapping, Sequence

from backtester import BacktestSummary, run_backtest
from predictor import PredictionWeights


NumberRow = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OptimizationMetric:
    """
    バックテスト結果から算出した最適化指標。

    3個以上の一致率を中心にしつつ、
    平均一致数と2個一致率も補助評価に使う。
    """

    objective: float
    average_best_matches: float
    hit_rate_2match: float
    hit_rate_3match: float
    hit_rate_4match: float
    hit_rate_5match: float


@dataclass(frozen=True, slots=True)
class OptimizationTrial:
    trial_id: int
    weights: PredictionWeights
    metric: OptimizationMetric
    backtest: BacktestSummary


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    best_weights: PredictionWeights
    best_metric: OptimizationMetric
    trials: tuple[OptimizationTrial, ...]
    evaluated_count: int
    seed: int

    def to_dict(self, *, include_trials: bool = True) -> dict:
        result = {
            "best_weights": _weights_to_dict(self.best_weights),
            "best_metric": _metric_to_dict(self.best_metric),
            "evaluated_count": self.evaluated_count,
            "seed": self.seed,
        }

        if include_trials:
            result["trials"] = [
                {
                    "trial_id": trial.trial_id,
                    "weights": _weights_to_dict(trial.weights),
                    "metric": _metric_to_dict(trial.metric),
                    "backtest": trial.backtest.to_dict(include_records=False),
                }
                for trial in self.trials
            ]

        return result


def _weights_to_dict(weights: PredictionWeights) -> dict[str, float]:
    return {
        field_name: float(getattr(weights, field_name))
        for field_name in weights.__dataclass_fields__
    }


def _metric_to_dict(metric: OptimizationMetric) -> dict[str, float]:
    return {
        "objective": metric.objective,
        "average_best_matches": metric.average_best_matches,
        "hit_rate_2match": metric.hit_rate_2match,
        "hit_rate_3match": metric.hit_rate_3match,
        "hit_rate_4match": metric.hit_rate_4match,
        "hit_rate_5match": metric.hit_rate_5match,
    }


def _safe(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def evaluate_backtest(summary: BacktestSummary) -> OptimizationMetric:
    """
    バックテスト結果を単一目的値へ変換する。

    高一致ほど強く評価するが、件数が少ない高一致だけに
    過剰適合しないよう平均一致数と2個一致率も含める。
    """
    average_best = _safe(summary.average_best_matches)
    hit2 = _safe(summary.hit_rate_2match)
    hit3 = _safe(summary.hit_rate_3match)
    hit4 = _safe(summary.hit_rate_4match)
    hit5 = _safe(summary.hit_rate_5match)

    objective = (
        0.20 * average_best
        + 0.15 * hit2
        + 0.30 * hit3
        + 0.25 * hit4
        + 0.10 * hit5
    )

    return OptimizationMetric(
        objective=round(float(objective), 8),
        average_best_matches=round(average_best, 8),
        hit_rate_2match=round(hit2, 8),
        hit_rate_3match=round(hit3, 8),
        hit_rate_4match=round(hit4, 8),
        hit_rate_5match=round(hit5, 8),
    )


def default_search_space() -> dict[str, tuple[float, ...]]:
    """
    初期探索空間。

    全組合せ探索ではなくランダム探索に使う。
    diversityは候補選択用なので別範囲にしている。
    """
    return {
        "global_frequency": (0.4, 0.7, 1.0, 1.3, 1.6),
        "recent_frequency": (0.6, 0.9, 1.2, 1.5, 1.8),
        "delay": (0.2, 0.5, 0.8, 1.1, 1.4),
        "pair": (0.5, 0.8, 1.1, 1.4, 1.7),
        "triplet": (0.0, 0.3, 0.6, 0.9, 1.2),
        "repeat": (0.4, 0.7, 1.0, 1.3),
        "sum_shape": (0.3, 0.5, 0.7, 0.9),
        "odd_shape": (0.2, 0.4, 0.6, 0.8),
        "low_shape": (0.2, 0.4, 0.6, 0.8),
        "consecutive_shape": (0.1, 0.3, 0.5, 0.7),
        "span_shape": (0.1, 0.3, 0.5),
        "block_shape": (0.2, 0.4, 0.6, 0.8),
        "diversity": (0.15, 0.25, 0.35, 0.45, 0.55),
    }


def _validate_search_space(
    search_space: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    valid_fields = set(PredictionWeights.__dataclass_fields__)
    normalized: dict[str, tuple[float, ...]] = {}

    for field_name, values in search_space.items():
        if field_name not in valid_fields:
            raise ValueError(f"Unknown weight field: {field_name}")

        converted = tuple(float(value) for value in values)
        if not converted:
            raise ValueError(
                f"Search-space field {field_name!r} has no candidate values."
            )
        if any(value < 0 for value in converted):
            raise ValueError(
                f"Search-space field {field_name!r} contains a negative value."
            )

        normalized[field_name] = converted

    if not normalized:
        raise ValueError("search_space must not be empty.")

    return normalized


def _sample_weights(
    rng: Random,
    search_space: Mapping[str, Sequence[float]],
    base: PredictionWeights,
) -> PredictionWeights:
    values = _weights_to_dict(base)

    for field_name, candidates in search_space.items():
        values[field_name] = float(rng.choice(tuple(candidates)))

    return PredictionWeights(**values)


def _weight_signature(weights: PredictionWeights) -> tuple[float, ...]:
    return tuple(
        round(float(getattr(weights, field_name)), 10)
        for field_name in weights.__dataclass_fields__
    )


def _sort_trials(
    trials: Iterable[OptimizationTrial],
) -> tuple[OptimizationTrial, ...]:
    return tuple(
        sorted(
            trials,
            key=lambda trial: (
                trial.metric.objective,
                trial.metric.hit_rate_4match,
                trial.metric.hit_rate_3match,
                trial.metric.average_best_matches,
            ),
            reverse=True,
        )
    )


def optimize_weights(
    history: Iterable[Sequence[int]],
    config: Mapping[str, object] | object,
    *,
    trial_count: int = 30,
    train_window: int | None = None,
    tested_periods: int | None = None,
    candidate_count: int | None = None,
    top_k: int = 5,
    search_space: Mapping[str, Sequence[float]] | None = None,
    base_weights: PredictionWeights | Mapping[str, object] | None = None,
    seed: int = 2025,
) -> OptimizationResult:
    """
    ランダム探索でPredictionWeightsを最適化する。

    trial_countを増やすほど探索は広がるが、
    各trialでウォークフォワード検証を行うため実行時間も増える。
    """
    if trial_count <= 0:
        raise ValueError("trial_count must be greater than zero.")

    normalized_history = tuple(
        tuple(sorted(int(number) for number in draw))
        for draw in history
    )
    if not normalized_history:
        raise ValueError("history must contain at least one draw.")

    resolved_space = _validate_search_space(
        search_space if search_space is not None else default_search_space()
    )
    resolved_base = (
        base_weights
        if isinstance(base_weights, PredictionWeights)
        else PredictionWeights.from_mapping(base_weights)
    )

    rng = Random(seed)
    trials: list[OptimizationTrial] = []
    seen: set[tuple[float, ...]] = set()

    # 既定重みも必ず評価する。
    candidates: list[PredictionWeights] = [resolved_base]
    seen.add(_weight_signature(resolved_base))

    max_unique = 1
    for values in resolved_space.values():
        max_unique *= len(values)

    target_count = min(trial_count, max_unique + 1)
    attempts = 0
    max_attempts = max(100, target_count * 50)

    while len(candidates) < target_count and attempts < max_attempts:
        sampled = _sample_weights(rng, resolved_space, resolved_base)
        signature = _weight_signature(sampled)
        attempts += 1

        if signature in seen:
            continue

        seen.add(signature)
        candidates.append(sampled)

    for trial_index, weights in enumerate(candidates, start=1):
        summary = run_backtest(
            normalized_history,
            config,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            top_k=top_k,
            weights=weights,
            seed=seed + trial_index * 1000,
            include_records=False,
        )
        metric = evaluate_backtest(summary)

        trials.append(
            OptimizationTrial(
                trial_id=trial_index,
                weights=weights,
                metric=metric,
                backtest=summary,
            )
        )

    ranked = _sort_trials(trials)
    if not ranked:
        raise RuntimeError("No optimization trial was evaluated.")

    best = ranked[0]

    return OptimizationResult(
        best_weights=best.weights,
        best_metric=best.metric,
        trials=ranked,
        evaluated_count=len(ranked),
        seed=seed,
    )


def build_small_grid(
    values: Mapping[str, Sequence[float]],
    *,
    base_weights: PredictionWeights | Mapping[str, object] | None = None,
) -> tuple[PredictionWeights, ...]:
    """
    少数項目だけの厳密なグリッド探索候補を生成する補助関数。

    多数項目を渡すと組合せ爆発するため、
    2〜4項目程度の比較に限定して使う。
    """
    normalized = _validate_search_space(values)
    resolved_base = (
        base_weights
        if isinstance(base_weights, PredictionWeights)
        else PredictionWeights.from_mapping(base_weights)
    )
    base = _weights_to_dict(resolved_base)

    field_names = tuple(normalized)
    result: list[PredictionWeights] = []

    for combination_values in product(
        *(normalized[field_name] for field_name in field_names)
    ):
        current = dict(base)
        current.update(zip(field_names, combination_values))
        result.append(PredictionWeights(**current))

    return tuple(result)
