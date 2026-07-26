from __future__ import annotations

from dataclasses import dataclass
from math import log
from random import Random
from typing import Iterable, Mapping, Sequence

from features import (
    ModelContext,
    build_shape_features,
    candidate_pair_score,
    candidate_repeat_counts,
    candidate_repeat_likelihood,
    candidate_shape_likelihood,
    candidate_triplet_score,
    validate_candidate,
)


NumberRow = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PredictionWeights:
    """
    候補スコアで使用する重み。

    optimizer.pyから辞書形式で渡す場合にも対応する。
    """

    global_frequency: float = 1.00
    recent_frequency: float = 1.20
    delay: float = 0.80
    pair: float = 1.10
    triplet: float = 0.60
    repeat: float = 1.00
    sum_shape: float = 0.70
    odd_shape: float = 0.60
    low_shape: float = 0.60
    consecutive_shape: float = 0.45
    span_shape: float = 0.35
    block_shape: float = 0.55
    diversity: float = 0.35

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> "PredictionWeights":
        if values is None:
            return cls()

        aliases = {
            "freq": "global_frequency",
            "frequency": "global_frequency",
            "recent": "recent_frequency",
            "pair_weight": "pair",
            "triplet_weight": "triplet",
            "repeat_weight": "repeat",
            "distribution": "sum_shape",
        }

        normalized: dict[str, float] = {}
        valid_fields = set(cls.__dataclass_fields__)

        for key, value in values.items():
            field_name = aliases.get(key, key)
            if field_name not in valid_fields:
                continue
            try:
                normalized[field_name] = float(value)
            except (TypeError, ValueError):
                continue

        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: NumberRow
    total_score: float
    components: Mapping[str, float]
    repeat_counts: tuple[int, ...]
    passed_filters: bool


@dataclass(frozen=True, slots=True)
class PredictionResult:
    selected: tuple[CandidateScore, ...]
    ranked: tuple[CandidateScore, ...]
    generated_count: int
    accepted_count: int
    seed: int | None

    @property
    def numbers(self) -> tuple[NumberRow, ...]:
        return tuple(item.candidate for item in self.selected)


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


def _weighted_sample_without_replacement(
    population: Sequence[int],
    weights: Sequence[float],
    count: int,
    rng: Random,
) -> NumberRow:
    """
    Efraimidis-Spirakis方式による重み付き非復元抽出。
    """
    if len(population) != len(weights):
        raise ValueError("population and weights must have the same length.")
    if count <= 0:
        raise ValueError("count must be greater than zero.")
    if count > len(population):
        raise ValueError("count cannot exceed population size.")

    keyed: list[tuple[float, int]] = []
    for number, weight in zip(population, weights):
        safe_weight = max(float(weight), 1e-12)
        key = rng.random() ** (1.0 / safe_weight)
        keyed.append((key, number))

    keyed.sort(reverse=True)
    return tuple(sorted(number for _, number in keyed[:count]))


def _number_generation_weights(
    context: ModelContext,
    weights: PredictionWeights,
) -> tuple[float, ...]:
    result: list[float] = []

    for number in context.number_pool:
        components = context.number_score_components[number]
        score = (
            weights.global_frequency
            * components.get("global_frequency", 0.0)
            + weights.recent_frequency
            * components.get("recent_frequency", 0.0)
            + weights.delay
            * components.get("delay", 0.0)
        )

        # 0重みを避けつつ、差をやや強調する。
        result.append(max(0.05, 0.20 + score) ** 2)

    return tuple(result)


def _candidate_signature(candidate: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(number) for number in candidate))


def _passes_shape_filters(
    candidate: Sequence[int],
    context: ModelContext,
    config: Mapping[str, object] | object,
) -> bool:
    shape = build_shape_features(
        candidate,
        min_num=context.min_num,
        max_num=context.max_num,
        ranges=context.block_ranges,
    )

    allowed_odd = _config_value(
        config,
        "allowed_odd_counts",
        default=None,
    )
    if allowed_odd and shape.odd_count not in set(int(v) for v in allowed_odd):
        return False

    allowed_low = _config_value(
        config,
        "allowed_low_counts",
        default=None,
    )
    if allowed_low and shape.low_count not in set(int(v) for v in allowed_low):
        return False

    # 極端な合計値を除外する。平均±2.5σの範囲。
    sigma = context.sum_distribution.standard_deviation
    if sigma > 0:
        lower = context.sum_distribution.average - 2.5 * sigma
        upper = context.sum_distribution.average + 2.5 * sigma
        if not lower <= shape.total <= upper:
            return False

    # ブロックに全数字が偏る候補を除外する。
    if shape.block_counts and max(shape.block_counts) >= context.pick_count:
        return False

    return True


def _passes_repeat_filters(
    candidate: Sequence[int],
    context: ModelContext,
) -> bool:
    repeat_counts = candidate_repeat_counts(candidate, context)
    if not repeat_counts:
        return True

    # 直前回との全一致・極端な大量一致を除外。
    if repeat_counts[0] >= context.pick_count - 1:
        return False

    # 直近3回すべてと全く重ならない候補を強制排除しない。
    # repeat分布はスコア側で評価する。
    return True


def generate_candidates(
    context: ModelContext,
    config: Mapping[str, object] | object,
    *,
    candidate_count: int,
    weights: PredictionWeights | Mapping[str, object] | None = None,
    seed: int | None = None,
    max_attempt_multiplier: int = 20,
) -> tuple[NumberRow, ...]:
    """
    多段候補生成。

    Stage 1:
        Frequency / Recent / Delayによる重み付き生成
    Stage 2:
        形状フィルタ
    Stage 3:
        Repeatフィルタ
    """
    if candidate_count <= 0:
        raise ValueError("candidate_count must be greater than zero.")
    if max_attempt_multiplier <= 0:
        raise ValueError("max_attempt_multiplier must be greater than zero.")

    resolved_weights = (
        weights
        if isinstance(weights, PredictionWeights)
        else PredictionWeights.from_mapping(weights)
    )

    rng = Random(seed)
    number_weights = _number_generation_weights(context, resolved_weights)

    accepted: dict[NumberRow, None] = {}
    max_attempts = candidate_count * max_attempt_multiplier

    for _ in range(max_attempts):
        candidate = _weighted_sample_without_replacement(
            context.number_pool,
            number_weights,
            context.pick_count,
            rng,
        )

        if candidate in accepted:
            continue
        if not _passes_shape_filters(candidate, context, config):
            continue
        if not _passes_repeat_filters(candidate, context):
            continue

        accepted[candidate] = None
        if len(accepted) >= candidate_count:
            break

    # フィルタで必要数に届かない場合は、重複だけ避けて補完する。
    while len(accepted) < candidate_count and len(accepted) < _combination_limit(
        len(context.number_pool),
        context.pick_count,
    ):
        candidate = _weighted_sample_without_replacement(
            context.number_pool,
            number_weights,
            context.pick_count,
            rng,
        )
        accepted[candidate] = None

    return tuple(accepted)


def _combination_limit(n: int, r: int) -> int:
    if r < 0 or n < 0 or r > n:
        return 0
    r = min(r, n - r)
    value = 1
    for i in range(1, r + 1):
        value = value * (n - r + i) // i
    return value


def score_candidate(
    candidate: Sequence[int],
    context: ModelContext,
    *,
    weights: PredictionWeights | Mapping[str, object] | None = None,
) -> CandidateScore:
    """
    1候補を総合評価する。

    Frequency / Recent / Delay / Pair / Triplet / Repeat /
    Shape分布を一つのスコアへ統合する。
    """
    normalized = validate_candidate(candidate, context)
    resolved_weights = (
        weights
        if isinstance(weights, PredictionWeights)
        else PredictionWeights.from_mapping(weights)
    )

    number_components = [
        context.number_score_components[number]
        for number in normalized
    ]

    global_frequency = sum(
        item.get("global_frequency", 0.0)
        for item in number_components
    ) / len(number_components)

    recent_frequency = sum(
        item.get("recent_frequency", 0.0)
        for item in number_components
    ) / len(number_components)

    delay = sum(
        item.get("delay", 0.0)
        for item in number_components
    ) / len(number_components)

    pair = candidate_pair_score(normalized, context)
    triplet = candidate_triplet_score(normalized, context)
    repeat = candidate_repeat_likelihood(normalized, context)

    shape = candidate_shape_likelihood(normalized, context)
    components = {
        "global_frequency": global_frequency,
        "recent_frequency": recent_frequency,
        "delay": delay,
        "pair": pair,
        "triplet": triplet,
        "repeat": repeat,
        "sum_shape": shape["sum"],
        "odd_shape": shape["odd"],
        "low_shape": shape["low"],
        "consecutive_shape": shape["consecutive"],
        "span_shape": shape["span"],
        "block_shape": shape["blocks"],
    }

    total = (
        resolved_weights.global_frequency * global_frequency
        + resolved_weights.recent_frequency * recent_frequency
        + resolved_weights.delay * delay
        + resolved_weights.pair * pair
        + resolved_weights.triplet * triplet
        + resolved_weights.repeat * repeat
        + resolved_weights.sum_shape * _safe_log_probability(shape["sum"])
        + resolved_weights.odd_shape * _safe_log_probability(shape["odd"])
        + resolved_weights.low_shape * _safe_log_probability(shape["low"])
        + resolved_weights.consecutive_shape
        * _safe_log_probability(shape["consecutive"])
        + resolved_weights.span_shape * _safe_log_probability(shape["span"])
        + resolved_weights.block_shape * _safe_log_probability(shape["blocks"])
    )

    return CandidateScore(
        candidate=normalized,
        total_score=float(total),
        components=components,
        repeat_counts=candidate_repeat_counts(normalized, context),
        passed_filters=True,
    )


def _safe_log_probability(value: float) -> float:
    """
    確率値を0..1程度の加点へ変換する。
    """
    safe = max(float(value), 1e-12)
    return 1.0 / (1.0 + abs(log(safe)))


def rank_candidates(
    candidates: Iterable[Sequence[int]],
    context: ModelContext,
    *,
    weights: PredictionWeights | Mapping[str, object] | None = None,
) -> tuple[CandidateScore, ...]:
    scored = [
        score_candidate(candidate, context, weights=weights)
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


def candidate_overlap(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    return len(set(left).intersection(right))


def select_diverse(
    ranked: Sequence[CandidateScore],
    *,
    top_k: int,
    diversity_weight: float = 0.35,
    max_overlap: int | None = None,
) -> tuple[CandidateScore, ...]:
    """
    高得点候補から、数字の重複を抑えた複数口を選ぶ。
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not ranked:
        return ()

    selected: list[CandidateScore] = []
    remaining = list(ranked)

    while remaining and len(selected) < top_k:
        best_index = 0
        best_adjusted = float("-inf")

        for index, item in enumerate(remaining):
            if not selected:
                adjusted = item.total_score
            else:
                overlaps = [
                    candidate_overlap(item.candidate, chosen.candidate)
                    for chosen in selected
                ]
                maximum_overlap = max(overlaps)

                if max_overlap is not None and maximum_overlap > max_overlap:
                    continue

                average_overlap = sum(overlaps) / len(overlaps)
                adjusted = (
                    item.total_score
                    - diversity_weight * average_overlap
                )

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index

        selected.append(remaining.pop(best_index))

    return tuple(selected)


def predict(
    context: ModelContext,
    config: Mapping[str, object] | object,
    *,
    candidate_count: int | None = None,
    top_k: int | None = None,
    weights: PredictionWeights | Mapping[str, object] | None = None,
    seed: int | None = 2025,
) -> PredictionResult:
    """
    候補生成、スコアリング、多様性選択を一括実行する。
    """
    if candidate_count is None:
        candidate_count = int(
            _config_value(
                config,
                "final_candidates",
                default=10000,
            )
        )
    if top_k is None:
        top_k = int(
            _config_value(
                config,
                "top_k",
                default=5,
            )
        )

    resolved_weights = (
        weights
        if isinstance(weights, PredictionWeights)
        else PredictionWeights.from_mapping(weights)
    )

    candidates = generate_candidates(
        context,
        config,
        candidate_count=candidate_count,
        weights=resolved_weights,
        seed=seed,
    )
    ranked = rank_candidates(
        candidates,
        context,
        weights=resolved_weights,
    )
    selected = select_diverse(
        ranked,
        top_k=top_k,
        diversity_weight=resolved_weights.diversity,
        max_overlap=max(1, context.pick_count - 2),
    )

    return PredictionResult(
        selected=selected,
        ranked=ranked,
        generated_count=candidate_count,
        accepted_count=len(candidates),
        seed=seed,
    )
