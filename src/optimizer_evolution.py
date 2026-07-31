from __future__ import annotations

import math
from random import Random
from typing import Mapping, Sequence


DEFAULT_EVOLUTION_COUNT = 4
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_MUTATION_SCALE = 0.08


def _normalize_float(
    value: object,
    *,
    default: float = 0.0,
) -> float:
    """
    値を有限floatへ正規化する。
    """
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return float(default)

    if not math.isfinite(normalized):
        return float(default)

    return normalized


def _normalize_weight_mapping(
    weights: Mapping[str, object],
) -> dict[str, float]:
    """
    Config内の重みを非負floatへ正規化する。
    """
    normalized: dict[str, float] = {}

    for key, value in weights.items():
        number = _normalize_float(value)

        normalized[str(key)] = max(
            0.0,
            number,
        )

    if not normalized:
        raise ValueError(
            "Evolution parent weights "
            "must not be empty."
        )

    total = sum(normalized.values())

    if total <= 0.0:
        equal_weight = 1.0 / len(
            normalized
        )

        return {
            key: equal_weight
            for key in normalized
        }

    return {
        key: value / total
        for key, value in normalized.items()
    }


def _normalize_filters(
    filters: object,
) -> dict[str, object]:
    """
    Config内のフィルターを辞書へ正規化する。
    """
    if not isinstance(filters, Mapping):
        return {}

    return {
        str(key): value
        for key, value in filters.items()
    }


def _normalize_parent(
    parent: Mapping[str, object],
) -> dict[str, object]:
    """
    交叉に利用する親Configを正規化する。
    """
    weights = parent.get("w")

    if not isinstance(weights, Mapping):
        raise ValueError(
            "Evolution parent config must "
            "contain a mapping named 'w'."
        )

    return {
        "name": str(
            parent.get(
                "name",
                "unknown_parent",
            )
        ),
        "w": _normalize_weight_mapping(
            weights
        ),
        "f": _normalize_filters(
            parent.get("f", {})
        ),
    }


def _normalize_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    """
    交叉・突然変異後の重みを合計1へ正規化する。
    """
    non_negative = {
        str(key): max(
            0.0,
            _normalize_float(value),
        )
        for key, value in weights.items()
    }

    total = sum(non_negative.values())

    if total <= 0.0:
        if not non_negative:
            raise ValueError(
                "Evolution child weights "
                "must not be empty."
            )

        equal_weight = 1.0 / len(
            non_negative
        )

        return {
            key: equal_weight
            for key in non_negative
        }

    return {
        key: round(
            value / total,
            8,
        )
        for key, value in non_negative.items()
    }


def _mix_filter_value(
    left_value: object,
    right_value: object,
    rng: Random,
) -> object:
    """
    2つの親のフィルター値から
    子の値を1つ選択する。
    """
    if left_value == right_value:
        return left_value

    return (
        left_value
        if rng.random() < 0.5
        else right_value
    )


def crossover_config(
    left_parent: Mapping[str, object],
    right_parent: Mapping[str, object],
    *,
    child_name: str,
    rng: Random,
) -> dict[str, object]:
    """
    2つの親Configを交叉して子Configを生成する。

    各重みは、
    ・親Aと親Bの加重平均
    ・親Aまたは親Bから直接継承
    のいずれかで生成する。
    """
    left = _normalize_parent(
        left_parent
    )
    right = _normalize_parent(
        right_parent
    )

    left_weights = left["w"]
    right_weights = right["w"]

    if not isinstance(
        left_weights,
        Mapping,
    ):
        raise TypeError(
            "Normalized left weights "
            "must be a mapping."
        )

    if not isinstance(
        right_weights,
        Mapping,
    ):
        raise TypeError(
            "Normalized right weights "
            "must be a mapping."
        )

    weight_keys = sorted({
        *left_weights.keys(),
        *right_weights.keys(),
    })

    child_weights: dict[str, float] = {}

    blend_ratio = rng.uniform(
        0.25,
        0.75,
    )

    for key in weight_keys:
        left_value = _normalize_float(
            left_weights.get(key)
        )
        right_value = _normalize_float(
            right_weights.get(key)
        )

        crossover_mode = rng.random()

        if crossover_mode < 0.60:
            child_value = (
                left_value * blend_ratio
                + right_value
                * (1.0 - blend_ratio)
            )
        elif crossover_mode < 0.80:
            child_value = left_value
        else:
            child_value = right_value

        child_weights[str(key)] = max(
            0.0,
            child_value,
        )

    left_filters = left["f"]
    right_filters = right["f"]

    if not isinstance(
        left_filters,
        Mapping,
    ):
        left_filters = {}

    if not isinstance(
        right_filters,
        Mapping,
    ):
        right_filters = {}

    filter_keys = sorted({
        *left_filters.keys(),
        *right_filters.keys(),
    })

    child_filters = {
        str(key): _mix_filter_value(
            left_filters.get(key),
            right_filters.get(key),
            rng,
        )
        for key in filter_keys
    }

    return {
        "name": str(child_name),
        "w": _normalize_weights(
            child_weights
        ),
        "f": child_filters,
    }


def mutate_child(
    child: Mapping[str, object],
    *,
    rng: Random,
    mutation_rate: float = (
        DEFAULT_MUTATION_RATE
    ),
    mutation_scale: float = (
        DEFAULT_MUTATION_SCALE
    ),
) -> dict[str, object]:
    """
    子Configの重みに小さな突然変異を加える。
    """
    normalized_child = _normalize_parent(
        child
    )

    normalized_mutation_rate = min(
        1.0,
        max(
            0.0,
            _normalize_float(
                mutation_rate
            ),
        ),
    )

    normalized_mutation_scale = max(
        0.0,
        _normalize_float(
            mutation_scale
        ),
    )

    weights = normalized_child["w"]

    if not isinstance(weights, Mapping):
        raise TypeError(
            "Normalized child weights "
            "must be a mapping."
        )

    mutated_weights: dict[str, float] = {}

    mutation_applied = False

    for key, value in weights.items():
        normalized_value = (
            _normalize_float(value)
        )

        if (
            rng.random()
            < normalized_mutation_rate
        ):
            delta = rng.uniform(
                -normalized_mutation_scale,
                normalized_mutation_scale,
            )

            normalized_value = max(
                0.0,
                normalized_value + delta,
            )
            mutation_applied = True

        mutated_weights[str(key)] = (
            normalized_value
        )

    if (
        not mutation_applied
        and mutated_weights
        and normalized_mutation_rate > 0.0
    ):
        forced_key = rng.choice(
            list(mutated_weights)
        )

        forced_delta = rng.uniform(
            -normalized_mutation_scale,
            normalized_mutation_scale,
        )

        mutated_weights[forced_key] = max(
            0.0,
            mutated_weights[forced_key]
            + forced_delta,
        )

    return {
        "name": str(
            normalized_child["name"]
        ),
        "w": _normalize_weights(
            mutated_weights
        ),
        "f": dict(
            normalized_child["f"]
        ),
    }


def generate_evolution_candidates(
    parent_configs: Sequence[
        Mapping[str, object]
    ],
    *,
    count: int = DEFAULT_EVOLUTION_COUNT,
    rng: Random,
    mutation_rate: float = (
        DEFAULT_MUTATION_RATE
    ),
    mutation_scale: float = (
        DEFAULT_MUTATION_SCALE
    ),
) -> list[dict[str, object]]:
    """
    上位Config同士を交叉・突然変異させ、
    Evolution候補を生成する。

    親が2件未満の場合は候補を生成しない。
    """
    normalized_count = max(
        0,
        int(count),
    )

    if (
        normalized_count == 0
        or len(parent_configs) < 2
    ):
        return []

    normalized_parents = [
        _normalize_parent(parent)
        for parent in parent_configs
    ]

    candidates: list[
        dict[str, object]
    ] = []

    for index in range(
        normalized_count
    ):
        left_index, right_index = (
            rng.sample(
                range(
                    len(
                        normalized_parents
                    )
                ),
                2,
            )
        )

        left_parent = (
            normalized_parents[
                left_index
            ]
        )
        right_parent = (
            normalized_parents[
                right_index
            ]
        )

        child_name = (
            f"evolution_{index + 1:02d}_"
            f"{left_parent['name']}_"
            f"{right_parent['name']}"
        )

        child = crossover_config(
            left_parent,
            right_parent,
            child_name=child_name,
            rng=rng,
        )

        mutated_child = mutate_child(
            child,
            rng=rng,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
        )

        candidates.append(
            mutated_child
        )

    return candidates


__all__ = [
    "crossover_config",
    "generate_evolution_candidates",
    "mutate_child",
]