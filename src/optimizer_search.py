from __future__ import annotations

import math
from random import Random
from typing import Mapping, Sequence


WEIGHT_KEYS = (
    "freq",
    "recent",
    "pair",
    "triplet",
    "delay",
    "dist",
    "repeat",
)


# 既存の固定設定。
# 探索の初期点および比較基準として維持する。
BASE_CONFIGS = [
    {
        "name": "balanced_strict",
        "w": {
            "freq": 0.22,
            "recent": 0.24,
            "pair": 0.22,
            "triplet": 0.08,
            "delay": 0.08,
            "dist": 0.16,
        },
        "s": {
            "g": 0.35,
            "r": 0.40,
            "d": 0.25,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "balanced_loose",
        "w": {
            "freq": 0.22,
            "recent": 0.24,
            "pair": 0.22,
            "triplet": 0.08,
            "delay": 0.08,
            "dist": 0.16,
        },
        "s": {
            "g": 0.35,
            "r": 0.40,
            "d": 0.25,
        },
        "f": {
            "max_block": 3,
            "max_first": 3,
            "max_con": 2,
            "max_common": 4,
        },
    },
    {
        "name": "no_delay_strict",
        "w": {
            "freq": 0.24,
            "recent": 0.26,
            "pair": 0.24,
            "triplet": 0.08,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {
            "g": 0.45,
            "r": 0.55,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "no_delay_loose",
        "w": {
            "freq": 0.24,
            "recent": 0.26,
            "pair": 0.24,
            "triplet": 0.08,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {
            "g": 0.45,
            "r": 0.55,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 3,
            "max_con": 2,
            "max_common": 4,
        },
    },
    {
        "name": "freq_pair_strict",
        "w": {
            "freq": 0.30,
            "recent": 0.18,
            "pair": 0.28,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.18,
        },
        "s": {
            "g": 0.65,
            "r": 0.35,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "recent_pair_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.34,
            "pair": 0.26,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.20,
        },
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "delay_light_strict",
        "w": {
            "freq": 0.18,
            "recent": 0.20,
            "pair": 0.20,
            "triplet": 0.05,
            "delay": 0.15,
            "dist": 0.22,
        },
        "s": {
            "g": 0.30,
            "r": 0.35,
            "d": 0.35,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "dist_heavy_strict",
        "w": {
            "freq": 0.18,
            "recent": 0.20,
            "pair": 0.20,
            "triplet": 0.04,
            "delay": 0.04,
            "dist": 0.34,
        },
        "s": {
            "g": 0.40,
            "r": 0.45,
            "d": 0.15,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "repeat_light_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.30,
            "pair": 0.24,
            "triplet": 0.06,
            "delay": 0.00,
            "dist": 0.18,
            "repeat": 0.08,
        },
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
    {
        "name": "repeat_medium_strict",
        "w": {
            "freq": 0.14,
            "recent": 0.27,
            "pair": 0.22,
            "triplet": 0.05,
            "delay": 0.00,
            "dist": 0.17,
            "repeat": 0.15,
        },
        "s": {
            "g": 0.25,
            "r": 0.75,
            "d": 0.00,
        },
        "f": {
            "max_block": 3,
            "max_first": 2,
            "max_con": 1,
            "max_common": 3,
        },
    },
]


# 旧コードや外部参照との互換性。
CONFIGS = BASE_CONFIGS


def normalized_weight_dict(
    raw_weights: Mapping[str, object],
) -> dict[str, float]:
    """
    探索設定の重みを0以上へ補正し、合計1へ正規化する。

    すべての値が0以下の場合は、全項目を均等な重みにする。
    """
    values = {
        key: max(
            0.0,
            float(raw_weights.get(key, 0.0)),
        )
        for key in WEIGHT_KEYS
    }
    total = sum(values.values())

    if total <= 0:
        equal_weight = 1.0 / len(WEIGHT_KEYS)
        return {
            key: equal_weight
            for key in WEIGHT_KEYS
        }

    return {
        key: round(value / total, 8)
        for key, value in values.items()
    }


def config_signature(
    config: Mapping[str, object],
) -> tuple[float, ...]:
    """
    設定の重みから、重複判定用の署名を作成する。
    """
    raw_weights = config.get("w", {})

    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    normalized = normalized_weight_dict(
        raw_weights
    )

    return tuple(
        round(normalized[key], 6)
        for key in WEIGHT_KEYS
    )


def copy_search_config(
    config: Mapping[str, object],
    *,
    name: str,
    origin: str,
    parent: str | None = None,
) -> dict[str, object]:
    """
    探索に使用できる統一形式へ設定を変換する。
    """
    raw_weights = config.get("w", {})

    if not isinstance(raw_weights, Mapping):
        raw_weights = {}

    filters = config.get("f", {})

    if not isinstance(filters, Mapping):
        filters = {}

    scoring = config.get("s", {})

    if not isinstance(scoring, Mapping):
        scoring = {}

    return {
        "name": name,
        "w": normalized_weight_dict(
            raw_weights
        ),
        "s": dict(scoring),
        "f": dict(filters),
        "search_origin": origin,
        "parent": parent,
    }


def build_base_candidates() -> list[dict[str, object]]:
    """
    固定設定を探索候補形式へ変換する。
    """
    return [
        copy_search_config(
            config,
            name=str(config["name"]),
            origin="fixed",
        )
        for config in BASE_CONFIGS
    ]


def generate_random_candidates(
    *,
    count: int,
    rng: Random,
    inherited_filters: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    単体上でランダムな重み設定を生成する。

    指数分布由来の正の値を生成して正規化するため、
    各設定の重み合計は常に1となる。
    """
    if count < 0:
        raise ValueError(
            "count must be zero or greater."
        )

    candidates: list[dict[str, object]] = []

    for index in range(1, count + 1):
        raw_weights = {
            key: -math.log(
                max(rng.random(), 1e-12)
            )
            for key in WEIGHT_KEYS
        }

        # tripletおよびdelayが極端に支配しにくいよう抑える。
        raw_weights["triplet"] *= 0.65
        raw_weights["delay"] *= 0.80

        candidates.append({
            "name": f"random_{index:02d}",
            "w": normalized_weight_dict(
                raw_weights
            ),
            "s": {},
            "f": dict(inherited_filters),
            "search_origin": "random",
            "parent": None,
        })

    return candidates


def mutate_weights(
    weights: Mapping[str, object],
    *,
    rng: Random,
    scale: float,
) -> dict[str, float]:
    """
    親設定の近傍に、新しい重み設定を生成する。
    """
    if scale < 0:
        raise ValueError(
            "scale must be zero or greater."
        )

    normalized = normalized_weight_dict(
        weights
    )
    mutated: dict[str, float] = {}

    for key in WEIGHT_KEYS:
        base = normalized[key]
        additive = rng.gauss(
            0.0,
            scale,
        )
        multiplicative = math.exp(
            rng.gauss(
                0.0,
                scale * 0.75,
            )
        )

        mutated[key] = max(
            0.0,
            base * multiplicative + additive,
        )

    return normalized_weight_dict(mutated)


def generate_local_candidates(
    parents: Sequence[Mapping[str, object]],
    *,
    count: int,
    rng: Random,
) -> list[dict[str, object]]:
    """
    上位の親設定を基準に局所探索候補を生成する。
    """
    if count < 0:
        raise ValueError(
            "count must be zero or greater."
        )

    if not parents or count == 0:
        return []

    candidates: list[dict[str, object]] = []
    scales = (
        0.025,
        0.05,
        0.08,
    )

    for index in range(1, count + 1):
        parent = parents[
            (index - 1) % len(parents)
        ]

        parent_weights = parent.get(
            "w",
            {},
        )

        if not isinstance(
            parent_weights,
            Mapping,
        ):
            parent_weights = {}

        parent_filters = parent.get(
            "f",
            {},
        )

        if not isinstance(
            parent_filters,
            Mapping,
        ):
            parent_filters = {}

        parent_name = str(
            parent["name"]
        )
        scale = scales[
            (index - 1) % len(scales)
        ]

        candidates.append({
            "name": (
                f"local_{index:02d}_"
                f"{parent_name}"
            ),
            "w": mutate_weights(
                parent_weights,
                rng=rng,
                scale=scale,
            ),
            "s": {},
            "f": dict(parent_filters),
            "search_origin": "local",
            "parent": parent_name,
        })

    return candidates


def deduplicate_configs(
    configs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """
    正規化後の重みが同一となる探索設定を除外する。

    最初に現れた設定を残すため、固定設定の優先順位は維持される。
    """
    unique: list[dict[str, object]] = []
    signatures: set[
        tuple[float, ...]
    ] = set()

    for config in configs:
        signature = config_signature(
            config
        )

        if signature in signatures:
            continue

        signatures.add(signature)
        unique.append(dict(config))

    return unique


def find_parent_configs(
    results: Sequence[Mapping[str, object]],
    configs: Sequence[Mapping[str, object]],
    *,
    parent_count: int,
) -> list[dict[str, object]]:
    """
    評価結果の上位設定に対応する探索設定を取得する。
    """
    if parent_count < 0:
        raise ValueError(
            "parent_count must be zero or greater."
        )

    config_by_name = {
        str(config["name"]): dict(config)
        for config in configs
    }

    parents: list[dict[str, object]] = []

    for result in results[:parent_count]:
        name = str(result["config"])

        if name not in config_by_name:
            raise KeyError(
                "Evaluated configuration was not "
                f"found: {name}"
            )

        parents.append(
            config_by_name[name]
        )

    return parents