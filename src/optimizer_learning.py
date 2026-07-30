from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Mapping, Sequence

from predictor import PredictionWeights

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
ANALYSIS_PATH = (
    OUTPUT_DIR
    / "feature_memory_analysis.json"
)
LEARNING_STRENGTH_PATH = (
    OUTPUT_DIR
    / "learning_strength.json"
)


def _load_analysis() -> dict:
    if not ANALYSIS_PATH.exists():
        return {}

    try:
        return json.loads(
            ANALYSIS_PATH.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return {}

def _load_learning_strength_store() -> dict:
    if not LEARNING_STRENGTH_PATH.exists():
        return {}

    try:
        data = json.loads(
            LEARNING_STRENGTH_PATH.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}

def load_learning_weights(
    game_name: str,
) -> dict[str, float]:
    analysis = _load_analysis()

    game = analysis.get(game_name)

    if not isinstance(game, dict):
        return {}

    overall = game.get("all_history")

    if not isinstance(overall, dict):
        return {}

    ranking = overall.get("features")

    if not isinstance(ranking, list):
        return {}

    weights: dict[str, float] = {}

    for item in ranking:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature")
        score = item.get(
            "average_selection_score_drop_percent"
        )

        if (
            not isinstance(feature, str)
            or not isinstance(
                score,
                (int, float),
            )
        ):
            continue

        learning_score = float(score) / 100.0

        learning_score = max(
            min(learning_score, 1.0),
                -1.0,
        )

        weights[feature] = round(
            learning_score * 0.10,
            4,
        )

    return weights


def apply_learning_weights(
    base_weights: (
        PredictionWeights
        | Mapping[str, float]
    ),
    learning_weights: Mapping[str, float],
    *,
    strength: float = 0.5,
) -> PredictionWeights | dict[str, float]:
    """
    過去のFeature Ablation分析を使って、
    既存の予測重みを弱く補正する。

    PredictionWeightsと辞書形式の
    両方に対応する。

    learning_weightsは最大±0.10であり、
    strength=0.5の場合、既存重みの補正率は
    最大で±5%となる。
    """
    normalized_strength = max(
        0.0,
        min(float(strength), 1.0),
    )

    aliases = {
        "freq": "global_frequency",
        "frequency": "global_frequency",
        "global_frequency": (
            "global_frequency"
        ),
        "recent": "recent_frequency",
        "recent_frequency": (
            "recent_frequency"
        ),
        "delay": "delay",
        "pair": "pair",
        "pair_weight": "pair",
        "triplet": "triplet",
        "triplet_weight": "triplet",
        "repeat": "repeat",
        "repeat_weight": "repeat",
        "distribution": "sum_shape",
        "sum": "sum_shape",
        "sum_shape": "sum_shape",
        "odd": "odd_shape",
        "odd_shape": "odd_shape",
        "low": "low_shape",
        "low_shape": "low_shape",
        "consecutive": (
            "consecutive_shape"
        ),
        "consecutive_shape": (
            "consecutive_shape"
        ),
        "span": "span_shape",
        "span_shape": "span_shape",
        "block": "block_shape",
        "blocks": "block_shape",
        "block_shape": "block_shape",
        "diversity": "diversity",
    }

    normalized_learning: dict[
        str,
        float,
    ] = {}

    for feature, value in (
        learning_weights.items()
    ):
        normalized_name = aliases.get(
            str(feature),
            str(feature),
        )

        try:
            learning_delta = float(value)
        except (TypeError, ValueError):
            continue

        normalized_learning[
            normalized_name
        ] = max(
            -0.10,
            min(learning_delta, 0.10),
        )

    if isinstance(
        base_weights,
        PredictionWeights,
    ):
        adjusted_values: dict[
            str,
            float,
        ] = {}

        for field_name in (
            base_weights.__dataclass_fields__
        ):
            original = float(
                getattr(
                    base_weights,
                    field_name,
                )
            )

            learning_delta = (
                normalized_learning.get(
                    field_name,
                    0.0,
                )
            )

            multiplier = (
                1.0
                + learning_delta
                * normalized_strength
            )

            adjusted_values[field_name] = (
                round(
                    max(
                        0.0,
                        original * multiplier,
                    ),
                    6,
                )
            )

        return PredictionWeights(
            **adjusted_values
        )

    adjusted_mapping: dict[
        str,
        float,
    ] = {}

    for feature, original_value in (
        base_weights.items()
    ):
        original = float(original_value)

        normalized_name = aliases.get(
            str(feature),
            str(feature),
        )

        learning_delta = (
            normalized_learning.get(
                normalized_name,
                0.0,
            )
        )

        multiplier = (
            1.0
            + learning_delta
            * normalized_strength
        )

        adjusted_mapping[str(feature)] = (
            round(
                max(
                    0.0,
                    original * multiplier,
                ),
                6,
            )
        )

    return adjusted_mapping

def load_learning_strength(
    game_name: str,
) -> float:
    """
    保存済みの最適strengthを優先して返す。

    保存値が存在しない場合は、
    Feature Memoryの実行回数に応じた
    従来の段階値へフォールバックする。
    """
    store = _load_learning_strength_store()

    stored_games = store.get("games")

    if isinstance(stored_games, dict):
        stored_game = stored_games.get(
            game_name
        )

        if isinstance(stored_game, dict):
            stored_strength = stored_game.get(
                "best_strength"
            )

            try:
                normalized_strength = float(
                    stored_strength
                )
            except (TypeError, ValueError):
                normalized_strength = -1.0

            if 0.0 <= normalized_strength <= 1.0:
                return normalized_strength

    analysis = _load_analysis()

    games = analysis.get("games")

    if not isinstance(games, dict):
        return 0.10

    game = games.get(game_name)

    if not isinstance(game, dict):
        return 0.10

    try:
        run_count = int(
            game.get(
                "total_run_count",
                0,
            )
        )
    except (TypeError, ValueError):
        run_count = 0

    if run_count >= 20:
        return 0.80

    if run_count >= 10:
        return 0.50

    if run_count >= 6:
        return 0.35

    if run_count >= 3:
        return 0.20

    return 0.10
    
def save_learning_strength_evaluation(
    game_name: str,
    best_strength: float,
    tested_strengths: Sequence[
        Mapping[str, object]
    ],
) -> None:
    """
    strength候補の評価結果と最良値を
    ゲーム別に保存する。
    """
    normalized_best = float(
        best_strength
    )

    if not 0.0 <= normalized_best <= 1.0:
        raise ValueError(
            "best_strength must be "
            "between 0.0 and 1.0."
        )

    store = _load_learning_strength_store()

    games = store.get("games")

    if not isinstance(games, dict):
        games = {}

    evaluated_at = datetime.now(
        timezone.utc
    ).isoformat()

    games[str(game_name)] = {
        "best_strength": round(
            normalized_best,
            6,
        ),
        "evaluated_at": evaluated_at,
        "tested_strengths": [
            dict(item)
            for item in tested_strengths
            if isinstance(item, Mapping)
        ],
    }

    output = {
        "schema_version": "1.0",
        "updated_at": evaluated_at,
        "games": games,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        LEARNING_STRENGTH_PATH.with_suffix(
            ".json.tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        LEARNING_STRENGTH_PATH
    )

def print_learning_weights(
    game_name: str,
) -> None:
    weights = load_learning_weights(
        game_name
    )

    print(
        f"\n=== LEARNING WEIGHTS "
        f"({game_name}) ==="
    )

    if not weights:
        print("No learning data.")
        return

    for feature, weight in sorted(
        weights.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{feature:<15} {weight:+.4f}"
        )


__all__ = [
    "apply_learning_weights",
    "load_learning_strength",
    "load_learning_weights",
    "print_learning_weights",
    "save_learning_strength_evaluation",
]