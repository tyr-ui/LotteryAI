from pathlib import Path
import json
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
ANALYSIS_PATH = (
    OUTPUT_DIR
    / "feature_memory_analysis.json"
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


def load_learning_weights(
    game_name: str,
) -> dict[str, float]:
    analysis = _load_analysis()

    game = analysis.get(game_name)

    if not isinstance(game, dict):
        return {}

    overall = game.get("overall")

    if not isinstance(overall, dict):
        return {}

    ranking = overall.get("ranking")

    if not isinstance(ranking, list):
        return {}

    weights: dict[str, float] = {}

    for item in ranking:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature")
        score = item.get("avg_importance")

        if (
            not isinstance(feature, str)
            or not isinstance(
                score,
                (int, float),
            )
        ):
            continue

        score = max(
            min(float(score), 1.0),
            -1.0,
        )

        weights[feature] = round(
            score * 0.10,
            4,
        )

    return weights


def apply_learning_weights(
    base_weights: Mapping[str, float],
    learning_weights: Mapping[str, float],
    *,
    strength: float = 0.5,
) -> dict[str, float]:
    """
    過去のFeature Ablation分析を使って、
    既存の予測重みを弱く補正する。

    learning_weightsは最大±0.10であり、
    strength=0.5の場合、既存重みの補正率は
    最大で±5%となる。

    元のbase_weightsは変更せず、
    補正後の新しいdictを返す。
    """
    normalized_strength = max(
        0.0,
        min(float(strength), 1.0),
    )

    adjusted_weights: dict[str, float] = {}

    for feature, original_value in (
        base_weights.items()
    ):
        original = float(original_value)

        learning_delta = float(
            learning_weights.get(
                feature,
                0.0,
            )
        )

        learning_delta = max(
            -0.10,
            min(learning_delta, 0.10),
        )

        multiplier = (
            1.0
            + learning_delta
            * normalized_strength
        )

        adjusted = original * multiplier

        adjusted_weights[feature] = round(
            max(0.0, adjusted),
            6,
        )

    return adjusted_weights


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
    "load_learning_weights",
    "print_learning_weights",
]