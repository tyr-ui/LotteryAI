from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
ANALYSIS_PATH = OUTPUT_DIR / "feature_memory_analysis.json"


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

    weights = {}

    for item in ranking:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature")

        score = item.get("avg_importance")

        if (
            not isinstance(feature, str)
            or not isinstance(score, (int, float))
        ):
            continue

        score = max(min(float(score), 1.0), -1.0)

        weights[feature] = round(
            score * 0.10,
            4,
        )

    return weights


def print_learning_weights(
    game_name: str,
) -> None:
    weights = load_learning_weights(
        game_name
    )

    print(
        f"\n=== LEARNING WEIGHTS ({game_name}) ==="
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