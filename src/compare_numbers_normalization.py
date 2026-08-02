from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numbers_backtester
from data_loader import dataframe_to_history, load_game_data
from games import LOTTO_GAMES
from numbers_backtester import NumbersBacktestSummary
from numbers_predictor import (
    NumbersPredictionResult,
    NumbersPredictionWeights,
)
from numbers_predictor_rank_v2 import (
    NORMALIZATION_VERSION,
    predict_numbers_rank_v2,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
OPTIMIZER_RESULT_PATH = OUTPUT_DIR / "optimizer_result.json"
RESULT_PATH = OUTPUT_DIR / "numbers_normalization_ab.json"
GAME_KEYS = ("numbers3", "numbers4")

COMPARISON_METRICS = (
    "selection_score",
    "average_best_position_matches",
    "average_position_matches_per_ticket",
    "average_best_unordered_matches",
    "average_unordered_matches_per_ticket",
    "straight_hit_rate",
    "box_hit_rate",
    "hit_rate_1_position",
    "hit_rate_2_position",
    "hit_rate_3_position",
    "hit_rate_4_position",
)

PredictNumbers = Callable[..., NumbersPredictionResult]


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return loaded


def _save_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def _mapping_value(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected mapping for {field_name}."
        )

    return value


def _numeric_value(
    value: object,
) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_deltas(
    baseline: Mapping[str, object],
    experiment: Mapping[str, object],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}

    for metric_name in COMPARISON_METRICS:
        baseline_value = _numeric_value(
            baseline.get(metric_name)
        )
        experiment_value = _numeric_value(
            experiment.get(metric_name)
        )

        if (
            baseline_value is None
            or experiment_value is None
        ):
            result[metric_name] = None
            continue

        result[metric_name] = round(
            experiment_value - baseline_value,
            6,
        )

    return result


def _run_backtest_with_predictor(
    history: Sequence[Sequence[int]],
    config: Mapping[str, object],
    *,
    predictor: PredictNumbers,
    weights: NumbersPredictionWeights,
    tested_periods: int,
) -> NumbersBacktestSummary:
    original_predictor = (
        numbers_backtester.predict_numbers
    )
    numbers_backtester.predict_numbers = predictor

    try:
        return numbers_backtester.run_numbers_backtest(
            history,
            config,
            train_window=int(
                config.get("train_window", 500)
            ),
            tested_periods=tested_periods,
            top_k=int(config.get("top_k", 10)),
            weights=weights,
        )
    finally:
        numbers_backtester.predict_numbers = (
            original_predictor
        )


def _compare_game(
    game_key: str,
    baseline_output: Mapping[str, object],
) -> dict[str, object]:
    raw_config = LOTTO_GAMES[game_key]
    config = _mapping_value(
        raw_config,
        field_name=f"LOTTO_GAMES[{game_key}]",
    )
    baseline_section = _mapping_value(
        baseline_output.get(game_key),
        field_name=f"optimizer_result[{game_key}]",
    )
    baseline_backtest = _mapping_value(
        baseline_section.get("numbers_backtest"),
        field_name=(
            f"optimizer_result[{game_key}]"
            ".numbers_backtest"
        ),
    )
    selected_weights = _mapping_value(
        baseline_section.get("selected_weights"),
        field_name=(
            f"optimizer_result[{game_key}]"
            ".selected_weights"
        ),
    )

    loaded = load_game_data(
        game_key,
        config,
    )
    latest_draw_no = int(
        loaded.validation["latest_draw_no"]
    )
    baseline_latest_draw_no = int(
        baseline_section["latest_draw_no"]
    )

    if latest_draw_no != baseline_latest_draw_no:
        raise RuntimeError(
            f"{game_key}: baseline latest_draw_no="
            f"{baseline_latest_draw_no}, current data="
            f"{latest_draw_no}. Run the normal Full Run first."
        )

    history = dataframe_to_history(
        loaded.dataframe,
        config,
    )
    weights = NumbersPredictionWeights.from_mapping(
        selected_weights
    )
    tested_periods = int(
        baseline_backtest.get(
            "tested_periods",
            config.get("tested_periods", 180),
        )
    )

    started = perf_counter()
    experiment_summary = (
        _run_backtest_with_predictor(
            history,
            config,
            predictor=predict_numbers_rank_v2,
            weights=weights,
            tested_periods=tested_periods,
        )
    )
    elapsed_seconds = round(
        perf_counter() - started,
        3,
    )
    experiment_backtest = (
        experiment_summary.to_dict()
    )

    return {
        "latest_draw_no": latest_draw_no,
        "tested_periods": tested_periods,
        "selected_config": baseline_section.get(
            "selected_config"
        ),
        "shared_weights": {
            key: float(value)
            for key, value in selected_weights.items()
        },
        "baseline": {
            "predictor": "numbers_predictor",
            "normalization": "none",
            "backtest": dict(baseline_backtest),
        },
        "experiment": {
            "predictor": "numbers_predictor_rank_v2",
            "normalization": NORMALIZATION_VERSION,
            "elapsed_seconds": elapsed_seconds,
            "backtest": experiment_backtest,
        },
        "delta_experiment_minus_baseline": (
            _metric_deltas(
                baseline_backtest,
                experiment_backtest,
            )
        ),
    }


def main() -> None:
    started = perf_counter()
    baseline_output = _load_json(
        OPTIMIZER_RESULT_PATH
    )

    comparisons = {
        game_key: _compare_game(
            game_key,
            baseline_output,
        )
        for game_key in GAME_KEYS
    }

    result = {
        "status": "ok",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Compare current Numbers scoring with "
            "rank-percentile normalization using the "
            "same data, selected weights, top_k, train "
            "window, and tested periods."
        ),
        "normalization_version": (
            NORMALIZATION_VERSION
        ),
        "baseline_source": str(
            OPTIMIZER_RESULT_PATH.relative_to(ROOT)
        ),
        "total_elapsed_seconds": round(
            perf_counter() - started,
            3,
        ),
        "comparisons": comparisons,
    }

    _save_json(
        RESULT_PATH,
        result,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
