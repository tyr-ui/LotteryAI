from pathlib import Path
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from data_loader import dataframe_to_history, load_game_data
from games import LOTTO_GAMES
from optimizer import optimize
from review_output import write_review_outputs
from feature_memory_analyzer import save_feature_memory_analysis
from optimizer_learning import print_learning_weights
from numbers_backtester import run_numbers_backtest
from numbers_features import build_numbers_model_context
from numbers_predictor import predict_numbers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def game_family(config: dict) -> str:
    return str(config.get("family", "lotto")).lower()


def is_scheduled_no_new_data(
    previous_output: dict,
    validations: dict[str, dict],
) -> bool:
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        return False
    if not previous_output:
        return False

    return all(
        previous_output.get(game_key, {}).get("latest_draw_no")
        == validations[game_key].get("latest_draw_no")
        for game_key in LOTTO_GAMES
    )


def get_actual_numbers(
    df,
    draw_no: int,
    main_cols: list[str],
    *,
    ordered: bool,
) -> list[int] | None:
    hit = df[df["draw_no"] == draw_no]
    if hit.empty:
        return None

    row = hit.iloc[0]
    values = [int(row[column]) for column in main_cols]
    return values if ordered else sorted(values)


def unordered_matches(
    left: list[int],
    right: list[int],
) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)

    return sum(
        min(left_counts[digit], right_counts[digit])
        for digit in left_counts.keys() | right_counts.keys()
    )


def _empty_previous_evaluation(
    draw_type: str,
    status: str,
    message: str,
) -> dict:
    return {
        "draw_type": draw_type,
        "status": status,
        "message": message,
    }


def evaluate_previous_for_type(
    draw_type: str,
    previous_section: dict | None,
    current_df,
    main_cols: list[str],
    *,
    family: str,
) -> dict:
    if not previous_section:
        return _empty_previous_evaluation(
            draw_type,
            "no_previous_output",
            "前回のoptimizer_result.jsonがないため、答え合わせ対象がありません。",
        )

    target_draw_no = previous_section.get("next_draw_no")
    previous_predictions = previous_section.get("prediction", [])

    if target_draw_no is None or not previous_predictions:
        return _empty_previous_evaluation(
            draw_type,
            "no_previous_prediction",
            "前回予想の回号または予想データが見つかりません。",
        )

    actual_numbers = get_actual_numbers(
        current_df,
        int(target_draw_no),
        main_cols,
        ordered=(family == "numbers"),
    )

    if actual_numbers is None:
        latest_draw_no = (
            int(current_df["draw_no"].max())
            if len(current_df)
            else None
        )
        return {
            "draw_type": draw_type,
            "status": "pending",
            "target_draw_no": int(target_draw_no),
            "latest_draw_no": latest_draw_no,
            "message": "まだ前回予想対象回の結果がデータに反映されていません。",
        }

    evaluated_predictions = []

    if family == "numbers":
        for prediction in previous_predictions:
            numbers = [
                int(value)
                for value in prediction.get(
                    "numbers",
                    prediction.get("digits", []),
                )
            ]
            position_matches = sum(
                predicted == actual
                for predicted, actual in zip(numbers, actual_numbers)
            )
            unordered = unordered_matches(numbers, actual_numbers)

            evaluated_predictions.append({
                "pattern_id": prediction.get("pattern_id"),
                "numbers": numbers,
                "number": "".join(str(value) for value in numbers),
                "position_matches": position_matches,
                "unordered_matches": unordered,
                "straight_hit": numbers == actual_numbers,
                "box_hit": sorted(numbers) == sorted(actual_numbers),
                "score": prediction.get("score"),
                "model": prediction.get("model"),
            })

        position_counts = [
            item["position_matches"]
            for item in evaluated_predictions
        ]
        unordered_counts = [
            item["unordered_matches"]
            for item in evaluated_predictions
        ]

        return {
            "draw_type": draw_type,
            "status": "evaluated",
            "draw_no": int(target_draw_no),
            "actual_numbers": actual_numbers,
            "actual_number": "".join(
                str(value)
                for value in actual_numbers
            ),
            "evaluated_at": now_iso(),
            "predictions": evaluated_predictions,
            "best_match_count": (
                max(position_counts)
                if position_counts
                else 0
            ),
            "avg_match_count": (
                round(float(np.mean(position_counts)), 4)
                if position_counts
                else 0.0
            ),
            "best_unordered_match_count": (
                max(unordered_counts)
                if unordered_counts
                else 0
            ),
            "straight_hit": any(
                item["straight_hit"]
                for item in evaluated_predictions
            ),
            "box_hit": any(
                item["box_hit"]
                for item in evaluated_predictions
            ),
            **{
                f"hit_rate_{threshold}match": (
                    round(
                        sum(
                            count >= threshold
                            for count in position_counts
                        )
                        / len(position_counts),
                        4,
                    )
                    if position_counts
                    else 0.0
                )
                for threshold in range(1, 5)
            },
        }

    actual_set = set(actual_numbers)

    for prediction in previous_predictions:
        numbers = sorted(
            int(value)
            for value in prediction.get("numbers", [])
        )
        matched_numbers = sorted(actual_set & set(numbers))

        evaluated_predictions.append({
            "pattern_id": prediction.get("pattern_id"),
            "numbers": numbers,
            "matches": len(matched_numbers),
            "matched_numbers": matched_numbers,
            "score": prediction.get("score"),
            "model": prediction.get("model"),
        })

    match_counts = [
        prediction["matches"]
        for prediction in evaluated_predictions
    ]

    return {
        "draw_type": draw_type,
        "status": "evaluated",
        "draw_no": int(target_draw_no),
        "actual_numbers": actual_numbers,
        "evaluated_at": now_iso(),
        "predictions": evaluated_predictions,
        "best_match_count": max(match_counts) if match_counts else 0,
        "avg_match_count": (
            round(float(np.mean(match_counts)), 4)
            if match_counts
            else 0.0
        ),
        **{
            f"hit_rate_{threshold}match": (
                round(
                    sum(count >= threshold for count in match_counts)
                    / len(match_counts),
                    4,
                )
                if match_counts
                else 0.0
            )
            for threshold in range(1, 6)
        },
    }


def merge_evaluation_history(
    existing_history: list[dict],
    new_evaluations: list[dict],
) -> list[dict]:
    merged = {}

    for item in [*existing_history, *new_evaluations]:
        if item.get("status") != "evaluated":
            continue
        merged[
            (
                item.get("draw_type"),
                item.get("draw_no"),
            )
        ] = item

    history = list(merged.values())
    history.sort(
        key=lambda item: (
            item.get("draw_type", ""),
            item.get("draw_no", 0),
        )
    )
    return history


def build_evaluation_summary(
    history: list[dict],
) -> dict:
    summary = {}

    for draw_type in LOTTO_GAMES:
        items = [
            item
            for item in history
            if item.get("draw_type") == draw_type
            and item.get("status") == "evaluated"
        ]

        if not items:
            summary[draw_type] = {
                "evaluated_draws": 0,
                "avg_best_match_count": None,
                "avg_all_pattern_matches": None,
                "max_best_match_count": None,
                "best_draw_no": None,
                "latest_evaluated_draw_no": None,
            }
            continue

        best_counts = [
            item["best_match_count"]
            for item in items
        ]
        average_counts = [
            item["avg_match_count"]
            for item in items
        ]
        best_item = max(
            items,
            key=lambda item: item["best_match_count"],
        )

        summary[draw_type] = {
            "evaluated_draws": len(items),
            "avg_best_match_count": round(
                float(np.mean(best_counts)),
                4,
            ),
            "avg_all_pattern_matches": round(
                float(np.mean(average_counts)),
                4,
            ),
            "max_best_match_count": int(max(best_counts)),
            "best_draw_no": int(best_item["draw_no"]),
            "latest_evaluated_draw_no": int(
                max(item["draw_no"] for item in items)
            ),
        }

    return summary


def print_evaluation(
    title: str,
    evaluation: dict,
) -> None:
    print(
        f"\n=== {title} PREVIOUS PREDICTION EVALUATION ==="
    )
    print(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2,
        )
    )


def print_result(
    title: str,
    latest_draw_no: int,
    next_draw_no: int,
    result: dict,
) -> None:
    print(f"\n=== {title} OPTIMIZER RESULT ===")
    print(f"latest_draw_no: {latest_draw_no}")
    print(f"next_draw_no: {next_draw_no}")
    print(
        "selected_config: "
        f'{result.get("selected_config")}'
    )

    print("\n--- PREDICTION ---")
    for pattern in result.get("prediction", []):
        print(
            f'{pattern.get("pattern_id")}: '
            f'{pattern.get("numbers", [])} '
            f'score={pattern.get("score")} '
            f'model={pattern.get("model")}'
        )

    print("\n--- RANDOM BASELINE ---")
    print(
        json.dumps(
            result.get("random_baseline", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n--- RANKED CONFIGS ---")
    print(
        json.dumps(
            result.get("ranked_configs", []),
            ensure_ascii=False,
            indent=2,
        )
    )


def numbers_prediction_to_output(
    prediction,
) -> list[dict[str, object]]:
    return [
        {
            "pattern_id": f"P{index}",
            "numbers": list(item.candidate),
            "digits": list(item.candidate),
            "number": item.number,
            "score": round(float(item.total_score), 6),
            "model": "numbers_default",
            "components": dict(item.components),
            "exact_repeat_count": item.exact_repeat_count,
            "unordered_repeat_count": item.unordered_repeat_count,
        }
        for index, item in enumerate(
            prediction.selected,
            start=1,
        )
    ]


def run_numbers_game(
    df,
    game_config: dict,
) -> dict:
    history = dataframe_to_history(
        df,
        game_config,
    )
    context = build_numbers_model_context(
        history,
        game_config,
    )
    top_k = int(
        game_config.get("top_k", 10)
    )

    prediction_result = predict_numbers(
        context,
        top_k=top_k,
    )
    backtest = run_numbers_backtest(
        history,
        game_config,
        train_window=int(game_config["train_window"]),
        tested_periods=int(game_config["tested_periods"]),
        top_k=top_k,
    )
    backtest_output = backtest.to_dict()

    return {
        "random_baseline": {},
        "selected_random_filtered_baseline": {},
        "ranked_configs": [{
            "config": "numbers_default",
            **backtest_output,
        }],
        "selected_config": "numbers_default",
        "selected_weights": {},
        "selected_filters": {},
        "learning_summary": {
            "strength": 0.0,
            "strength_optimized": False,
            "tested_strengths": [],
            "loaded_weights": {},
            "base_prediction_weights": {},
            "applied_prediction_weights": {},
            "weight_diff": {},
        },
        "search_metadata": {
            "algorithm": "numbers_full_enumeration",
            "candidate_space_size": (
                prediction_result.generated_count
            ),
            "optimizer_connected": False,
            "note": (
                "Numbers3/4 currently use full candidate "
                "enumeration with default weights. "
                "Numbers optimizer integration is the next phase."
            ),
        },
        "feature_ablation": [],
        "optimizer_experience": {},
        "numbers_backtest": backtest_output,
        "prediction": numbers_prediction_to_output(
            prediction_result
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    previous_output_path = (
        OUTPUT_DIR / "optimizer_result.json"
    )
    history_path = (
        OUTPUT_DIR / "evaluation_history.json"
    )
    summary_path = (
        OUTPUT_DIR / "evaluation_summary.json"
    )

    previous_output = load_json(
        previous_output_path,
        {},
    )

    datasets = {}
    validations = {}

    for game_key, game_config in LOTTO_GAMES.items():
        loaded = load_game_data(
            game_key,
            game_config,
            destination=(
                ROOT
                / "data"
                / "raw"
                / f"{game_key}.csv"
            ),
        )
        datasets[game_key] = loaded.dataframe
        validations[game_key] = dict(
            loaded.validation
        )

    if is_scheduled_no_new_data(
        previous_output,
        validations,
    ):
        print("\n=== NO NEW DATA ===")
        print(
            "Scheduled run detected, but latest "
            "draw numbers have not changed."
        )
        for game_key, game_config in LOTTO_GAMES.items():
            latest_draw_no = validations[
                game_key
            ]["latest_draw_no"]
            print(
                f'{game_config["display_name"]} '
                f"latest_draw_no remains {latest_draw_no}."
            )
        print("Output files were not rewritten.")
        return

    previous_evaluations = {}

    for game_key, game_config in LOTTO_GAMES.items():
        previous_evaluations[game_key] = (
            evaluate_previous_for_type(
                draw_type=game_key,
                previous_section=previous_output.get(game_key),
                current_df=datasets[game_key],
                main_cols=game_config["main_cols"],
                family=game_family(game_config),
            )
        )

    existing_history = load_json(
        history_path,
        [],
    )
    evaluation_history = merge_evaluation_history(
        existing_history,
        list(previous_evaluations.values()),
    )
    evaluation_summary = build_evaluation_summary(
        evaluation_history
    )

    optimizer_results = {}

    for game_key, game_config in LOTTO_GAMES.items():
        if game_family(game_config) == "numbers":
            optimizer_results[game_key] = run_numbers_game(
                datasets[game_key],
                game_config,
            )
        else:
            optimizer_results[game_key] = optimize(
                df=datasets[game_key],
                main_cols=game_config["main_cols"],
                min_num=game_config["min_num"],
                max_num=game_config["max_num"],
                pick_count=game_config["pick_count"],
                train_window=game_config["train_window"],
                tested_periods=game_config["tested_periods"],
                bt_candidates=game_config[
                    "backtest_candidates"
                ],
                final_candidates=game_config[
                    "final_candidates"
                ],
            )

    game_output = {}

    for game_key in LOTTO_GAMES:
        validation = validations[game_key]
        game_output[game_key] = {
            "latest_draw_no": validation["latest_draw_no"],
            "next_draw_no": validation["latest_draw_no"] + 1,
            "rows": validation["rows"],
            "validation": validation,
            **optimizer_results[game_key],
        }

    output = {
        "status": "ok",
        "note": (
            "run_pipeline evaluates the previous prediction "
            "if the target draw is now available, then creates "
            "the next prediction."
        ),
        "generated_at": now_iso(),
        "previous_evaluation": previous_evaluations,
        "evaluation_summary": evaluation_summary,
        **game_output,
    }

    save_json(
        OUTPUT_DIR / "optimizer_result.json",
        output,
    )

    for game_key, game_config in LOTTO_GAMES.items():
        save_json(
            OUTPUT_DIR / game_config["prediction_filename"],
            optimizer_results[game_key]["prediction"],
        )

    save_json(history_path, evaluation_history)
    save_json(summary_path, evaluation_summary)

    run_summary_path, review_bundle_path = (
        write_review_outputs(
            output_dir=OUTPUT_DIR,
            output=output,
            game_keys=list(LOTTO_GAMES.keys()),
        )
    )

    save_feature_memory_analysis()

    print("\n=== REVIEW OUTPUTS ===")
    print(f"run_summary: {run_summary_path}")
    print(f"review_bundle: {review_bundle_path}")

    for game_key, game_config in LOTTO_GAMES.items():
        result = optimizer_results[game_key]
        section = output[game_key]

        if game_family(game_config) == "lotto":
            print_learning_weights(game_key)

        print_evaluation(
            game_config["display_name"],
            previous_evaluations[game_key],
        )
        print_result(
            game_config["display_name"],
            section["latest_draw_no"],
            section["next_draw_no"],
            result,
        )

    print("\n=== EVALUATION SUMMARY ===")
    print(
        json.dumps(
            evaluation_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    short_output = {"status": "ok"}

    for game_key in LOTTO_GAMES:
        result = optimizer_results[game_key]
        section = output[game_key]

        short_output[
            f"{game_key}_previous_evaluation_status"
        ] = previous_evaluations[game_key].get("status")
        short_output[
            f"{game_key}_latest_draw_no"
        ] = section["latest_draw_no"]
        short_output[
            f"{game_key}_next_draw_no"
        ] = section["next_draw_no"]
        short_output[
            f"{game_key}_selected_config"
        ] = result["selected_config"]
        short_output[
            f"{game_key}_prediction"
        ] = [
            pattern["numbers"]
            for pattern in result["prediction"]
        ]

    print("\n=== SHORT JSON ===")
    print(
        json.dumps(
            short_output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
