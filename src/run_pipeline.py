from pathlib import Path
import json
import os
from collections import Counter

import numpy as np

from data_loader import dataframe_to_history, load_game_data
from games import LOTTO_GAMES
from optimizer import optimize
from review_output import write_review_outputs
from feature_memory_analyzer import save_feature_memory_analysis
from optimizer_learning import load_learning_strength, print_learning_weights
from numbers_optimizer import optimize_numbers
from optimizer_experience import save_optimizer_experience
from common import now_iso
from storage import save_json, load_json
from evaluation_dashboard import write_evaluation_dashboard
from carryover import fetch_carryover_snapshot
from notification_summary import write_notification_summary


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"



def game_family(config: dict) -> str:
    return str(config.get("family", "lotto")).lower()




def save_prediction_outputs(
    output_dir: Path,
    optimizer_results: dict[str, dict],
    game_configs: dict[str, dict],
) -> None:
    """
    各ゲームの予想結果を個別JSONへ保存する。

    Numbers系では、通常予想に加えて
    BOX専用予想も保存する。
    """
    for game_key, game_config in game_configs.items():
        result = optimizer_results[game_key]

        save_json(
            output_dir / game_config["prediction_filename"],
            result["prediction"],
        )

        if game_family(game_config) == "numbers":
            save_json(
                output_dir / f"prediction_box_{game_key}.json",
                result.get("box_prediction", []),
            )

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


def build_experience_prediction_weights(
    selected_weights: dict[str, object],
) -> dict[str, float]:
    """
    Optimizerの選択重みを、Experience保存用の
    PredictionWeights相当のJSON形式へ変換する。
    """
    weight_keys = (
        "freq",
        "recent",
        "pair",
        "triplet",
        "delay",
        "dist",
        "repeat",
    )
    normalized = {
        key: max(0.0, float(selected_weights.get(key, 0.0)))
        for key in weight_keys
    }
    total = sum(normalized.values())

    if total <= 0.0:
        equal = 1.0 / len(weight_keys)
        normalized = {key: equal for key in weight_keys}
    else:
        normalized = {
            key: value / total
            for key, value in normalized.items()
        }

    shape_weight = normalized["dist"] / 6.0

    return {
        "global_frequency": round(normalized["freq"], 6),
        "recent_frequency": round(normalized["recent"], 6),
        "delay": round(normalized["delay"], 6),
        "pair": round(normalized["pair"], 6),
        "triplet": round(normalized["triplet"], 6),
        "repeat": round(normalized["repeat"], 6),
        "sum_shape": round(shape_weight, 6),
        "odd_shape": round(shape_weight, 6),
        "low_shape": round(shape_weight, 6),
        "consecutive_shape": round(shape_weight, 6),
        "span_shape": round(shape_weight, 6),
        "block_shape": round(shape_weight, 6),
        "diversity": 0.35,
    }


def save_lotto_optimizer_experience(
    game_key: str,
    optimizer_result: dict[str, object],
) -> dict[str, object]:
    """LOTO系Optimizerの今回の勝者をExperienceへ保存する。"""
    ranked_configs = optimizer_result.get("ranked_configs", [])
    if not isinstance(ranked_configs, list) or not ranked_configs:
        raise ValueError(
            f"{game_key}: ranked_configs is empty; "
            "optimizer experience cannot be saved."
        )

    best_evaluation = ranked_configs[0]
    if not isinstance(best_evaluation, dict):
        raise TypeError(
            f"{game_key}: best optimizer evaluation must be a mapping."
        )

    selected_weights = optimizer_result.get("selected_weights", {})
    selected_filters = optimizer_result.get("selected_filters", {})
    if not isinstance(selected_weights, dict):
        raise TypeError(
            f"{game_key}: selected_weights must be a mapping."
        )
    if not isinstance(selected_filters, dict):
        selected_filters = {}

    selected_config = str(optimizer_result.get("selected_config", ""))
    if not selected_config:
        raise ValueError(
            f"{game_key}: selected_config is missing."
        )

    return save_optimizer_experience(
        game_name=game_key,
        config_name=selected_config,
        config={
            "w": selected_weights,
            "f": selected_filters,
        },
        evaluation=best_evaluation,
        prediction_weights=build_experience_prediction_weights(
            selected_weights
        ),
        learning_strength=load_learning_strength(game_key),
    )


def save_numbers_optimizer_experience(
    game_key: str,
    optimizer_result: dict[str, object],
) -> dict[str, object]:
    """Numbers Optimizerの今回の勝者をExperienceへ保存する。"""
    ranked_configs = optimizer_result.get("ranked_configs", [])
    if not isinstance(ranked_configs, list) or not ranked_configs:
        raise ValueError(
            f"{game_key}: ranked_configs is empty; "
            "numbers optimizer experience cannot be saved."
        )

    best_evaluation = ranked_configs[0]
    if not isinstance(best_evaluation, dict):
        raise TypeError(
            f"{game_key}: best Numbers evaluation must be a mapping."
        )

    selected_weights = optimizer_result.get("selected_weights", {})
    if not isinstance(selected_weights, dict) or not selected_weights:
        raise TypeError(
            f"{game_key}: selected_weights must be a non-empty mapping."
        )

    selected_config = str(optimizer_result.get("selected_config", ""))
    if not selected_config:
        raise ValueError(
            f"{game_key}: selected_config is missing."
        )

    default_result = optimizer_result.get("random_baseline", {})
    default_selection_score = (
        float(default_result.get("selection_score", 0.0) or 0.0)
        if isinstance(default_result, dict)
        else 0.0
    )
    selected_selection_score = float(
        best_evaluation.get("selection_score", 0.0) or 0.0
    )

    experience_evaluation = {
        "selection_score": selected_selection_score,
        "avg_matches": best_evaluation.get(
            "average_best_position_matches",
            0.0,
        ),
        "average_matches_per_ticket": best_evaluation.get(
            "average_position_matches_per_ticket",
            0.0,
        ),
        "hit_rate_2match": best_evaluation.get(
            "hit_rate_2_position",
            0.0,
        ),
        "hit_rate_3match": best_evaluation.get(
            "hit_rate_3_position",
            0.0,
        ),
        "hit_rate_4match": best_evaluation.get(
            "hit_rate_4_position",
            0.0,
        ),
        "avg_matches_std": 0.0,
        "random_uplift": (
            selected_selection_score - default_selection_score
        ),
    }

    return save_optimizer_experience(
        game_name=game_key,
        config_name=selected_config,
        config={
            "w": selected_weights,
            "f": {},
        },
        evaluation=experience_evaluation,
        prediction_weights=selected_weights,
        learning_strength=1.0,
    )


def run_numbers_game(
    df,
    game_config: dict,
) -> dict:
    history = dataframe_to_history(
        df,
        game_config,
    )

    return optimize_numbers(
        history,
        game_config,
    )


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
            optimizer_results[game_key][
                "experience_save"
            ] = save_numbers_optimizer_experience(
                game_key,
                optimizer_results[game_key],
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
            optimizer_results[game_key][
                "experience_save"
            ] = save_lotto_optimizer_experience(
                game_key,
                optimizer_results[game_key],
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

    previous_carryover_snapshot = load_json(
        OUTPUT_DIR / "carryover.json",
        {},
    )
    carryover_snapshot = fetch_carryover_snapshot(
        {
            game_key: game_output[game_key]["latest_draw_no"]
            for game_key in ("loto6", "loto7")
        },
        previous_snapshot=previous_carryover_snapshot,
    )

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
        "carryover": carryover_snapshot,
        **game_output,
    }

    save_json(
        OUTPUT_DIR / "optimizer_result.json",
        output,
    )
    save_json(
        OUTPUT_DIR / "carryover.json",
        carryover_snapshot,
    )

    save_prediction_outputs(
        OUTPUT_DIR,
        optimizer_results,
        LOTTO_GAMES,
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

    write_evaluation_dashboard(
        OUTPUT_DIR,
    )
    dashboard_json_path = (
        OUTPUT_DIR
        / "evaluation_dashboard.json"
    )
    dashboard_markdown_path = (
        OUTPUT_DIR
        / "evaluation_dashboard.md"
    )

    save_feature_memory_analysis()

    notification_summary_path = write_notification_summary(
        OUTPUT_DIR,
        output,
    )

    print("\n=== REVIEW OUTPUTS ===")
    print(f"run_summary: {run_summary_path}")
    print(f"review_bundle: {review_bundle_path}")
    print(
        f"notification_summary: {notification_summary_path}"
    )
    print(
        "evaluation_dashboard_json: "
        f"{dashboard_json_path}"
    )
    print(
        "evaluation_dashboard_markdown: "
        f"{dashboard_markdown_path}"
    )

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
