from pathlib import Path
import json
import os
from datetime import datetime, timezone

import numpy as np

from main import validate_lottery
from optimizer import load_data, optimize, print_result


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


def is_scheduled_no_new_data(
    previous_output: dict,
    loto6_val: dict,
    loto7_val: dict,
    miniloto_val: dict,
) -> bool:
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        return False

    if not previous_output:
        return False

    previous_loto6_latest = (
        previous_output.get("loto6", {}).get("latest_draw_no")
    )
    previous_loto7_latest = (
        previous_output.get("loto7", {}).get("latest_draw_no")
    )
    previous_miniloto_latest = (
        previous_output.get("miniloto", {}).get("latest_draw_no")
    )

    current_loto6_latest = loto6_val.get("latest_draw_no")
    current_loto7_latest = loto7_val.get("latest_draw_no")
    current_miniloto_latest = miniloto_val.get("latest_draw_no")

    return (
        previous_loto6_latest == current_loto6_latest
        and previous_loto7_latest == current_loto7_latest
        and previous_miniloto_latest == current_miniloto_latest
    )


def get_actual_numbers(
    df,
    draw_no: int,
    main_cols: list[str],
) -> list[int] | None:
    hit = df[df["draw_no"] == draw_no]

    if hit.empty:
        return None

    row = hit.iloc[0]
    return sorted(int(row[c]) for c in main_cols)


def evaluate_previous_for_type(
    draw_type: str,
    previous_section: dict | None,
    current_df,
    main_cols: list[str],
) -> dict:
    if not previous_section:
        return {
            "draw_type": draw_type,
            "status": "no_previous_output",
            "message": (
                "前回のoptimizer_result.jsonがないため、"
                "答え合わせ対象がありません。"
            ),
        }

    target_draw_no = previous_section.get("next_draw_no")
    previous_predictions = previous_section.get("prediction", [])

    if target_draw_no is None or not previous_predictions:
        return {
            "draw_type": draw_type,
            "status": "no_previous_prediction",
            "message": (
                "前回予想の回号または予想データが"
                "見つかりません。"
            ),
        }

    actual_numbers = get_actual_numbers(
        current_df,
        int(target_draw_no),
        main_cols,
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
            "message": (
                "まだ前回予想対象回の結果が"
                "データに反映されていません。"
            ),
        }

    actual_set = set(actual_numbers)
    evaluated_predictions = []

    for pred in previous_predictions:
        numbers = sorted(
            int(n)
            for n in pred.get("numbers", [])
        )
        matched_numbers = sorted(
            actual_set & set(numbers)
        )

        evaluated_predictions.append({
            "pattern_id": pred.get("pattern_id"),
            "numbers": numbers,
            "matches": len(matched_numbers),
            "matched_numbers": matched_numbers,
            "score": pred.get("score"),
            "model": pred.get("model"),
        })

    match_counts = [
        p["matches"]
        for p in evaluated_predictions
    ]

    return {
        "draw_type": draw_type,
        "status": "evaluated",
        "draw_no": int(target_draw_no),
        "actual_numbers": actual_numbers,
        "evaluated_at": now_iso(),
        "predictions": evaluated_predictions,
        "best_match_count": (
            int(max(match_counts))
            if match_counts
            else 0
        ),
        "avg_match_count": (
            round(float(np.mean(match_counts)), 4)
            if match_counts
            else 0.0
        ),
        "hit_rate_1match": (
            round(
                sum(m >= 1 for m in match_counts)
                / len(match_counts),
                4,
            )
            if match_counts
            else 0.0
        ),
        "hit_rate_2match": (
            round(
                sum(m >= 2 for m in match_counts)
                / len(match_counts),
                4,
            )
            if match_counts
            else 0.0
        ),
        "hit_rate_3match": (
            round(
                sum(m >= 3 for m in match_counts)
                / len(match_counts),
                4,
            )
            if match_counts
            else 0.0
        ),
        "hit_rate_4match": (
            round(
                sum(m >= 4 for m in match_counts)
                / len(match_counts),
                4,
            )
            if match_counts
            else 0.0
        ),
        "hit_rate_5match": (
            round(
                sum(m >= 5 for m in match_counts)
                / len(match_counts),
                4,
            )
            if match_counts
            else 0.0
        ),
    }


def merge_evaluation_history(
    existing_history: list[dict],
    new_evaluations: list[dict],
) -> list[dict]:
    merged = {}

    for item in existing_history:
        if item.get("status") != "evaluated":
            continue

        key = (
            item.get("draw_type"),
            item.get("draw_no"),
        )
        merged[key] = item

    for item in new_evaluations:
        if item.get("status") != "evaluated":
            continue

        key = (
            item.get("draw_type"),
            item.get("draw_no"),
        )
        merged[key] = item

    history = list(merged.values())
    history.sort(
        key=lambda x: (
            x.get("draw_type", ""),
            x.get("draw_no", 0),
        )
    )

    return history


def build_evaluation_summary(
    history: list[dict],
) -> dict:
    summary = {}

    for draw_type in [
        "loto6",
        "loto7",
        "miniloto",
    ]:
        items = [
            h
            for h in history
            if (
                h.get("draw_type") == draw_type
                and h.get("status") == "evaluated"
            )
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
            i["best_match_count"]
            for i in items
        ]
        avg_counts = [
            i["avg_match_count"]
            for i in items
        ]

        best_item = max(
            items,
            key=lambda x: x["best_match_count"],
        )

        summary[draw_type] = {
            "evaluated_draws": len(items),
            "avg_best_match_count": round(
                float(np.mean(best_counts)),
                4,
            ),
            "avg_all_pattern_matches": round(
                float(np.mean(avg_counts)),
                4,
            ),
            "max_best_match_count": int(
                max(best_counts)
            ),
            "best_draw_no": int(
                best_item["draw_no"]
            ),
            "latest_evaluated_draw_no": int(
                max(i["draw_no"] for i in items)
            ),
        }

    return summary


def print_evaluation(
    title: str,
    evaluation: dict,
) -> None:
    print(
        f"\n=== {title} "
        "PREVIOUS PREDICTION EVALUATION ==="
    )
    print(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    loto6, loto7, miniloto = load_data(
        include_miniloto=True
    )

    loto6_cols = [
        "main1",
        "main2",
        "main3",
        "main4",
        "main5",
        "main6",
    ]
    loto7_cols = [
        "main1",
        "main2",
        "main3",
        "main4",
        "main5",
        "main6",
        "main7",
    ]
    miniloto_cols = [
        "main1",
        "main2",
        "main3",
        "main4",
        "main5",
    ]

    loto6_val = validate_lottery(
        loto6,
        loto6_cols,
        ["bonus"],
        1,
        43,
    )
    loto7_val = validate_lottery(
        loto7,
        loto7_cols,
        ["bonus1", "bonus2"],
        1,
        37,
    )
    miniloto_val = validate_lottery(
        miniloto,
        miniloto_cols,
        ["bonus"],
        1,
        31,
    )

    if is_scheduled_no_new_data(
        previous_output,
        loto6_val,
        loto7_val,
        miniloto_val,
    ):
        print("\n=== NO NEW DATA ===")
        print(
            "Scheduled run detected, but latest "
            "draw numbers have not changed."
        )
        print(
            "LOTO6 latest_draw_no remains "
            f'{loto6_val["latest_draw_no"]}.'
        )
        print(
            "LOTO7 latest_draw_no remains "
            f'{loto7_val["latest_draw_no"]}.'
        )
        print(
            "MINILOTO latest_draw_no remains "
            f'{miniloto_val["latest_draw_no"]}.'
        )
        print("Output files were not rewritten.")
        return

    previous_evaluation_loto6 = (
        evaluate_previous_for_type(
            draw_type="loto6",
            previous_section=previous_output.get(
                "loto6"
            ),
            current_df=loto6,
            main_cols=loto6_cols,
        )
    )

    previous_evaluation_loto7 = (
        evaluate_previous_for_type(
            draw_type="loto7",
            previous_section=previous_output.get(
                "loto7"
            ),
            current_df=loto7,
            main_cols=loto7_cols,
        )
    )

    previous_evaluation_miniloto = (
        evaluate_previous_for_type(
            draw_type="miniloto",
            previous_section=previous_output.get(
                "miniloto"
            ),
            current_df=miniloto,
            main_cols=miniloto_cols,
        )
    )

    existing_history = load_json(
        history_path,
        [],
    )

    evaluation_history = (
        merge_evaluation_history(
            existing_history,
            [
                previous_evaluation_loto6,
                previous_evaluation_loto7,
                previous_evaluation_miniloto,
            ],
        )
    )

    evaluation_summary = (
        build_evaluation_summary(
            evaluation_history
        )
    )

    loto6_result = optimize(
        loto6,
        loto6_cols,
        1,
        43,
        6,
        500,
        45,
        300,
        10000,
    )
    loto7_result = optimize(
        loto7,
        loto7_cols,
        1,
        37,
        7,
        240,
        45,
        300,
        10000,
    )
    miniloto_result = optimize(
        miniloto,
        miniloto_cols,
        1,
        31,
        5,
        500,
        45,
        300,
        10000,
    )

    output = {
        "status": "ok",
        "note": (
            "run_pipeline evaluates the previous "
            "prediction if the target draw is now "
            "available, then creates the next prediction."
        ),
        "generated_at": now_iso(),
        "previous_evaluation": {
            "loto6": previous_evaluation_loto6,
            "loto7": previous_evaluation_loto7,
            "miniloto": (
                previous_evaluation_miniloto
            ),
        },
        "evaluation_summary": (
            evaluation_summary
        ),
        "loto6": {
            "latest_draw_no": (
                loto6_val["latest_draw_no"]
            ),
            "next_draw_no": (
                loto6_val["latest_draw_no"] + 1
            ),
            "rows": loto6_val["rows"],
            "validation": loto6_val,
            **loto6_result,
        },
        "loto7": {
            "latest_draw_no": (
                loto7_val["latest_draw_no"]
            ),
            "next_draw_no": (
                loto7_val["latest_draw_no"] + 1
            ),
            "rows": loto7_val["rows"],
            "validation": loto7_val,
            **loto7_result,
        },
        "miniloto": {
            "latest_draw_no": (
                miniloto_val["latest_draw_no"]
            ),
            "next_draw_no": (
                miniloto_val["latest_draw_no"]
                + 1
            ),
            "rows": miniloto_val["rows"],
            "validation": miniloto_val,
            **miniloto_result,
        },
    }

    save_json(
        OUTPUT_DIR / "optimizer_result.json",
        output,
    )
    save_json(
        OUTPUT_DIR
        / "prediction_optimizer_loto6.json",
        loto6_result["prediction"],
    )
    save_json(
        OUTPUT_DIR
        / "prediction_optimizer_loto7.json",
        loto7_result["prediction"],
    )
    save_json(
        OUTPUT_DIR
        / "prediction_optimizer_miniloto.json",
        miniloto_result["prediction"],
    )
    save_json(
        history_path,
        evaluation_history,
    )
    save_json(
        summary_path,
        evaluation_summary,
    )

    print_evaluation(
        "LOTO6",
        previous_evaluation_loto6,
    )
    print_result(
        "LOTO6",
        output["loto6"]["latest_draw_no"],
        output["loto6"]["next_draw_no"],
        loto6_result,
    )

    print_evaluation(
        "LOTO7",
        previous_evaluation_loto7,
    )
    print_result(
        "LOTO7",
        output["loto7"]["latest_draw_no"],
        output["loto7"]["next_draw_no"],
        loto7_result,
    )

    print_evaluation(
        "MINILOTO",
        previous_evaluation_miniloto,
    )
    print_result(
        "MINILOTO",
        output["miniloto"]["latest_draw_no"],
        output["miniloto"]["next_draw_no"],
        miniloto_result,
    )

    print("\n=== EVALUATION SUMMARY ===")
    print(
        json.dumps(
            evaluation_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== SHORT JSON ===")
    print(
        json.dumps(
            {
                "status": "ok",
                "loto6_previous_evaluation_status": (
                    previous_evaluation_loto6.get(
                        "status"
                    )
                ),
                "loto6_latest_draw_no": (
                    output["loto6"][
                        "latest_draw_no"
                    ]
                ),
                "loto6_next_draw_no": (
                    output["loto6"][
                        "next_draw_no"
                    ]
                ),
                "loto6_selected_config": (
                    loto6_result[
                        "selected_config"
                    ]
                ),
                "loto6_prediction": [
                    p["numbers"]
                    for p in loto6_result[
                        "prediction"
                    ]
                ],
                "loto7_previous_evaluation_status": (
                    previous_evaluation_loto7.get(
                        "status"
                    )
                ),
                "loto7_latest_draw_no": (
                    output["loto7"][
                        "latest_draw_no"
                    ]
                ),
                "loto7_next_draw_no": (
                    output["loto7"][
                        "next_draw_no"
                    ]
                ),
                "loto7_selected_config": (
                    loto7_result[
                        "selected_config"
                    ]
                ),
                "loto7_prediction": [
                    p["numbers"]
                    for p in loto7_result[
                        "prediction"
                    ]
                ],
                "miniloto_previous_evaluation_status": (
                    previous_evaluation_miniloto.get(
                        "status"
                    )
                ),
                "miniloto_latest_draw_no": (
                    output["miniloto"][
                        "latest_draw_no"
                    ]
                ),
                "miniloto_next_draw_no": (
                    output["miniloto"][
                        "next_draw_no"
                    ]
                ),
                "miniloto_selected_config": (
                    miniloto_result[
                        "selected_config"
                    ]
                ),
                "miniloto_prediction": [
                    p["numbers"]
                    for p in miniloto_result[
                        "prediction"
                    ]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()