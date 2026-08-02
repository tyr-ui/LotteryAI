from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from common import now_iso


DEFAULT_TOP_CONFIGS = 5


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prediction_summary(prediction: list[dict]) -> list[dict]:
    return [
        {
            "pattern_id": item.get("pattern_id"),
            "numbers": item.get("numbers", []),
            "score": item.get("score"),
            "model": item.get("model"),
        }
        for item in prediction
    ]


def _ranked_config_summary(item: dict) -> dict:
    keep_keys = (
        "config",
        "tested_periods",
        "evaluated_seeds",
        "avg_matches",
        "average_matches_per_ticket",
        "hit_rate_1match",
        "hit_rate_2match",
        "hit_rate_3match",
        "hit_rate_4match",
        "hit_rate_5match",
        "hit_rate_6match",
        "hit_rate_7match",
        "avg_matches_std",
        "seed_avg_matches",
        "selection_score",
        "random_unfiltered_avg",
        "random_filtered_avg",
        "random_uplift",
        "weights",
        "filters",
        "search_origin",
        "parent",
    )
    return {key: item.get(key) for key in keep_keys if key in item}


def build_run_summary(
    output: dict,
    game_keys: list[str],
) -> dict:
    games = {}

    for game_key in game_keys:
        section = output.get(game_key, {})
        games[game_key] = {
            "latest_draw_no": section.get("latest_draw_no"),
            "next_draw_no": section.get("next_draw_no"),
            "rows": section.get("rows"),
            "selected_config": section.get("selected_config"),
            "selected_weights": section.get("selected_weights"),
            "selected_filters": section.get("selected_filters"),
            "feature_ablation": section.get(
                "feature_ablation"
            ),
            "prediction": _prediction_summary(
                section.get("prediction", [])
            ),
            "previous_evaluation": output
            .get("previous_evaluation", {})
            .get(game_key),
            "evaluation_summary": output
            .get("evaluation_summary", {})
            .get(game_key),
        }

    return {
        "schema_version": "1.0",
        "status": output.get("status"),
        "generated_at": output.get("generated_at"),
        "summary_generated_at": now_iso(),
        "games": games,
    }


def build_review_bundle(
    output: dict,
    game_keys: list[str],
    top_configs: int = DEFAULT_TOP_CONFIGS,
) -> dict:
    games = {}

    for game_key in game_keys:
        section = output.get(game_key, {})
        ranked_configs = section.get("ranked_configs", [])

        games[game_key] = {
            "latest_draw_no": section.get("latest_draw_no"),
            "next_draw_no": section.get("next_draw_no"),
            "rows": section.get("rows"),
            "validation": section.get("validation"),
            "selected_config": section.get("selected_config"),
            "selected_weights": section.get("selected_weights"),
            "selected_filters": section.get("selected_filters"),
            "learning_summary": section.get(
                "learning_summary"
            ),
            "search_metadata": section.get("search_metadata"),
            "feature_ablation": section.get(
                "feature_ablation"
            ),
            "prediction": _prediction_summary(
                section.get("prediction", [])
            ),
            "random_baseline": section.get("random_baseline"),
            "selected_random_filtered_baseline": section.get(
                "selected_random_filtered_baseline"
            ),
            "top_ranked_configs": [
                _ranked_config_summary(item)
                for item in ranked_configs[:top_configs]
            ],
            "ranked_config_count": len(ranked_configs),
            "previous_evaluation": output
            .get("previous_evaluation", {})
            .get(game_key),
            "evaluation_summary": output
            .get("evaluation_summary", {})
            .get(game_key),
        }

    return {
        "schema_version": "1.0",
        "status": output.get("status"),
        "note": output.get("note"),
        "generated_at": output.get("generated_at"),
        "bundle_generated_at": now_iso(),
        "top_configs_per_game": top_configs,
        "games": games,
    }


def write_review_outputs(
    output_dir: Path,
    output: dict,
    game_keys: list[str],
    top_configs: int = DEFAULT_TOP_CONFIGS,
) -> tuple[Path, Path]:
    run_summary_path = output_dir / "run_summary.json"
    review_bundle_path = output_dir / "review_bundle.json"

    save_json(
        run_summary_path,
        build_run_summary(output, game_keys),
    )
    save_json(
        review_bundle_path,
        build_review_bundle(
            output,
            game_keys,
            top_configs=top_configs,
        ),
    )

    return run_summary_path, review_bundle_path
