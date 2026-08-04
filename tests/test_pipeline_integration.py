from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import run_pipeline


@dataclass
class FakeLoadedGame:
    dataframe: pd.DataFrame
    validation: dict
    source: str = "test"


def _game_configs() -> dict[str, dict]:
    return {
        "loto6": {
            "family": "lotto",
            "display_name": "LOTO6",
            "main_cols": ["n1", "n2"],
            "prediction_filename": "prediction_loto6.json",
            "min_num": 1,
            "max_num": 43,
            "pick_count": 2,
            "train_window": 10,
            "tested_periods": 5,
            "backtest_candidates": 10,
            "final_candidates": 10,
        },
        "loto7": {
            "family": "lotto",
            "display_name": "LOTO7",
            "main_cols": ["n1", "n2"],
            "prediction_filename": "prediction_loto7.json",
            "min_num": 1,
            "max_num": 37,
            "pick_count": 2,
            "train_window": 10,
            "tested_periods": 5,
            "backtest_candidates": 10,
            "final_candidates": 10,
        },
        "miniloto": {
            "family": "lotto",
            "display_name": "ミニロト",
            "main_cols": ["n1", "n2"],
            "prediction_filename": "prediction_miniloto.json",
            "min_num": 1,
            "max_num": 31,
            "pick_count": 2,
            "train_window": 10,
            "tested_periods": 5,
            "backtest_candidates": 10,
            "final_candidates": 10,
        },
        "numbers3": {
            "family": "numbers",
            "display_name": "Numbers3",
            "main_cols": ["n1", "n2", "n3"],
            "prediction_filename": "prediction_numbers3.json",
        },
        "numbers4": {
            "family": "numbers",
            "display_name": "Numbers4",
            "main_cols": ["n1", "n2", "n3", "n4"],
            "prediction_filename": "prediction_numbers4.json",
        },
    }


def _dataframe(game_key: str, latest_draw_no: int) -> pd.DataFrame:
    width = 3 if game_key == "numbers3" else 4 if game_key == "numbers4" else 2

    rows = []
    for draw_no in range(1, latest_draw_no + 1):
        row = {"draw_no": draw_no}
        for index in range(1, width + 1):
            if game_key.startswith("numbers"):
                row[f"n{index}"] = (draw_no + index) % 10
            else:
                row[f"n{index}"] = draw_no + index
        rows.append(row)

    return pd.DataFrame(rows)


def _optimizer_result(game_key: str, latest_draw_no: int) -> dict:
    if game_key.startswith("numbers"):
        numbers = [1, 2, 3] if game_key == "numbers3" else [1, 2, 3, 4]
    else:
        numbers = [1, 2]

    return {
        "selected_config": f"config_{game_key}_{latest_draw_no}",
        "selected_weights": {"freq": 1.0},
        "selected_filters": {},
        "ranked_configs": [
            {
                "config": f"config_{game_key}_{latest_draw_no}",
                "selection_score": 1.0,
            }
        ],
        "prediction": [
            {
                "pattern_id": "P1",
                "numbers": numbers,
                "score": 1.0,
                "model": f"config_{game_key}_{latest_draw_no}",
            }
        ],
        "box_prediction": [],
        "random_baseline": {},
        "trained_through_draw_no": latest_draw_no,
    }


def _previous_section(game_key: str, latest_draw_no: int) -> dict:
    result = _optimizer_result(game_key, latest_draw_no)
    return {
        "latest_draw_no": latest_draw_no,
        "next_draw_no": latest_draw_no + 1,
        "rows": latest_draw_no,
        "validation": {
            "status": "ok",
            "rows": latest_draw_no,
            "latest_draw_no": latest_draw_no,
        },
        **result,
    }


class PipelineIntegrationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.game_configs = _game_configs()

    def _run_main(
        self,
        *,
        output_dir: Path,
        latest_draws: dict[str, int],
        previous_output: dict,
        run_mode: str = "auto",
    ) -> tuple[Mock, Mock]:
        loaded_by_game = {
            game_key: FakeLoadedGame(
                dataframe=_dataframe(game_key, latest_draws[game_key]),
                validation={
                    "status": "ok",
                    "rows": latest_draws[game_key],
                    "latest_draw_no": latest_draws[game_key],
                },
            )
            for game_key in self.game_configs
        }

        optimizer_calls = Mock()
        notification_calls = Mock()

        def fake_load_game_data(game_key, game_config, destination):
            return loaded_by_game[game_key]

        def fake_run_all_optimizers(datasets, game_configs, max_workers=None):
            optimizer_calls(list(game_configs.keys()))
            return {
                game_key: _optimizer_result(
                    game_key,
                    latest_draws[game_key],
                )
                for game_key in game_configs
            }

        def fake_save_json(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        def fake_load_json(path, default):
            path = Path(path)
            if path.name == "optimizer_result.json":
                return previous_output
            if path.name == "evaluation_history.json":
                return []
            if path.name == "carryover.json":
                return {
                    "games": {
                        "loto6": {"status": "none"},
                        "loto7": {"status": "none"},
                    }
                }
            return default

        def fake_write_review_outputs(output_dir, output, game_keys):
            first = Path(output_dir) / "run_summary.json"
            second = Path(output_dir) / "review_bundle.json"
            fake_save_json(first, {})
            fake_save_json(second, {})
            return first, second

        def fake_write_notification_summary(output_dir, output):
            notification_calls(output)
            path = Path(output_dir) / "notification_summary.md"
            path.write_text("test", encoding="utf-8")
            return path

        with (
            patch.object(run_pipeline, "ROOT", output_dir.parent),
            patch.object(run_pipeline, "OUTPUT_DIR", output_dir),
            patch.object(run_pipeline, "LOTTO_GAMES", self.game_configs),
            patch.object(run_pipeline, "resolve_run_mode", return_value=run_mode),
            patch.object(run_pipeline, "load_game_data", side_effect=fake_load_game_data),
            patch.object(run_pipeline, "load_json", side_effect=fake_load_json),
            patch.object(run_pipeline, "save_json", side_effect=fake_save_json),
            patch.object(
                run_pipeline,
                "run_all_optimizers",
                side_effect=fake_run_all_optimizers,
            ),
            patch.object(
                run_pipeline,
                "save_lotto_optimizer_experience",
                return_value={"status": "saved"},
            ),
            patch.object(
                run_pipeline,
                "save_numbers_optimizer_experience",
                return_value={"status": "saved"},
            ),
            patch.object(
                run_pipeline,
                "evaluate_previous_for_type",
                side_effect=lambda draw_type, **kwargs: {
                    "draw_type": draw_type,
                    "status": "evaluated",
                    "draw_no": latest_draws[draw_type],
                    "best_match_count": 0,
                    "avg_match_count": 0.0,
                },
            ),
            patch.object(
                run_pipeline,
                "fetch_carryover_snapshot",
                return_value={
                    "games": {
                        "loto6": {"status": "none"},
                        "loto7": {"status": "none"},
                    }
                },
            ),
            patch.object(
                run_pipeline,
                "write_review_outputs",
                side_effect=fake_write_review_outputs,
            ),
            patch.object(run_pipeline, "write_evaluation_dashboard"),
            patch.object(run_pipeline, "save_feature_memory_analysis"),
            patch.object(
                run_pipeline,
                "write_notification_summary",
                side_effect=fake_write_notification_summary,
            ),
            patch.object(run_pipeline, "print_learning_weights"),
            patch.object(run_pipeline, "print_evaluation"),
            patch.object(run_pipeline, "print_result"),
            patch.object(run_pipeline, "save_prediction_outputs"),
        ):
            run_pipeline.main()

        return optimizer_calls, notification_calls

    def test_first_run_optimizes_all_five_games(self) -> None:
        latest_draws = {
            "loto6": 10,
            "loto7": 20,
            "miniloto": 30,
            "numbers3": 40,
            "numbers4": 50,
        }

        with tempfile.TemporaryDirectory() as directory:
            optimizer_calls, notification_calls = self._run_main(
                output_dir=Path(directory) / "output",
                latest_draws=latest_draws,
                previous_output={},
            )

        optimizer_calls.assert_called_once_with(
            ["loto6", "loto7", "miniloto", "numbers3", "numbers4"]
        )
        self.assertEqual(notification_calls.call_count, 1)

    def test_second_run_optimizes_only_updated_numbers_games(self) -> None:
        previous_draws = {
            "loto6": 10,
            "loto7": 20,
            "miniloto": 30,
            "numbers3": 40,
            "numbers4": 50,
        }
        latest_draws = {
            **previous_draws,
            "numbers3": 41,
            "numbers4": 51,
        }

        previous_output = {
            "previous_evaluation": {},
            **{
                game_key: _previous_section(game_key, draw_no)
                for game_key, draw_no in previous_draws.items()
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            optimizer_calls, notification_calls = self._run_main(
                output_dir=Path(directory) / "output",
                latest_draws=latest_draws,
                previous_output=previous_output,
            )

        optimizer_calls.assert_called_once_with(["numbers3", "numbers4"])
        self.assertEqual(notification_calls.call_count, 1)

        output = notification_calls.call_args.args[0]
        self.assertEqual(
            output["run_metadata"]["optimized_games"],
            ["numbers3", "numbers4"],
        )
        self.assertEqual(
            output["run_metadata"]["reused_games"],
            ["loto6", "loto7", "miniloto"],
        )
        self.assertEqual(output["loto6"]["latest_draw_no"], 10)
        self.assertEqual(output["numbers3"]["latest_draw_no"], 41)

    def test_third_run_with_no_changes_skips_optimizers_and_outputs(self) -> None:
        latest_draws = {
            "loto6": 10,
            "loto7": 20,
            "miniloto": 30,
            "numbers3": 40,
            "numbers4": 50,
        }

        previous_output = {
            "previous_evaluation": {},
            **{
                game_key: _previous_section(game_key, draw_no)
                for game_key, draw_no in latest_draws.items()
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            optimizer_calls, notification_calls = self._run_main(
                output_dir=Path(directory) / "output",
                latest_draws=latest_draws,
                previous_output=previous_output,
            )

        optimizer_calls.assert_not_called()
        notification_calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
