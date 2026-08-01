from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from games import LOTTO_GAMES
from run_pipeline import save_prediction_outputs


class PredictionOutputContractTest(unittest.TestCase):
    def test_all_required_prediction_files_are_written(self) -> None:
        optimizer_results: dict[str, dict] = {}

        for game_key, game_config in LOTTO_GAMES.items():
            result = {
                "prediction": [
                    {
                        "pattern_id": "P1",
                        "numbers": [1],
                    }
                ]
            }

            if str(game_config.get("family", "lotto")).lower() == "numbers":
                result["box_prediction"] = [
                    {
                        "pattern_id": "B1",
                        "numbers": [1],
                    }
                ]

            optimizer_results[game_key] = result

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            save_prediction_outputs(
                output_dir,
                optimizer_results,
                LOTTO_GAMES,
            )

            expected_files = {
                str(config["prediction_filename"])
                for config in LOTTO_GAMES.values()
            }
            expected_files.update({
                "prediction_box_numbers3.json",
                "prediction_box_numbers4.json",
            })

            actual_files = {
                path.name
                for path in output_dir.glob("*.json")
            }

            self.assertEqual(
                expected_files,
                actual_files,
            )

            for game_key, game_config in LOTTO_GAMES.items():
                prediction_path = (
                    output_dir
                    / str(game_config["prediction_filename"])
                )
                prediction_data = json.loads(
                    prediction_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    prediction_data[0]["pattern_id"],
                    "P1",
                )

                family = str(
                    game_config.get("family", "lotto")
                ).lower()
                box_path = (
                    output_dir
                    / f"prediction_box_{game_key}.json"
                )

                if family == "numbers":
                    self.assertTrue(box_path.exists())
                    box_data = json.loads(
                        box_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        box_data[0]["pattern_id"],
                        "B1",
                    )
                else:
                    self.assertFalse(box_path.exists())


if __name__ == "__main__":
    unittest.main()
