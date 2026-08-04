from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optimizer_experience


def _entry(name: str, trained_through: int | None) -> dict:
    return {
        "evaluated_at": "2026-08-04T00:00:00+00:00",
        "config_name": name,
        "selection_score": 1.0,
        "avg_matches": 1.0,
        "average_matches_per_ticket": 0.5,
        "hit_rate_2match": 0.1,
        "hit_rate_3match": 0.0,
        "hit_rate_4match": 0.0,
        "avg_matches_std": 0.0,
        "random_uplift": 0.1,
        "learning_strength": 1.0,
        "trained_through_draw_no": trained_through,
        "config": {"w": {"freq": 1.0}, "f": {}},
        "prediction_weights": {"freq": 1.0},
    }


class ExperienceLegacyMigrationTest(unittest.TestCase):

    def test_load_moves_untracked_entries_to_legacy_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer_experience.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.4",
                        "games": {
                            "loto6": {
                                "history": [
                                    _entry("legacy", None),
                                    _entry("safe", 100),
                                ],
                                "latest": _entry("legacy", None),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(optimizer_experience, "EXPERIENCE_PATH", path):
                store = optimizer_experience._load_store()

            game = store["games"]["loto6"]
            self.assertEqual(["safe"], [x["config_name"] for x in game["history"]])
            self.assertEqual(
                ["legacy"],
                [x["config_name"] for x in game["legacy_history"]],
            )
            self.assertEqual(1, game["history_count"])
            self.assertEqual(1, game["legacy_history_count"])
            self.assertEqual("safe", game["latest"]["config_name"])

            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("1.5", persisted["schema_version"])

    def test_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer_experience.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.4",
                        "games": {
                            "numbers3": {
                                "history": [_entry("legacy", None)],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(optimizer_experience, "EXPERIENCE_PATH", path):
                first = optimizer_experience._load_store()
                second = optimizer_experience._load_store()

            self.assertEqual(first, second)
            legacy = second["games"]["numbers3"]["legacy_history"]
            self.assertEqual(1, len(legacy))

    def test_save_preserves_legacy_but_uses_only_tracked_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer_experience.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.4",
                        "games": {
                            "loto7": {
                                "history": [_entry("legacy", None)],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(optimizer_experience, "EXPERIENCE_PATH", path):
                optimizer_experience.save_optimizer_experience(
                    game_name="loto7",
                    config_name="new_safe",
                    config={"w": {"freq": 1.0}, "f": {}},
                    evaluation={"selection_score": 2.0},
                    prediction_weights={"freq": 1.0},
                    learning_strength=1.0,
                    trained_through_draw_no=200,
                )
                configs = optimizer_experience.load_experience_configs("loto7")

            persisted = json.loads(path.read_text(encoding="utf-8"))
            game = persisted["games"]["loto7"]
            self.assertEqual(1, len(game["history"]))
            self.assertEqual(1, len(game["legacy_history"]))
            self.assertEqual("new_safe", game["history"][0]["config_name"])
            self.assertEqual("legacy", game["legacy_history"][0]["config_name"])
            self.assertEqual(1, len(configs))
            self.assertEqual("experience_loto7_1", configs[0]["name"])
            self.assertEqual(200, configs[0]["trained_through_draw_no"])


if __name__ == "__main__":
    unittest.main()
