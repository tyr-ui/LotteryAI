from __future__ import annotations

import unittest
from unittest import mock

import optimizer_experience


class ExperienceTrainingProvenanceTest(unittest.TestCase):

    @staticmethod
    def _entry(name: str, trained_through_draw_no: int | None) -> dict:
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
            "trained_through_draw_no": trained_through_draw_no,
            "config": {
                "w": {"freq": 1.0 if name == "safe" else 0.5},
                "f": {},
            },
            "prediction_weights": {},
        }

    def test_cutoff_excludes_future_and_untracked_entries(self):
        store = {
            "schema_version": optimizer_experience.SCHEMA_VERSION,
            "games": {
                "loto6": {
                    "history": [
                        self._entry("safe", 100),
                        self._entry("future", 130),
                        self._entry("legacy", None),
                    ]
                }
            },
        }

        with mock.patch.object(
            optimizer_experience,
            "_load_store",
            return_value=store,
        ):
            configs = optimizer_experience.load_experience_configs(
                "loto6",
                limit=10,
                max_trained_through_draw_no=110,
            )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["w"], {"freq": 1.0})

    def test_no_cutoff_keeps_legacy_entries_for_normal_operation(self):
        store = {
            "schema_version": optimizer_experience.SCHEMA_VERSION,
            "games": {
                "loto6": {
                    "history": [self._entry("legacy", None)]
                }
            },
        }

        with mock.patch.object(
            optimizer_experience,
            "_load_store",
            return_value=store,
        ):
            configs = optimizer_experience.load_experience_configs(
                "loto6",
                limit=10,
            )

        self.assertEqual(len(configs), 1)

    def test_save_records_training_boundary(self):
        empty_store = {
            "schema_version": optimizer_experience.SCHEMA_VERSION,
            "games": {},
        }
        captured = {}

        def fake_save(path, output):
            captured.update(output)

        with (
            mock.patch.object(
                optimizer_experience,
                "_load_store",
                return_value=empty_store,
            ),
            mock.patch.object(
                optimizer_experience,
                "save_experience_store",
                side_effect=fake_save,
            ),
        ):
            optimizer_experience.save_optimizer_experience(
                game_name="loto6",
                config_name="test",
                config={"w": {"freq": 1.0}, "f": {}},
                evaluation={"selection_score": 1.0},
                prediction_weights={},
                learning_strength=1.0,
                trained_through_draw_no=2095,
            )

        latest = captured["games"]["loto6"]["latest"]
        self.assertEqual(latest["trained_through_draw_no"], 2095)


if __name__ == "__main__":
    unittest.main()
