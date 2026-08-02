import unittest
from unittest.mock import patch

from optimizer_ablation import (
    ABLATION_FEATURES,
    _build_ablated_config,
    _calculate_drop,
    run_feature_ablation,
)


class OptimizerAblationTest(unittest.TestCase):

    def setUp(self):
        self.config = {
            "name": "selected_config",
            "w": {
                "freq": 0.20,
                "recent": 0.18,
                "pair": 0.16,
                "triplet": 0.10,
                "delay": 0.12,
                "dist": 0.14,
                "repeat": 0.10,
            },
            "s": {"g": 0.4, "r": 0.4, "d": 0.2},
            "f": {"max_block": 3},
        }

    def test_build_ablated_config_does_not_modify_original(self):
        original_weights = dict(self.config["w"])

        ablated = _build_ablated_config(
            self.config,
            feature="freq",
        )

        self.assertEqual(self.config["w"], original_weights)
        self.assertEqual(ablated["w"]["freq"], 0.0)
        self.assertEqual(ablated["parent"], "selected_config")
        self.assertEqual(ablated["ablated_feature"], "freq")
        self.assertEqual(ablated["search_origin"], "ablation")

    def test_build_ablated_config_rejects_unknown_feature(self):
        with self.assertRaises(ValueError):
            _build_ablated_config(
                self.config,
                feature="unknown",
            )

    def test_calculate_drop_preserves_direction(self):
        self.assertEqual(
            _calculate_drop(2.0, 1.5),
            (0.5, 25.0),
        )
        self.assertEqual(
            _calculate_drop(2.0, 2.5),
            (-0.5, -25.0),
        )
        self.assertEqual(
            _calculate_drop(0.0, 1.0),
            (-1.0, 0.0),
        )

    @patch("optimizer_ablation.evaluate_config")
    def test_run_feature_ablation_evaluates_all_features(
        self,
        mock_evaluate,
    ):
        feature_penalty = {
            feature: (index + 1) / 100.0
            for index, feature in enumerate(ABLATION_FEATURES)
        }

        def fake_evaluate(
            history,
            game_config,
            optimizer_config,
            **kwargs,
        ):
            feature = optimizer_config.get("ablated_feature")
            penalty = feature_penalty.get(feature, 0.0)

            return {
                "config": optimizer_config["name"],
                "avg_matches": 2.0 - penalty,
                "selection_score": 3.0 - (penalty * 2.0),
                "weights": dict(optimizer_config["w"]),
                "tested_periods": kwargs["tested_periods"],
                "evaluated_seeds": len(kwargs["seeds"]),
            }

        mock_evaluate.side_effect = fake_evaluate

        results = run_feature_ablation(
            history=[(1, 2, 3, 4, 5, 6)] * 30,
            game_config={
                "key": "loto6",
                "pick_count": 6,
                "min_num": 1,
                "max_num": 43,
            },
            optimizer_config=self.config,
            baseline_result={},
            train_window=20,
            tested_periods=5,
            candidate_count=50,
            seeds=(2025, 2026),
            random_baselines={},
        )

        self.assertEqual(len(results), len(ABLATION_FEATURES))
        self.assertEqual(
            mock_evaluate.call_count,
            1 + len(ABLATION_FEATURES),
        )
        self.assertEqual(
            {row["feature"] for row in results},
            set(ABLATION_FEATURES),
        )
        self.assertEqual(
            [row["rank"] for row in results],
            list(range(1, len(results) + 1)),
        )
        self.assertEqual(results[0]["feature"], "repeat")
        self.assertGreater(
            results[0]["selection_score_drop"],
            0.0,
        )

    @patch("optimizer_ablation.evaluate_config")
    def test_zero_weight_feature_reuses_baseline(
        self,
        mock_evaluate,
    ):
        config = {
            **self.config,
            "w": {
                **self.config["w"],
                "delay": 0.0,
            },
        }

        mock_evaluate.return_value = {
            "config": "selected_config",
            "avg_matches": 2.0,
            "selection_score": 3.0,
            "weights": dict(config["w"]),
            "tested_periods": 5,
            "evaluated_seeds": 2,
        }

        results = run_feature_ablation(
            history=[(1, 2, 3, 4, 5, 6)] * 30,
            game_config={"key": "loto6"},
            optimizer_config=config,
            baseline_result={},
            train_window=20,
            tested_periods=5,
            candidate_count=50,
            seeds=(2025, 2026),
            random_baselines={},
        )

        self.assertEqual(mock_evaluate.call_count, 7)

        delay_result = next(
            row for row in results
            if row["feature"] == "delay"
        )
        self.assertFalse(delay_result["active"])
        self.assertEqual(delay_result["avg_matches_drop"], 0.0)
        self.assertEqual(delay_result["selection_score_drop"], 0.0)


if __name__ == "__main__":
    unittest.main()
