from __future__ import annotations

import unittest

from statistical_evaluation import (
    build_game_statistical_report,
    data_volume_label,
    operational_evaluation_info,
    paired_difference_summary,
)


class StatisticalEvaluationTest(unittest.TestCase):
    def test_paired_summary_uses_same_draw_differences(self):
        rows = [
            {"model": 3, "random": 2},
            {"model": 1, "random": 2},
            {"model": 2, "random": 2},
        ]
        result = paired_difference_summary(
            rows,
            model_key="model",
            baseline_key="random",
        )
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 1, 1))
        self.assertEqual(result["judgement"], "現時点では差を特定できない")

    def test_positive_interval_has_limited_wording(self):
        rows = [{"model": 2, "random": 1} for _ in range(30)]
        result = paired_difference_summary(rows, model_key="model", baseline_key="random")
        self.assertEqual(result["judgement"], "今回の評価期間では正の差を検出")
        self.assertGreater(result["confidence_interval_95"]["lower"], 0)

    def test_data_volume_is_not_a_hard_validity_boundary(self):
        self.assertEqual(data_volume_label(12), "標本数が少ない（12回）")
        self.assertEqual(data_volume_label(30), "暫定評価（30回）")
        self.assertEqual(data_volume_label(100), "継続評価可能（100回）")

    def test_operational_period_uses_evaluated_rows(self):
        history = [
            {"draw_type": "loto6", "status": "evaluated", "evaluated_at": "2026-07-15T10:00:00+00:00"},
            {"draw_type": "loto6", "status": "evaluated", "evaluated_at": "2026-08-01T10:00:00+00:00"},
            {"draw_type": "loto7", "status": "evaluated", "evaluated_at": "2026-07-01T10:00:00+00:00"},
        ]
        result = operational_evaluation_info(history, "loto6")
        self.assertEqual(result["evaluated_draws"], 2)
        self.assertEqual(result["started_at"], "2026-07-15")
        self.assertEqual(result["latest_evaluated_at"], "2026-08-01")

    def test_lotto_report_reads_production_holdout_pairs(self):
        section = {
            "final_candidate_holdout": {
                "paired_draw_results": [
                    {"model_best_match_count": 3, "uniform_best_match_count": 2},
                    {"model_best_match_count": 1, "uniform_best_match_count": 1},
                ]
            }
        }
        result = build_game_statistical_report("loto6", section, [])
        self.assertEqual(result["baseline"], "一様ランダム")
        self.assertEqual(result["paired_evaluation"]["sample_size"], 2)

    def test_numbers_report_reads_box_dedicated_pairs(self):
        section = {
            "holdout_evaluation": {
                "box_dedicated_evaluation": {
                    "paired_draw_results": [
                        {"model_box_hit": True, "random_box_hit": False},
                        {"model_box_hit": False, "random_box_hit": False},
                    ]
                }
            }
        }
        result = build_game_statistical_report("numbers3", section, [])
        self.assertEqual(result["baseline"], "BOX専用ランダム")
        self.assertEqual(result["paired_evaluation"]["mean_difference"], 0.5)

    def test_permutation_p_value_targets_mean_and_sign_test_targets_wins(self):
        rows = [
            {"model": 11, "random": 1},
            *[{"model": 0, "random": 1} for _ in range(9)],
        ]
        result = paired_difference_summary(
            rows,
            model_key="model",
            baseline_key="random",
        )
        self.assertIn("permutation_p_value_reference", result)
        self.assertIn("sign_test_p_value_reference", result)
        self.assertEqual(
            result["permutation_p_value_reference"]["target"],
            "mean_difference",
        )
        self.assertEqual(
            result["sign_test_p_value_reference"]["target"],
            "win_loss_imbalance",
        )
        self.assertNotEqual(
            result["permutation_p_value_reference"]["value"],
            result["sign_test_p_value_reference"]["value"],
        )

    def test_numbers_report_uses_multi_seed_mean_baseline(self):
        section = {
            "holdout_evaluation": {
                "box_dedicated_evaluation": {
                    "paired_draw_results": [
                        {
                            "model_box_hit": 1.0,
                            "random_box_hit_mean": 1 / 3,
                        },
                        {
                            "model_box_hit": 0.0,
                            "random_box_hit_mean": 0.0,
                        },
                    ]
                }
            }
        }
        result = build_game_statistical_report("numbers3", section, [])
        self.assertAlmostEqual(
            result["paired_evaluation"]["mean_difference"],
            1 / 3,
            places=5,
        )

    def test_non_finite_values_are_ignored(self):
        rows = [
            {"model": float("nan"), "random": 0},
            {"model": float("inf"), "random": 0},
            {"model": True, "random": False},
        ]
        result = paired_difference_summary(
            rows,
            model_key="model",
            baseline_key="random",
        )
        self.assertEqual(result["sample_size"], 1)
        self.assertEqual(result["mean_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
