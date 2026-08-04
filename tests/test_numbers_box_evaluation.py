from __future__ import annotations

import unittest

import pandas as pd

from numbers_backtester import run_numbers_box_backtest
from run_pipeline import evaluate_previous_for_type


class NumbersBoxEvaluationTest(unittest.TestCase):

    def test_box_backtest_selects_unique_box_signatures(self) -> None:
        history = [
            ((index + 1) % 10, (index + 2) % 10, (index + 3) % 10)
            for index in range(80)
        ]
        config = {
            "digit_count": 3,
            "digit_min": 0,
            "digit_max": 9,
            "train_window": 20,
            "tested_periods": 10,
            "top_k": 10,
        }

        summary = run_numbers_box_backtest(
            history,
            config,
            include_records=True,
        )

        self.assertEqual(10, summary.tested_periods)
        self.assertIsNotNone(summary.box_hit_rate)
        for record in summary.records:
            signatures = [tuple(sorted(row)) for row in record.predicted_boxes]
            self.assertEqual(len(signatures), len(set(signatures)))

    def test_previous_result_evaluates_box_prediction_separately(self) -> None:
        previous = {
            "next_draw_no": 2,
            "prediction": [
                {"pattern_id": "P1", "numbers": [1, 2, 3]},
            ],
            "box_prediction": [
                {"pattern_id": "B1", "numbers": [1, 2, 3]},
                {"pattern_id": "B2", "numbers": [4, 5, 6]},
            ],
        }
        current = pd.DataFrame(
            [{"draw_no": 2, "n1": 3, "n2": 1, "n3": 2}]
        )

        evaluation = evaluate_previous_for_type(
            "numbers3",
            previous,
            current,
            ["n1", "n2", "n3"],
            family="numbers",
        )

        box = evaluation["box_prediction_evaluation"]
        self.assertEqual("evaluated", box["status"])
        self.assertTrue(box["box_hit"])
        self.assertEqual(3, box["best_unordered_match_count"])
        self.assertEqual(2, len(box["predictions"]))


if __name__ == "__main__":
    unittest.main()
