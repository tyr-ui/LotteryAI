import unittest
from collections import Counter

import pandas as pd

from numbers_backtester import (
    box_composition_signature,
    run_numbers_composition_matched_box_random_backtest,
)
from operational_controls import build_operational_controls


class CompositionMatchedControlTest(unittest.TestCase):
    def test_backtest_matches_model_composition_per_draw(self):
        history = [
            (i % 10, (i + 1) % 10, (i + 2) % 10)
            for i in range(20)
        ]
        model_records = [
            {
                "draw_index": 18,
                "box_dedicated_predicted": [(1, 2, 3), (4, 4, 5)],
            },
            {
                "draw_index": 19,
                "box_dedicated_predicted": [(1, 1, 1), (2, 3, 4)],
            },
        ]
        summary = run_numbers_composition_matched_box_random_backtest(
            history,
            {"digit_count": 3},
            model_records=model_records,
            seed=10,
            include_records=True,
        )
        self.assertEqual(summary.tested_periods, 2)
        for source, record in zip(model_records, summary.records):
            expected = Counter(
                box_composition_signature(row)
                for row in source["box_dedicated_predicted"]
            )
            actual = Counter(
                box_composition_signature(row)
                for row in record.predicted_boxes
            )
            self.assertEqual(actual, expected)

    def test_operational_numbers_controls_are_fixed_before_draw(self):
        df = pd.DataFrame({"draw_no": [1], "d1": [1], "d2": [2], "d3": [3]})
        result = {
            "prediction": [{"numbers": [1, 2, 3]}],
            "box_prediction": [
                {"numbers": [1, 2, 3]},
                {"numbers": [4, 4, 5]},
            ],
        }
        control = build_operational_controls(
            "numbers3",
            {"family": "numbers", "digit_count": 3},
            df,
            result,
            2,
            "2026-08-05T00:00:00+00:00",
        )
        self.assertTrue(control["generated_before_draw"])
        self.assertEqual(control["target_draw_no"], 2)
        expected = Counter(
            box_composition_signature(row["numbers"])
            for row in result["box_prediction"]
        )
        actual = Counter(
            box_composition_signature(row)
            for row in control["composition_matched_random_box_control"]
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
