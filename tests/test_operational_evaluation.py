import unittest

import pandas as pd

from operational_evaluation import evaluate_operational_controls


class OperationalEvaluationTest(unittest.TestCase):
    def test_evaluates_lotto_controls_once(self):
        controls = {"games": {"loto6": {
            "target_draw_no": 2,
            "evaluation_epoch": 3,
            "model_version": "epoch-3",
            "generated_at": "2026-08-05T00:00:00+00:00",
            "generated_before_draw": True,
            "model_prediction": [{"numbers": [1, 2, 3, 4, 5, 6]}],
            "uniform_random_control": [[1, 2, 3, 7, 8, 9]],
            "filtered_random_control": [[10, 11, 12, 13, 14, 15]],
        }}}
        df = pd.DataFrame({
            "draw_no": [2], "n1": [1], "n2": [2], "n3": [3],
            "n4": [4], "n5": [5], "n6": [6],
        })
        configs = {"loto6": {"family": "lotto", "main_cols": [f"n{i}" for i in range(1, 7)]}}
        history = evaluate_operational_controls(controls, {"loto6": df}, configs, [])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["model"]["best_match_count"], 6)
        self.assertEqual(history[0]["model_minus_uniform_best"], 3)
        again = evaluate_operational_controls(controls, {"loto6": df}, configs, history)
        self.assertEqual(len(again), 1)

    def test_evaluates_numbers_box_control(self):
        controls = {"games": {"numbers3": {
            "target_draw_no": 8,
            "evaluation_epoch": 1,
            "generated_before_draw": True,
            "model_prediction": [{"numbers": [1, 2, 3]}],
            "uniform_random_control": [[1, 2, 4]],
            "model_box_prediction": [{"numbers": [3, 1, 2]}],
            "composition_matched_random_box_control": [[1, 1, 2]],
        }}}
        df = pd.DataFrame({"draw_no": [8], "d1": [1], "d2": [2], "d3": [3]})
        configs = {"numbers3": {"family": "numbers", "main_cols": ["d1", "d2", "d3"]}}
        history = evaluate_operational_controls(controls, {"numbers3": df}, configs, [])
        self.assertTrue(history[0]["model"]["straight_hit"])
        self.assertTrue(history[0]["model_box"]["box_hit"])
        self.assertEqual(history[0]["model_box_minus_control_hit"], 1)


if __name__ == "__main__":
    unittest.main()
