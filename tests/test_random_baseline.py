import unittest
from unittest.mock import patch

from backtester import (
    run_filtered_random_backtest,
    run_uniform_random_backtest,
)
from numbers_backtester import run_numbers_uniform_random_backtest


class CombinationRandomBaselineTest(unittest.TestCase):

    def setUp(self):
        self.history = [
            tuple(range(start, start + 6))
            for start in range(1, 31)
        ]
        self.config = {
            "min_num": 1,
            "max_num": 43,
            "pick_count": 6,
            "train_window": 20,
            "tested_periods": 5,
            "backtest_candidates": 30,
        }

    @patch("backtester.predict")
    def test_uniform_random_does_not_use_model_ranking(self, mock_predict):
        summary = run_uniform_random_backtest(
            self.history,
            self.config,
            top_k=5,
            seed=2025,
            include_records=True,
        )

        mock_predict.assert_not_called()
        self.assertEqual(summary.tested_periods, 5)
        for record in summary.records:
            self.assertEqual(len(record.predicted), 5)
            self.assertEqual(len(set(record.predicted)), 5)
            for candidate in record.predicted:
                self.assertEqual(len(candidate), 6)
                self.assertEqual(len(set(candidate)), 6)
                self.assertTrue(all(1 <= value <= 43 for value in candidate))

    def test_uniform_random_is_reproducible_by_seed(self):
        first = run_uniform_random_backtest(
            self.history,
            self.config,
            top_k=3,
            seed=77,
            include_records=True,
        )
        second = run_uniform_random_backtest(
            self.history,
            self.config,
            top_k=3,
            seed=77,
            include_records=True,
        )
        different = run_uniform_random_backtest(
            self.history,
            self.config,
            top_k=3,
            seed=78,
            include_records=True,
        )

        self.assertEqual(first.records, second.records)
        self.assertNotEqual(first.records, different.records)

    @patch("backtester.predict")
    def test_filtered_random_does_not_use_score_ranking(self, mock_predict):
        summary = run_filtered_random_backtest(
            self.history,
            self.config,
            candidate_count=30,
            top_k=3,
            seed=2025,
            include_records=True,
        )

        mock_predict.assert_not_called()
        self.assertEqual(summary.tested_periods, 5)
        self.assertTrue(all(len(record.predicted) == 3 for record in summary.records))


class NumbersRandomBaselineTest(unittest.TestCase):

    def test_numbers3_uniform_random_preserves_digit_width(self):
        history = [
            (index % 10, (index + 1) % 10, (index + 2) % 10)
            for index in range(30)
        ]
        summary = run_numbers_uniform_random_backtest(
            history,
            {
                "digit_count": 3,
                "train_window": 20,
                "tested_periods": 5,
                "top_k": 10,
            },
            seed=2025,
            include_records=True,
        )

        self.assertEqual(summary.tested_periods, 5)
        for record in summary.records:
            self.assertEqual(len(record.predicted), 10)
            self.assertEqual(len(set(record.predicted)), 10)
            for candidate in record.predicted:
                self.assertEqual(len(candidate), 3)
                self.assertTrue(all(0 <= digit <= 9 for digit in candidate))

    def test_numbers4_uniform_random_is_reproducible(self):
        history = [
            (
                index % 10,
                (index + 1) % 10,
                (index + 2) % 10,
                (index + 3) % 10,
            )
            for index in range(30)
        ]
        config = {
            "digit_count": 4,
            "train_window": 20,
            "tested_periods": 5,
            "top_k": 10,
        }
        first = run_numbers_uniform_random_backtest(
            history, config, seed=11, include_records=True
        )
        second = run_numbers_uniform_random_backtest(
            history, config, seed=11, include_records=True
        )
        self.assertEqual(first.records, second.records)


if __name__ == "__main__":
    unittest.main()

class OptimizerRandomBaselineContractTest(unittest.TestCase):

    @patch("optimizer._run_backtest_result")
    def test_model_uplift_uses_uniform_not_filtered_random(
        self,
        mock_run_backtest,
    ):
        mock_run_backtest.return_value = {
            "config": "model",
            "tested_periods": 5,
            "avg_matches": 1.0,
            "average_matches_per_ticket": 1.0,
            "hit_rate_1match": 1.0,
            "hit_rate_2match": 0.0,
            "hit_rate_3match": 0.0,
            "hit_rate_4match": 0.0,
            "hit_rate_5match": 0.0,
            "hit_rate_6match": 0.0,
            "hit_rate_7match": 0.0,
        }
        from optimizer import _evaluate_config

        uniform = {
            1: {
                "config": "uniform_random",
                "tested_periods": 5,
                "avg_matches": 0.5,
                "average_matches_per_ticket": 0.5,
            }
        }
        filtered = {
            1: {
                "config": "filtered_random",
                "tested_periods": 5,
                "avg_matches": 0.8,
                "average_matches_per_ticket": 0.8,
            }
        }
        config = {
            "name": "model",
            "w": {
                "freq": 1,
                "recent": 1,
                "pair": 1,
                "triplet": 1,
                "delay": 1,
                "dist": 1,
                "repeat": 1,
            },
            "f": {},
        }
        result = _evaluate_config(
            history=[(1, 2, 3, 4, 5, 6)] * 30,
            game_config={
                "min_num": 1,
                "max_num": 43,
                "pick_count": 6,
            },
            optimizer_config=config,
            train_window=20,
            tested_periods=5,
            candidate_count=10,
            seeds=(1,),
            random_baselines=uniform,
            filtered_random_baselines=filtered,
        )

        self.assertEqual(result["random_unfiltered_avg"], 0.5)
        self.assertEqual(result["random_filtered_avg"], 0.8)
        self.assertEqual(result["random_uplift"], 0.5)
        self.assertEqual(
            result["random_filtered_baseline"]["config"],
            "filtered_random",
        )
