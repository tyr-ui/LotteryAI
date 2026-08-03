from __future__ import annotations

import unittest
from unittest.mock import patch

import optimizer
import optimizer_evaluation
from backtester import BacktestSummary
from predictor import PredictionWeights


def _summary() -> BacktestSummary:
    return BacktestSummary(
        tested_periods=3,
        average_best_matches=2.0,
        average_matches_per_ticket=1.0,
        hit_rate_1match=1.0,
        hit_rate_2match=0.5,
        hit_rate_3match=0.0,
        hit_rate_4match=0.0,
        hit_rate_5match=0.0,
        hit_rate_6match=0.0,
        hit_rate_7match=0.0,
        records=(),
    )


class FiveTicketEvaluationTest(unittest.TestCase):

    @patch("optimizer.run_backtest")
    def test_optimizer_model_backtest_uses_five_tickets(self, mock_run):
        mock_run.return_value = _summary()

        optimizer._run_backtest_result(
            history=[(1, 2, 3, 4, 5, 6)] * 20,
            game_config={"min_num": 1, "max_num": 43, "pick_count": 6},
            config_name="model",
            train_window=10,
            tested_periods=3,
            candidate_count=100,
            weights=PredictionWeights(),
            seed=2025,
        )

        self.assertEqual(
            mock_run.call_args.kwargs["top_k"],
            optimizer.OPTIMIZATION_TOP_K,
        )
        self.assertEqual(optimizer.OPTIMIZATION_TOP_K, 5)

    @patch("optimizer.run_uniform_random_backtest")
    def test_uniform_random_uses_same_five_ticket_budget(self, mock_run):
        mock_run.return_value = _summary()

        optimizer._run_random_backtest_result(
            history=[(1, 2, 3, 4, 5, 6)] * 20,
            game_config={"min_num": 1, "max_num": 43, "pick_count": 6},
            config_name="uniform_random",
            train_window=10,
            tested_periods=3,
            candidate_count=100,
            seed=2025,
            filtered=False,
        )

        self.assertEqual(
            mock_run.call_args.kwargs["top_k"],
            optimizer.OPTIMIZATION_TOP_K,
        )

    @patch("optimizer.run_filtered_random_backtest")
    def test_filtered_random_uses_same_five_ticket_budget(self, mock_run):
        mock_run.return_value = _summary()

        optimizer._run_random_backtest_result(
            history=[(1, 2, 3, 4, 5, 6)] * 20,
            game_config={"min_num": 1, "max_num": 43, "pick_count": 6},
            config_name="filtered_random",
            train_window=10,
            tested_periods=3,
            candidate_count=100,
            seed=2025,
            filtered=True,
        )

        self.assertEqual(
            mock_run.call_args.kwargs["top_k"],
            optimizer.OPTIMIZATION_TOP_K,
        )

    @patch("optimizer_evaluation.run_backtest")
    def test_ablation_evaluation_uses_same_five_ticket_budget(self, mock_run):
        mock_run.return_value = _summary()

        optimizer_evaluation.run_backtest_result(
            history=[(1, 2, 3, 4, 5, 6)] * 20,
            game_config={"min_num": 1, "max_num": 43, "pick_count": 6},
            config_name="ablation",
            train_window=10,
            tested_periods=3,
            candidate_count=100,
            weights=PredictionWeights(),
            seed=2025,
        )

        self.assertEqual(
            mock_run.call_args.kwargs["top_k"],
            optimizer_evaluation.OPTIMIZATION_TOP_K,
        )
        self.assertEqual(optimizer_evaluation.OPTIMIZATION_TOP_K, 5)


if __name__ == "__main__":
    unittest.main()
