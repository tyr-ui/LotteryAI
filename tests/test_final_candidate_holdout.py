from __future__ import annotations

import unittest
from unittest.mock import patch

import optimizer


class FinalCandidateHoldoutTest(unittest.TestCase):

    def test_disabled_without_holdout(self) -> None:
        result = optimizer._evaluate_final_candidate_holdout(
            [(1, 2, 3)],
            {"min_num": 1, "max_num": 43, "pick_count": 3},
            {"name": "winner", "w": {}, "f": {}},
            train_window=10,
            holdout_periods=0,
            candidate_count=10_000,
            selection_history_draws=100,
        )
        self.assertIsNone(result)

    @patch.object(optimizer, "_evaluate_config")
    @patch.object(optimizer, "_run_random_backtest_result")
    def test_uses_production_candidate_count_and_seed(
        self,
        random_backtest,
        evaluate_config,
    ) -> None:
        random_backtest.side_effect = [
            {"config": "uniform_random", "avg_matches": 1.0},
            {"config": "filtered_random", "avg_matches": 1.1},
        ]
        evaluate_config.return_value = {
            "config": "winner",
            "tested_periods": 30,
            "avg_matches": 2.0,
            "random_uplift": 1.0,
        }

        history = [(1, 2, 3)] * 200
        game_config = {"min_num": 1, "max_num": 43, "pick_count": 3}
        best_config = {"name": "winner", "w": {}, "f": {}}

        result = optimizer._evaluate_final_candidate_holdout(
            history,
            game_config,
            best_config,
            train_window=100,
            holdout_periods=30,
            candidate_count=10_000,
            selection_history_draws=170,
        )

        self.assertEqual(random_backtest.call_count, 2)
        for call in random_backtest.call_args_list:
            self.assertEqual(call.kwargs["candidate_count"], 10_000)
            self.assertEqual(call.kwargs["tested_periods"], 30)
            self.assertEqual(call.kwargs["seed"], optimizer.SEED)

        self.assertEqual(
            evaluate_config.call_args.kwargs["candidate_count"],
            10_000,
        )
        self.assertEqual(
            evaluate_config.call_args.kwargs["seeds"],
            (optimizer.SEED,),
        )
        self.assertEqual(
            result["evaluation_type"],
            "production_candidate_count_holdout",
        )
        self.assertEqual(result["candidate_count"], 10_000)
        self.assertEqual(result["ticket_count"], 5)
        self.assertEqual(result["holdout_periods"], 30)
        self.assertEqual(result["selection_history_draws"], 170)
        self.assertEqual(result["frozen_config"], "winner")
        self.assertEqual(result["production_seed"], optimizer.SEED)


if __name__ == "__main__":
    unittest.main()
