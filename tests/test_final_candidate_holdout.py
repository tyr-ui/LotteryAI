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

    @patch.object(optimizer, "_run_backtest_result")
    @patch.object(optimizer, "_run_random_backtest_result")
    def test_uses_production_candidate_count_and_seed(
        self,
        random_backtest,
        model_backtest,
    ) -> None:
        random_backtest.side_effect = [
            {"config": "uniform_random", "avg_matches": 1.0},
            {"config": "filtered_random", "avg_matches": 1.1},
        ]
        model_backtest.return_value = {
            "config": "winner",
            "tested_periods": 30,
            "avg_matches": 2.0,
            "random_uplift": 1.0,
            "records": [],
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
            model_backtest.call_args.kwargs["candidate_count"],
            10_000,
        )
        self.assertEqual(
            model_backtest.call_args.kwargs["seed"],
            optimizer.SEED,
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

    @patch.object(optimizer, "_run_backtest_result")
    @patch.object(optimizer, "_run_random_backtest_result")
    def test_paired_records_are_joined_by_draw_index(
        self,
        random_backtest,
        model_backtest,
    ) -> None:
        model_backtest.return_value = {
            "config": "winner",
            "tested_periods": 2,
            "avg_matches": 2.0,
            "records": [
                {"draw_index": 2, "actual": [2], "best_match_count": 1},
                {"draw_index": 1, "actual": [1], "best_match_count": 3},
            ],
        }
        random_backtest.side_effect = [
            {
                "config": "uniform_random",
                "avg_matches": 1.0,
                "records": [
                    {"draw_index": 1, "best_match_count": 2},
                    {"draw_index": 2, "best_match_count": 0},
                ],
            },
            {
                "config": "filtered_random",
                "avg_matches": 1.0,
                "records": [
                    {"draw_index": 2, "best_match_count": 1},
                    {"draw_index": 1, "best_match_count": 1},
                ],
            },
        ]
        result = optimizer._evaluate_final_candidate_holdout(
            [(1, 2, 3)] * 50,
            {"min_num": 1, "max_num": 43, "pick_count": 3},
            {"name": "winner", "w": {}, "f": {}},
            train_window=10,
            holdout_periods=2,
            candidate_count=100,
            selection_history_draws=48,
        )
        self.assertEqual(
            [1, 2],
            [row["draw_index"] for row in result["paired_draw_results"]],
        )
        self.assertEqual(result["paired_draw_results"][0]["model_minus_uniform"], 1)
        self.assertEqual(result["paired_draw_results"][1]["model_minus_filtered"], 0)


if __name__ == "__main__":
    unittest.main()
