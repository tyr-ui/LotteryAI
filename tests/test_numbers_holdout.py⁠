from types import SimpleNamespace
import unittest
from unittest.mock import patch

from numbers_backtester import NumbersBacktestSummary
from numbers_optimizer import (
    _resolve_holdout_periods,
    optimize_numbers,
)


def _summary(*, tested_periods: int, score: float = 1.0):
    return NumbersBacktestSummary(
        tested_periods=tested_periods,
        digit_count=3,
        average_best_position_matches=1.5,
        average_position_matches_per_ticket=0.5,
        average_best_unordered_matches=2.0,
        average_unordered_matches_per_ticket=1.0,
        straight_hit_rate=0.01,
        box_hit_rate=0.03,
        hit_rate_1_position=0.8,
        hit_rate_2_position=0.3,
        hit_rate_3_position=0.01,
        hit_rate_4_position=0.0,
        selection_score=score,
        records=(),
    )


class NumbersHoldoutTest(unittest.TestCase):

    def test_resolve_holdout_preserves_selection_history(self):
        self.assertEqual(
            _resolve_holdout_periods(
                history_size=700,
                train_window=500,
                requested_periods=60,
            ),
            60,
        )
        self.assertEqual(
            _resolve_holdout_periods(
                history_size=505,
                train_window=500,
                requested_periods=60,
            ),
            0,
        )

    def test_optimizer_selects_without_latest_sixty_draws(self):
        history = [
            (index % 10, (index + 1) % 10, (index + 2) % 10)
            for index in range(700)
        ]
        config = {
            "key": "numbers3",
            "digit_count": 3,
            "train_window": 500,
            "tested_periods": 180,
            "top_k": 10,
            "numbers_optimizer_periods": 24,
            "numbers_holdout_periods": 60,
        }

        evaluated_history_sizes = []
        backtest_calls = []
        random_calls = []

        def fake_evaluate(history_arg, config_arg, **kwargs):
            evaluated_history_sizes.append(len(tuple(history_arg)))
            return _summary(tested_periods=kwargs["tested_periods"])

        def fake_backtest(history_arg, config_arg, **kwargs):
            backtest_calls.append((len(tuple(history_arg)), kwargs["tested_periods"]))
            return _summary(tested_periods=kwargs["tested_periods"], score=1.2)

        def fake_random(history_arg, config_arg, **kwargs):
            random_calls.append((len(tuple(history_arg)), kwargs["tested_periods"]))
            return _summary(tested_periods=kwargs["tested_periods"], score=1.0)

        candidate = SimpleNamespace(
            candidate=(0, 1, 2),
            number="012",
            total_score=1.0,
            components={},
            exact_repeat_count=0,
            unordered_repeat_count=0,
        )
        prediction = SimpleNamespace(
            selected=(candidate,),
            ranked=(candidate,),
            generated_count=1000,
        )

        with (
            patch("numbers_optimizer.evaluate_numbers_weights", side_effect=fake_evaluate),
            patch("numbers_optimizer.run_numbers_backtest", side_effect=fake_backtest),
            patch(
                "numbers_optimizer.run_numbers_uniform_random_backtest",
                side_effect=fake_random,
            ),
            patch("numbers_optimizer.load_search_allocation", return_value={"counts": {"experience": 0, "random": 0, "local": 0, "evolution": 0}}),
            patch("numbers_optimizer.load_evolution_adaptation", return_value={}),
            patch("numbers_optimizer.load_experience_configs", return_value=[]),
            patch("numbers_optimizer.build_numbers_model_context", return_value=object()),
            patch("numbers_optimizer.predict_numbers", return_value=prediction),
        ):
            result = optimize_numbers(history, config)

        self.assertTrue(evaluated_history_sizes)
        self.assertEqual(set(evaluated_history_sizes), {640})
        self.assertIn((640, 180), backtest_calls)
        self.assertIn((700, 60), backtest_calls)
        self.assertTrue(all(size in {640, 700} for size, _ in random_calls))
        self.assertEqual(result["holdout_evaluation"]["tested_periods"], 60)
        self.assertEqual(
            result["holdout_evaluation"]["selection_history_draws"],
            640,
        )
        self.assertIn("random_baseline", result["holdout_evaluation"])


if __name__ == "__main__":
    unittest.main()
