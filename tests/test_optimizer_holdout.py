import unittest

from optimizer import HOLDOUT_PERIODS, _split_holdout_history


class OptimizerHoldoutTest(unittest.TestCase):

    def test_last_draws_are_excluded_from_selection_history(self):
        history = [(index,) for index in range(250)]

        selection, holdout = _split_holdout_history(
            history,
            train_window=100,
            tested_periods=90,
        )

        self.assertEqual(holdout, HOLDOUT_PERIODS)
        self.assertEqual(len(selection), 220)
        self.assertEqual(selection[-1], (219,))
        self.assertEqual(history[-holdout], (220,))

    def test_holdout_is_reduced_when_history_is_short(self):
        history = [(index,) for index in range(205)]

        selection, holdout = _split_holdout_history(
            history,
            train_window=100,
            tested_periods=90,
        )

        self.assertEqual(holdout, 15)
        self.assertEqual(len(selection), 190)

    def test_holdout_is_disabled_when_selection_window_is_not_available(self):
        history = [(index,) for index in range(180)]

        selection, holdout = _split_holdout_history(
            history,
            train_window=100,
            tested_periods=90,
        )

        self.assertEqual(holdout, 0)
        self.assertIs(selection, history)


if __name__ == "__main__":
    unittest.main()
