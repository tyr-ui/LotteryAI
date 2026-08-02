from __future__ import annotations

import unittest
from types import SimpleNamespace

from features import normalize_history
from games import LOTTO_GAMES
from numbers_predictor import format_number, generate_all_candidates


class CombinationPredictionConstraintTest(unittest.TestCase):
    def test_valid_combination_rows_are_sorted_and_preserved(self) -> None:
        for game_key in ("loto6", "loto7", "miniloto"):
            with self.subTest(game_key=game_key):
                config = LOTTO_GAMES[game_key]
                pick_count = int(config["pick_count"])
                min_num = int(config["min_num"])
                max_num = int(config["max_num"])

                source = tuple(
                    range(min_num + pick_count - 1, min_num - 1, -1)
                )
                normalized = normalize_history(
                    [source],
                    pick_count=pick_count,
                    min_num=min_num,
                    max_num=max_num,
                )

                candidate = normalized[0]
                self.assertEqual(pick_count, len(candidate))
                self.assertEqual(candidate, tuple(sorted(candidate)))
                self.assertEqual(pick_count, len(set(candidate)))
                self.assertGreaterEqual(candidate[0], min_num)
                self.assertLessEqual(candidate[-1], max_num)

    def test_combination_rows_reject_duplicate_numbers(self) -> None:
        config = LOTTO_GAMES["loto6"]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_history(
                [(1, 2, 3, 4, 5, 5)],
                pick_count=int(config["pick_count"]),
                min_num=int(config["min_num"]),
                max_num=int(config["max_num"]),
            )

    def test_combination_rows_reject_out_of_range_numbers(self) -> None:
        config = LOTTO_GAMES["miniloto"]

        with self.assertRaisesRegex(ValueError, "outside"):
            normalize_history(
                [(1, 2, 3, 4, int(config["max_num"]) + 1)],
                pick_count=int(config["pick_count"]),
                min_num=int(config["min_num"]),
                max_num=int(config["max_num"]),
            )

    def test_combination_rows_reject_wrong_number_count(self) -> None:
        config = LOTTO_GAMES["loto7"]

        with self.assertRaisesRegex(ValueError, "must contain"):
            normalize_history(
                [(1, 2, 3, 4, 5, 6)],
                pick_count=int(config["pick_count"]),
                min_num=int(config["min_num"]),
                max_num=int(config["max_num"]),
            )


class NumbersPredictionConstraintTest(unittest.TestCase):
    @staticmethod
    def _context(game_key: str) -> SimpleNamespace:
        config = LOTTO_GAMES[game_key]
        return SimpleNamespace(
            digit_count=int(config["digit_count"]),
            digit_min=int(config["digit_min"]),
            digit_max=int(config["digit_max"]),
        )

    def test_numbers3_generates_all_1000_candidates(self) -> None:
        candidates = generate_all_candidates(self._context("numbers3"))

        self.assertEqual(1000, len(candidates))
        self.assertEqual((0, 0, 0), candidates[0])
        self.assertEqual((9, 9, 9), candidates[-1])
        self.assertEqual(1000, len(set(candidates)))

        for candidate in candidates:
            self.assertEqual(3, len(candidate))
            self.assertTrue(all(0 <= digit <= 9 for digit in candidate))

    def test_numbers4_generates_all_10000_candidates(self) -> None:
        candidates = generate_all_candidates(self._context("numbers4"))

        self.assertEqual(10000, len(candidates))
        self.assertEqual((0, 0, 0, 0), candidates[0])
        self.assertEqual((9, 9, 9, 9), candidates[-1])
        self.assertEqual(10000, len(set(candidates)))

        for candidate in candidates:
            self.assertEqual(4, len(candidate))
            self.assertTrue(all(0 <= digit <= 9 for digit in candidate))

    def test_numbers_format_preserves_leading_zero(self) -> None:
        self.assertEqual("017", format_number((0, 1, 7)))
        self.assertEqual("0007", format_number((0, 0, 0, 7)))

    def test_numbers_candidates_allow_repeated_digits(self) -> None:
        numbers3 = set(generate_all_candidates(self._context("numbers3")))
        numbers4 = set(generate_all_candidates(self._context("numbers4")))

        self.assertIn((1, 1, 1), numbers3)
        self.assertIn((0, 0, 0), numbers3)
        self.assertIn((7, 7, 7, 7), numbers4)
        self.assertIn((0, 0, 0, 7), numbers4)


if __name__ == "__main__":
    unittest.main()
