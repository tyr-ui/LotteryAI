from __future__ import annotations

import unittest

import pandas as pd

from data_loader import (
    DataNormalizationError,
    normalize_game_dataframe,
)
from games import LOTTO_GAMES


class DataLoaderIntegrityTest(unittest.TestCase):

    def test_broken_latest_loto_row_is_not_silently_dropped(self) -> None:
        source = pd.DataFrame(
            {
                "draw_no": [2124, 2125],
                "date": ["2026/07/30", "2026/08/03"],
                "main1": [1, 8],
                "main2": [2, 9],
                "main3": [3, 10],
                "main4": [4, 11],
                "main5": [5, 12],
                "main6": [6, None],
                "bonus1": [7, 14],
            }
        )

        with self.assertRaises(DataNormalizationError) as context:
            normalize_game_dataframe(
                source,
                LOTTO_GAMES["loto6"],
            )

        error = context.exception
        self.assertEqual(2, error.raw_rows)
        self.assertEqual(1, error.normalized_rows)
        self.assertEqual(1, error.dropped_rows)
        self.assertEqual(("2125",), error.parse_error_rows)
        self.assertEqual(
            ("2125",),
            error.parse_error_columns["main6"],
        )

    def test_broken_numbers_latest_row_is_not_silently_dropped(self) -> None:
        source = pd.DataFrame(
            {
                "draw_no": [7039, 7040],
                "date": ["2026/07/31", "2026/08/03"],
                "digit1": [0, 1],
                "digit2": [1, None],
                "digit3": [7, 3],
            }
        )

        with self.assertRaises(DataNormalizationError) as context:
            normalize_game_dataframe(
                source,
                LOTTO_GAMES["numbers3"],
            )

        self.assertEqual(
            ("7040",),
            context.exception.parse_error_rows,
        )

    def test_valid_rows_are_preserved(self) -> None:
        source = pd.DataFrame(
            {
                "draw_no": [2, 1],
                "date": ["2026/01/08", "2026/01/01"],
                "digit1": [1, 0],
                "digit2": [0, 1],
                "digit3": [5, 7],
            }
        )

        normalized = normalize_game_dataframe(
            source,
            LOTTO_GAMES["numbers3"],
        )

        self.assertEqual([1, 2], normalized["draw_no"].tolist())
        self.assertEqual(2, len(normalized))


if __name__ == "__main__":
    unittest.main()
