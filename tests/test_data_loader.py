from __future__ import annotations

import unittest

import pandas as pd

from data_loader import (
    dataframe_to_history,
    normalize_game_dataframe,
    read_csv_text,
    validate_lottery,
)
from games import LOTTO_GAMES


class DataLoaderNormalizationTest(unittest.TestCase):
    def test_loto6_japanese_columns_are_normalized_and_sorted(self) -> None:
        source = pd.DataFrame(
            {
                "回別": [2, 1],
                "抽せん日": ["2026/01/08", "2026/01/01"],
                "本数字1": [7, 1],
                "本数字2": [8, 2],
                "本数字3": [9, 3],
                "本数字4": [10, 4],
                "本数字5": [11, 5],
                "本数字6": [12, 6],
                "ボーナス数字": [13, 7],
            }
        )

        normalized = normalize_game_dataframe(source, LOTTO_GAMES["loto6"])

        self.assertEqual(LOTTO_GAMES["loto6"]["all_columns"], list(normalized.columns))
        self.assertEqual([1, 2], normalized["draw_no"].tolist())
        self.assertEqual([1, 7], normalized["main1"].tolist())

    def test_numbers3_single_number_column_preserves_leading_zero(self) -> None:
        source = pd.DataFrame(
            {
                "回別": [2, 1],
                "抽せん日": ["2026/01/08", "2026/01/01"],
                "当選番号": ["105", "017"],
            }
        )

        normalized = normalize_game_dataframe(source, LOTTO_GAMES["numbers3"])

        self.assertEqual([1, 2], normalized["draw_no"].tolist())
        self.assertEqual([0, 1], normalized["digit1"].tolist())
        self.assertEqual([1, 0], normalized["digit2"].tolist())
        self.assertEqual([7, 5], normalized["digit3"].tolist())
        self.assertEqual(((0, 1, 7), (1, 0, 5)), dataframe_to_history(normalized, LOTTO_GAMES["numbers3"]))

    def test_read_csv_text_keeps_source_values_as_strings(self) -> None:
        frame = read_csv_text("draw_no,date,number\n1,2026/01/01,017\n")

        self.assertEqual("017", frame.loc[0, "number"])


class DataLoaderValidationTest(unittest.TestCase):
    def test_valid_loto6_dataframe_is_ok(self) -> None:
        frame = pd.DataFrame(
            [
                [1, "2026/01/01", 1, 2, 3, 4, 5, 6, 7],
                [2, "2026/01/08", 8, 9, 10, 11, 12, 13, 14],
            ],
            columns=LOTTO_GAMES["loto6"]["all_columns"],
        )

        report = validate_lottery(frame, LOTTO_GAMES["loto6"])

        self.assertEqual("ok", report["status"])
        self.assertEqual(0, report["duplicate_draw_no"])
        self.assertEqual(0, report["out_of_range_cells"])
        self.assertEqual(0, report["duplicate_main_numbers_rows"])

    def test_loto_validation_reports_duplicate_and_out_of_range(self) -> None:
        frame = pd.DataFrame(
            [
                [1, "2026/01/01", 1, 2, 3, 4, 5, 5, 44],
                [1, "2026/01/08", 8, 9, 10, 11, 12, 13, 14],
            ],
            columns=LOTTO_GAMES["loto6"]["all_columns"],
        )

        report = validate_lottery(frame, LOTTO_GAMES["loto6"])

        self.assertEqual("warning", report["status"])
        self.assertEqual(1, report["duplicate_draw_no"])
        self.assertEqual(1, report["out_of_range_cells"])
        self.assertEqual(1, report["duplicate_main_numbers_rows"])

    def test_numbers_validation_allows_repeated_digits_but_rejects_out_of_range(self) -> None:
        valid = pd.DataFrame(
            [
                [1, "2026/01/01", 0, 0, 7],
                [2, "2026/01/08", 1, 1, 1],
            ],
            columns=LOTTO_GAMES["numbers3"]["all_columns"],
        )
        invalid = valid.copy()
        invalid.loc[1, "digit3"] = 10

        valid_report = validate_lottery(valid, LOTTO_GAMES["numbers3"])
        invalid_report = validate_lottery(invalid, LOTTO_GAMES["numbers3"])

        self.assertEqual("ok", valid_report["status"])
        self.assertEqual(0, valid_report["out_of_range_cells"])
        self.assertEqual("warning", invalid_report["status"])
        self.assertEqual(1, invalid_report["out_of_range_cells"])


if __name__ == "__main__":
    unittest.main()
