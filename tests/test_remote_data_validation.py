from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data_loader import (
    RemoteDataValidationError,
    load_game_data,
)
from games import LOTTO_GAMES


class RemoteDataValidationTest(unittest.TestCase):

    @staticmethod
    def _valid_loto6_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "draw_no": [1, 2],
                "date": ["2000/01/01", "2000/01/08"],
                "main1": [1, 2],
                "main2": [3, 4],
                "main3": [5, 6],
                "main4": [7, 8],
                "main5": [9, 10],
                "main6": [11, 12],
                "bonus": [13, 14],
            }
        )

    @staticmethod
    def _invalid_remote_csv() -> str:
        return (
            "draw_no,date,main1,main2,main3,main4,main5,main6,bonus\n"
            "1,2000/01/01,1,3,5,7,9,11,13\n"
            "2,2000/01/08,2,4,6,8,10,99,14\n"
        )

    @patch("data_loader.download_game_csv")
    def test_warning_remote_data_falls_back_to_valid_cache(
        self,
        mocked_download: Mock,
    ) -> None:
        mocked_download.return_value = (
            self._invalid_remote_csv(),
            "official",
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "loto6.csv"
            self._valid_loto6_frame().to_csv(
                destination,
                index=False,
            )

            loaded = load_game_data(
                "loto6",
                LOTTO_GAMES["loto6"],
                destination=destination,
            )

            persisted = pd.read_csv(destination)

        self.assertEqual("cache", loaded.source)
        self.assertEqual("ok", loaded.validation["status"])
        self.assertEqual([11, 12], persisted["main6"].tolist())

    @patch("data_loader.download_game_csv")
    def test_warning_remote_data_is_rejected_without_valid_cache(
        self,
        mocked_download: Mock,
    ) -> None:
        mocked_download.return_value = (
            self._invalid_remote_csv(),
            "official",
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "loto6.csv"

            with self.assertRaisesRegex(
                RuntimeError,
                "no valid local cache",
            ) as context:
                load_game_data(
                    "loto6",
                    LOTTO_GAMES["loto6"],
                    destination=destination,
                )

        self.assertIsInstance(
            context.exception.__cause__,
            RemoteDataValidationError,
        )
        validation = context.exception.__cause__.validation
        self.assertEqual("warning", validation["status"])
        self.assertEqual(1, validation["out_of_range_cells"])


if __name__ == "__main__":
    unittest.main()