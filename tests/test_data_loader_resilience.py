from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from data_loader import (
    _request_with_retries,
    load_game_data,
)
from games import LOTTO_GAMES


class DataLoaderResilienceTest(unittest.TestCase):

    @patch("data_loader.time.sleep")
    @patch("data_loader.requests.get")
    def test_request_retries_timeout_then_succeeds(
        self,
        mocked_get: Mock,
        mocked_sleep: Mock,
    ) -> None:
        success = Mock()
        success.raise_for_status.return_value = None
        success.content = b"ok"
        mocked_get.side_effect = [
            requests.ConnectTimeout("temporary"),
            success,
        ]

        response = _request_with_retries(
            "https://example.invalid/data.csv",
            headers={},
            timeout=1,
            attempts=2,
        )

        self.assertIs(success, response)
        self.assertEqual(2, mocked_get.call_count)
        mocked_sleep.assert_called_once_with(1)

    @patch("data_loader.download_game_csv")
    def test_uses_valid_cache_when_remote_download_fails(
        self,
        mocked_download: Mock,
    ) -> None:
        mocked_download.side_effect = requests.ConnectTimeout(
            "temporary"
        )
        config = LOTTO_GAMES["loto6"]
        cached = pd.DataFrame(
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

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "loto6.csv"
            cached.to_csv(destination, index=False)

            loaded = load_game_data(
                "loto6",
                config,
                destination=destination,
            )

        self.assertEqual("cache", loaded.source)
        self.assertEqual(2, loaded.validation["latest_draw_no"])
        self.assertEqual(2, len(loaded.dataframe))

    @patch("data_loader.download_game_csv")
    def test_header_only_cache_is_not_accepted(
        self,
        mocked_download: Mock,
    ) -> None:
        mocked_download.side_effect = requests.ConnectTimeout(
            "temporary"
        )
        config = LOTTO_GAMES["loto7"]

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "loto7.csv"
            destination.write_text(
                "draw_no,date,main1,main2,main3,main4,"
                "main5,main6,main7,bonus1,bonus2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "no valid local cache",
            ):
                load_game_data(
                    "loto7",
                    config,
                    destination=destination,
                )


if __name__ == "__main__":
    unittest.main()
