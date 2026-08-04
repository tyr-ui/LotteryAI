from __future__ import annotations

import unittest

import pandas as pd

from data_loader import validate_lottery
from games import LOTTO_GAMES


class DataFreshnessAndGapTest(unittest.TestCase):
    def _loto6(self, draw_numbers, dates):
        rows = []
        for draw_no, date in zip(draw_numbers, dates):
            rows.append({
                "draw_no": draw_no,
                "date": date,
                "main1": 1,
                "main2": 2,
                "main3": 3,
                "main4": 4,
                "main5": 5,
                "main6": 6,
                "bonus": 7,
            })
        return pd.DataFrame(rows)

    def test_missing_draw_number_is_reported(self):
        today = pd.Timestamp.now().date().isoformat()
        report = validate_lottery(
            self._loto6([100, 102], [today, today]),
            LOTTO_GAMES["loto6"],
        )
        self.assertEqual(report["draw_number_gap_count"], 1)
        self.assertEqual(report["missing_draw_numbers"], [101])
        self.assertEqual(report["status"], "warning")

    def test_old_latest_date_is_warning(self):
        report = validate_lottery(
            self._loto6([100, 101], ["2000-01-01", "2000-01-08"]),
            LOTTO_GAMES["loto6"],
        )
        self.assertTrue(report["stale_data"])
        self.assertGreater(report["data_age_days"], 10)
        self.assertTrue(report["hard_stale_data"])
        self.assertEqual(report["freshness_status"], "error")
        self.assertEqual(report["status"], "ok")

    def test_recent_contiguous_data_is_ok(self):
        today = pd.Timestamp.now().date().isoformat()
        report = validate_lottery(
            self._loto6([100, 101], [today, today]),
            LOTTO_GAMES["loto6"],
        )
        self.assertEqual(report["draw_number_gap_count"], 0)
        self.assertFalse(report["stale_data"])
        self.assertEqual(report["status"], "ok")


if __name__ == "__main__":
    unittest.main()
