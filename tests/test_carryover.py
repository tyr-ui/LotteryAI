import unittest
from unittest.mock import MagicMock, patch

from carryover import (
    fetch_mizuho_carryover,
    format_yen_japanese,
    parse_mizuho_carryover_html,
)


class CarryoverTest(unittest.TestCase):

    def test_parse_expected_draw_and_amount(self):
        html = """
        <html><body>
          <section>ロト6 第2124回 2026年7月30日抽せん
          販売実績金額 1,260,775,000円
          キャリーオーバー 219,914,692円</section>
          <section>ロト6 第2123回 キャリーオーバー 0円</section>
        </body></html>
        """
        result = parse_mizuho_carryover_html(
            html,
            game_key="loto6",
            expected_draw_no=2124,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.draw_no, 2124)
        self.assertEqual(result.next_draw_no, 2125)
        self.assertEqual(result.amount_yen, 219_914_692)
        self.assertTrue(result.has_carryover)
        self.assertEqual(result.display_amount, "2億1,991万4,692円")

    def test_zero_amount_is_not_carryover(self):
        html = "ロト7 第688回 キャリーオーバー 0円"
        result = parse_mizuho_carryover_html(
            html,
            game_key="loto7",
            expected_draw_no=688,
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.has_carryover)
        self.assertEqual(result.display_amount, "なし")

    def test_stale_draw_is_not_treated_as_current(self):
        html = "ロト6 第2123回 キャリーオーバー 100,000,000円"
        result = parse_mizuho_carryover_html(
            html,
            game_key="loto6",
            expected_draw_no=2124,
        )
        self.assertEqual(result.status, "stale")
        self.assertEqual(result.draw_no, 2123)

    @patch("carryover.requests.get")
    def test_fetch_error_is_nonfatal_result(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("offline")
        result = fetch_mizuho_carryover(
            "loto6",
            expected_draw_no=2124,
        )
        self.assertEqual(result.status, "fetch_error")
        self.assertIsNone(result.amount_yen)

    def test_yen_format(self):
        self.assertEqual(format_yen_japanese(0), "なし")
        self.assertEqual(format_yen_japanese(800_000_000), "8億円")
        self.assertEqual(
            format_yen_japanese(1_365_436_540),
            "13億6,543万6,540円",
        )


if __name__ == "__main__":
    unittest.main()
