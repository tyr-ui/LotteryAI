import unittest
from unittest.mock import patch

from carryover import (
    fetch_carryover_snapshot,
    fetch_mizuho_carryover,
    format_yen_japanese,
    parse_mizuho_carryover_html,
    parse_official_status_html,
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
            html, game_key="loto6", expected_draw_no=2124,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.amount_yen, 219_914_692)
        self.assertTrue(result.has_carryover)

    def test_zero_amount_is_not_carryover(self):
        result = parse_mizuho_carryover_html(
            "ロト7 第688回 キャリーオーバー 0円",
            game_key="loto7", expected_draw_no=688,
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.has_carryover)
        self.assertEqual(result.display_amount, "なし")

    def test_official_explicit_banner_is_status_only(self):
        html = '<img alt="キャリーオーバー発生中">'
        result = parse_official_status_html(
            html, game_key="loto6", expected_draw_no=2125,
        )
        self.assertEqual(result.status, "status_only")
        self.assertTrue(result.has_carryover)
        self.assertEqual(result.display_amount, "発生中（金額未取得）")

    def test_explanatory_text_alone_is_not_treated_as_active(self):
        html = '<p>キャリーオーバー発生時は最高6億円です。</p>'
        result = parse_official_status_html(
            html, game_key="loto6", expected_draw_no=2125,
        )
        self.assertEqual(result.status, "status_unknown")
        self.assertIsNone(result.has_carryover)

    @patch("carryover.fetch_official_status")
    @patch("carryover.fetch_mizuho_carryover")
    def test_same_draw_uses_previous_successful_amount(self, primary, fallback):
        from carryover import CarryoverResult
        primary.return_value = CarryoverResult(
            "loto6", "fetch_error", "mizuho_bank", "url", 2125,
            None, 2126, None, None, "未取得", "offline",
        )
        fallback.return_value = CarryoverResult(
            "loto6", "status_unknown", "official", "url2", 2125,
            2125, 2126, None, None, "未取得", "unknown",
        )
        previous = {
            "games": {
                "loto6": {
                    "draw_no": 2125,
                    "amount_yen": 300_000_000,
                    "source_url": "cached-url",
                }
            }
        }
        snapshot = fetch_carryover_snapshot(
            {"loto6": 2125, "loto7": 688},
            previous_snapshot=previous,
        )
        row = snapshot["games"]["loto6"]
        self.assertEqual(row["status"], "cached")
        self.assertEqual(row["amount_yen"], 300_000_000)

    @patch("carryover.requests.get")
    def test_fetch_error_is_nonfatal_result(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("offline")
        result = fetch_mizuho_carryover("loto6", expected_draw_no=2124)
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
