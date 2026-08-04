import tempfile
import unittest
from pathlib import Path

from notification_summary import (
    build_discord_payload,
    build_notification_summary,
    write_notification_summary,
)


class NotificationSummaryTest(unittest.TestCase):

    def setUp(self):
        empty_game = {
            "next_draw_no": 1,
            "selected_config": "default",
            "prediction": [],
            "ranked_configs": [],
        }
        self.output = {
            "generated_at": "2026-08-03T10:00:00+00:00",
            "carryover": {
                "games": {
                    "loto6": {
                        "status": "ok",
                        "has_carryover": True,
                        "display_amount": "2億円",
                    },
                    "loto7": {
                        "status": "ok",
                        "has_carryover": False,
                        "display_amount": "なし",
                    },
                }
            },
            "previous_evaluation": {
                "loto6": {
                    "status": "evaluated",
                    "draw_no": 100,
                    "actual_numbers": [1, 2, 3, 4, 5, 6],
                    "best_match_count": 3,
                    "avg_match_count": 1.2,
                },
            },
            "loto6": {
                "next_draw_no": 101,
                "selected_config": "robust_a",
                "prediction": [
                    {"numbers": [1, 2, 3, 10, 20, 30]},
                ],
                "holdout_evaluation": {
                    "tested_periods": 30,
                    "avg_matches": 1.9,
                    "random_uplift": 0.2,
                },
                "ranked_configs": [{"random_uplift": 0.18}],
                "feature_ablation": [
                    {
                        "feature": "pair",
                        "active": True,
                        "selection_score_drop": 0.08,
                    },
                ],
            },
            "loto7": dict(empty_game),
            "miniloto": dict(empty_game),
            "numbers3": dict(empty_game),
            "numbers4": dict(empty_game),
        }

    def test_markdown_contains_requested_information(self):
        summary = build_notification_summary(self.output)
        self.assertIn("## AI総評", summary)
        self.assertIn("キャリーオーバー: 2億円", summary)
        self.assertIn("01 02 03 10 20 30", summary)
        self.assertIn("5口中最高一致 3", summary)
        self.assertIn("一様ランダム比 +0.200", summary)
        self.assertIn("pair (+0.080)", summary)
        self.assertIn("処理結果: 正常完了", summary)

    def test_discord_payload_uses_embed_cards(self):
        payload = build_discord_payload(self.output)
        self.assertEqual(payload["schema_version"], "2.0")
        self.assertTrue(payload["has_carryover"])
        self.assertGreaterEqual(len(payload["messages"]), 1)
        embeds = payload["messages"][0]["embeds"]
        self.assertEqual(embeds[0]["title"], "LotteryAI 予想・振り返り")
        loto6 = next(embed for embed in embeds if embed["title"].startswith("LOTO6"))
        self.assertIn("キャリーオーバー: 2億円", loto6["description"])
        self.assertLessEqual(len(embeds), 10)

    def test_write_creates_markdown_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            path = write_notification_summary(output_dir, self.output)
            self.assertTrue(path.exists())
            self.assertTrue((output_dir / "notification_payload.json").exists())


if __name__ == "__main__":
    unittest.main()
