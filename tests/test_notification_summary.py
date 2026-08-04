import tempfile
import unittest
from pathlib import Path

from notification_summary import (
    build_notification_summary,
    write_notification_summary,
)


class NotificationSummaryTest(unittest.TestCase):

    def setUp(self):
        self.output = {
            "generated_at": "2026-08-03T10:00:00+00:00",
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
                    "holdout_periods": 30,
                    "avg_matches": 1.9,
                    "random_uplift": 0.2,
                },
                "feature_ablation": [
                    {
                        "feature": "pair",
                        "active": True,
                        "selection_score_drop": 0.08,
                    },
                ],
            },
            "loto7": {},
            "miniloto": {},
            "numbers3": {},
            "numbers4": {},
        }

    def test_summary_contains_main_sections(self):
        summary = build_notification_summary(self.output)

        self.assertIn("# LotteryAI 予想・振り返り", summary)
        self.assertIn("## AI総評", summary)
        self.assertIn("## LOTO6", summary)
        self.assertIn("次回: 第101回", summary)
        self.assertIn("01 02 03 10 20 30", summary)
        self.assertIn("5口中最高一致 3", summary)
        self.assertIn("一様ランダム比 +0.200", summary)
        self.assertIn("pair (+0.080)", summary)
        self.assertIn("キャリーオーバー: 未取得", summary)

    def test_write_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_notification_summary(
                Path(directory),
                self.output,
            )

            self.assertTrue(path.exists())
            self.assertIn(
                "LotteryAI 予想・振り返り",
                path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
