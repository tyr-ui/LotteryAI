import json
import tempfile
import unittest
from pathlib import Path

from evaluation_dashboard import (
    build_evaluation_dashboard,
    render_evaluation_dashboard_markdown,
    write_evaluation_dashboard,
)


class EvaluationDashboardTest(unittest.TestCase):

    def write_json(self, directory: Path, name: str, data) -> None:
        (directory / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_missing_files_create_all_five_games(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = build_evaluation_dashboard(Path(temp_dir))

        self.assertEqual(dashboard["status"], "unknown")
        self.assertEqual(
            set(dashboard["games"]),
            {"loto6", "loto7", "miniloto", "numbers3", "numbers4"},
        )
        for game in dashboard["games"].values():
            self.assertEqual(
                game["observed_evaluation"]["status"],
                "未評価",
            )
            self.assertEqual(
                game["observed_evaluation"]["evaluated_draws"],
                0,
            )

    def test_loto_observed_evaluation_is_aggregated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.write_json(
                output_dir,
                "evaluation_history.json",
                [
                    {
                        "draw_type": "loto6",
                        "status": "evaluated",
                        "draw_no": 1,
                        "best_match_count": 2,
                        "avg_match_count": 0.8,
                        "hit_rate_1match": 0.8,
                        "hit_rate_2match": 0.2,
                        "hit_rate_3match": 0.0,
                        "hit_rate_4match": 0.0,
                    },
                    {
                        "draw_type": "loto6",
                        "status": "evaluated",
                        "draw_no": 2,
                        "best_match_count": 3,
                        "avg_match_count": 1.2,
                        "hit_rate_1match": 1.0,
                        "hit_rate_2match": 0.4,
                        "hit_rate_3match": 0.2,
                        "hit_rate_4match": 0.0,
                    },
                ],
            )
            self.write_json(
                output_dir,
                "evaluation_summary.json",
                {
                    "loto6": {
                        "evaluated_draws": 2,
                        "avg_best_match_count": 2.5,
                        "avg_all_pattern_matches": 1.0,
                        "max_best_match_count": 3,
                        "best_draw_no": 2,
                        "latest_evaluated_draw_no": 2,
                    }
                },
            )

            dashboard = build_evaluation_dashboard(output_dir)

        loto6 = dashboard["games"]["loto6"]["observed_evaluation"]
        self.assertEqual(loto6["status"], "データ不足")
        self.assertEqual(loto6["evaluated_draws"], 2)
        self.assertEqual(loto6["all_time"]["avg_best_match_count"], 2.5)
        self.assertEqual(loto6["recent_5"]["max_best_match_count"], 3)
        self.assertEqual(loto6["recent_5"]["avg_best_match_count"], 2.5)

    def test_current_prediction_and_search_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.write_json(
                output_dir,
                "run_summary.json",
                {
                    "status": "ok",
                    "generated_at": "2026-08-02T00:00:00+00:00",
                    "games": {
                        "loto7": {
                            "latest_draw_no": 688,
                            "next_draw_no": 689,
                            "selected_config": "experience_loto7_1",
                            "prediction": [
                                {"numbers": [1, 2, 3, 4, 5, 6, 7]}
                            ],
                        }
                    },
                },
            )
            self.write_json(
                output_dir,
                "optimizer_result.json",
                {
                    "loto7": {
                        "selected_config": "experience_loto7_1",
                        "ranked_configs": [
                            {
                                "config": "experience_loto7_1",
                                "search_origin": "experience",
                                "selection_score": 1.25,
                                "random_uplift": 0.15,
                            }
                        ],
                    }
                },
            )

            dashboard = build_evaluation_dashboard(output_dir)

        current = dashboard["games"]["loto7"]["current"]
        backtest = dashboard["games"]["loto7"]["optimizer_backtest"]
        self.assertEqual(current["next_draw_no"], 689)
        self.assertEqual(current["selected_search_source"], "experience")
        self.assertEqual(
            current["prediction"][0]["numbers"],
            [1, 2, 3, 4, 5, 6, 7],
        )
        self.assertEqual(backtest["selection_score"], 1.25)
        self.assertEqual(backtest["random_uplift"], 0.15)

    def test_markdown_contains_all_games_and_warning(self):
        dashboard = {
            "schema_version": "1.0",
            "generated_at": "2026-08-02T00:00:00+00:00",
            "overall": {
                "full_run_status": "ok",
                "best_observed_game": None,
                "least_evaluated_games": [
                    "loto6",
                    "loto7",
                    "miniloto",
                    "numbers3",
                    "numbers4",
                ],
            },
            "games": {
                key: {
                    "current": {
                        "next_draw_no": 1,
                        "selected_config": None,
                        "selected_search_source": None,
                        "prediction": [],
                    },
                    "observed_evaluation": {
                        "status": "未評価",
                        "evaluated_draws": 0,
                        "all_time": {},
                    },
                    "optimizer_backtest": {},
                    "experience": {"history_count": 0},
                    "warnings": ["事後評価が0回です。"],
                }
                for key in (
                    "loto6",
                    "loto7",
                    "miniloto",
                    "numbers3",
                    "numbers4",
                )
            },
        }

        markdown = render_evaluation_dashboard_markdown(dashboard)

        self.assertIn("# LotteryAI Evaluation Dashboard", markdown)
        self.assertIn("## LOTO6", markdown)
        self.assertIn("## LOTO7", markdown)
        self.assertIn("## ミニロト", markdown)
        self.assertIn("## Numbers3", markdown)
        self.assertIn("## Numbers4", markdown)
        self.assertIn("事後評価が0回です。", markdown)

    def test_write_dashboard_creates_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dashboard = write_evaluation_dashboard(output_dir)

            json_path = output_dir / "evaluation_dashboard.json"
            markdown_path = output_dir / "evaluation_dashboard.md"

            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                dashboard,
            )
            self.assertIn(
                "LotteryAI Evaluation Dashboard",
                markdown_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
