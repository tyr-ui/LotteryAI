import unittest

from run_pipeline import (
    carryover_content_changed,
    resolve_run_mode,
    reuse_previous_optimizer_result,
    select_games_for_optimization,
)


class DifferentialExecutionTest(unittest.TestCase):

    def setUp(self):
        self.games = {
            "loto6": {},
            "loto7": {},
            "numbers3": {},
        }
        self.previous = {
            "loto6": {"latest_draw_no": 100, "selected_config": "a"},
            "loto7": {"latest_draw_no": 200, "selected_config": "b"},
            "numbers3": {"latest_draw_no": 300, "selected_config": "c"},
        }
        self.validations = {
            "loto6": {"latest_draw_no": 101},
            "loto7": {"latest_draw_no": 200},
            "numbers3": {"latest_draw_no": 301},
        }

    def test_auto_selects_only_changed_draws(self):
        selected = select_games_for_optimization(
            self.previous,
            self.validations,
            self.games,
            run_mode="auto",
        )
        self.assertEqual(selected, ["loto6", "numbers3"])

    def test_all_selects_every_game(self):
        selected = select_games_for_optimization(
            self.previous,
            self.validations,
            self.games,
            run_mode="all",
        )
        self.assertEqual(selected, list(self.games))

    def test_first_run_selects_every_game(self):
        selected = select_games_for_optimization(
            {},
            self.validations,
            self.games,
            run_mode="auto",
        )
        self.assertEqual(selected, list(self.games))

    def test_resolve_run_mode(self):
        self.assertEqual(resolve_run_mode("auto"), "auto")
        self.assertEqual(resolve_run_mode("ALL"), "all")
        with self.assertRaises(ValueError):
            resolve_run_mode("invalid")

    def test_reuse_removes_section_metadata(self):
        reused = reuse_previous_optimizer_result({
            "latest_draw_no": 10,
            "next_draw_no": 11,
            "rows": 100,
            "validation": {"status": "ok"},
            "selected_config": "model",
            "prediction": [{"numbers": [1, 2, 3]}],
        })
        self.assertEqual(reused["selected_config"], "model")
        self.assertNotIn("latest_draw_no", reused)
        self.assertNotIn("validation", reused)

    def test_carryover_comparison_ignores_timestamp(self):
        previous = {
            "fetched_at": "old",
            "games": {"loto6": {"status": "none"}},
        }
        current = {
            "fetched_at": "new",
            "games": {"loto6": {"status": "none"}},
        }
        self.assertFalse(carryover_content_changed(previous, current))
        current["games"]["loto6"]["status"] = "status_only"
        self.assertTrue(carryover_content_changed(previous, current))


if __name__ == "__main__":
    unittest.main()
