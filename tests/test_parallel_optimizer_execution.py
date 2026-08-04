from unittest import mock
import unittest

import run_pipeline


class ParallelOptimizerExecutionTest(unittest.TestCase):

    def test_resolve_workers_is_capped_by_game_count(self):
        self.assertEqual(
            run_pipeline.resolve_optimizer_workers(
                2,
                configured=8,
            ),
            2,
        )

    def test_resolve_workers_rejects_zero(self):
        with self.assertRaises(ValueError):
            run_pipeline.resolve_optimizer_workers(
                5,
                configured=0,
            )

    def test_single_worker_preserves_game_order(self):
        configs = {
            "loto6": {"family": "lotto"},
            "numbers3": {"family": "numbers"},
        }
        datasets = {
            "loto6": object(),
            "numbers3": object(),
        }

        def fake_job(game_key, game_config, dataframe):
            return game_key, {"selected_config": game_key}

        with mock.patch.object(
            run_pipeline,
            "_run_optimizer_job",
            side_effect=fake_job,
        ):
            results = run_pipeline.run_all_optimizers(
                datasets,
                configs,
                max_workers=1,
            )

        self.assertEqual(
            list(results),
            ["loto6", "numbers3"],
        )
        self.assertEqual(
            results["numbers3"]["selected_config"],
            "numbers3",
        )


if __name__ == "__main__":
    unittest.main()