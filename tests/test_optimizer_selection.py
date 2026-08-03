import unittest

from optimizer import _rank_robust_finalists


class OptimizerRobustSelectionTest(unittest.TestCase):

    def test_one_seed_candidate_is_not_part_of_final_ranking(self):
        preliminary_results = [
            {
                "config": "robust_a",
                "selection_score": 1.80,
                "evaluated_seeds": 3,
            },
            {
                "config": "robust_b",
                "selection_score": 1.70,
                "evaluated_seeds": 3,
            },
            {
                "config": "one_seed_only",
                "selection_score": 2.10,
                "evaluated_seeds": 1,
            },
        ]

        robust_results_by_name = {
            row["config"]: row
            for row in preliminary_results
            if row["evaluated_seeds"] == 3
        }

        ranked = _rank_robust_finalists(
            robust_results_by_name
        )

        self.assertEqual(ranked[0]["config"], "robust_a")
        self.assertTrue(
            all(row["evaluated_seeds"] == 3 for row in ranked)
        )
        self.assertNotIn(
            "one_seed_only",
            {row["config"] for row in ranked},
        )

    def test_empty_robust_results_fail_closed(self):
        with self.assertRaises(RuntimeError):
            _rank_robust_finalists({})


if __name__ == "__main__":
    unittest.main()
