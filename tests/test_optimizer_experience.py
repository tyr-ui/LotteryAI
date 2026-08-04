
import copy
import unittest

from optimizer_experience import SCHEMA_VERSION, load_search_allocation


class OptimizerExperienceTest(unittest.TestCase):

    def test_schema_version_is_current(self):
        self.assertEqual(SCHEMA_VERSION, "1.4")

    def test_allocation_has_required_structure(self):
        allocation = load_search_allocation("loto6")

        self.assertIn("adaptive", allocation)
        self.assertIn("reason", allocation)
        self.assertIn("counts", allocation)
        self.assertIn("total_count", allocation)

        counts = allocation["counts"]
        for key in ("experience", "random", "local", "evolution"):
            self.assertIn(key, counts)
            self.assertIsInstance(counts[key], int)
            self.assertGreaterEqual(counts[key], 0)

    def test_total_search_budget_matches_counts(self):
        allocation = load_search_allocation("numbers3")
        counts = allocation["counts"]

        total = (
            counts["experience"]
            + counts["random"]
            + counts["local"]
            + counts["evolution"]
        )
        self.assertEqual(total, allocation["total_count"])
        self.assertGreater(total, 0)

    def test_load_search_allocation_is_pure(self):
        first = load_search_allocation("loto7")
        second = load_search_allocation("loto7")
        self.assertEqual(first, second)

        modified = copy.deepcopy(first)
        modified["counts"]["random"] += 1
        self.assertNotEqual(modified, second)


if __name__ == "__main__":
    unittest.main()
