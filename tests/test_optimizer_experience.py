import copy
import unittest

from optimizer_experience import (
    SCHEMA_VERSION,
    load_search_allocation,
)

class OptimizerExperienceTest(unittest.TestCase):

    def test_schema_version_is_current(self):
        self.assertEqual(SCHEMA_VERSION, "1.3")

    def test_allocation_has_required_keys(self):
        allocation = load_search_allocation("loto6")
        for key in ("experience", "random", "local", "evolution"):
            self.assertIn(key, allocation)
            self.assertIsInstance(allocation[key], int)
            self.assertGreaterEqual(allocation[key], 0)

    def test_total_search_budget_positive(self):
        allocation = load_search_allocation("numbers3")
        total = (
            allocation["experience"]
            + allocation["random"]
            + allocation["local"]
            + allocation["evolution"]
        )
        self.assertGreater(total, 0)

    def test_load_search_allocation_is_pure(self):
        a = load_search_allocation("loto7")
        b = load_search_allocation("loto7")
        self.assertEqual(a, b)
        c = copy.deepcopy(a)
        c["random"] += 1
        self.assertNotEqual(c, b)

if __name__ == "__main__":
    unittest.main()
