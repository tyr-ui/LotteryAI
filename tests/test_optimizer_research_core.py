import unittest
from random import Random

from optimizer_adaptation import build_evolution_adaptation, build_search_allocation
from optimizer_evolution import generate_evolution_candidates, mutate_child
from optimizer_search import (
    config_signature,
    deduplicate_configs,
    generate_random_candidates,
    mutate_weights,
)


PARENTS = [
    {"name": "a", "w": {"freq": .5, "recent": .5}, "f": {"max_con": 1}},
    {"name": "b", "w": {"freq": .2, "recent": .8}, "f": {"max_con": 2}},
]


class OptimizerResearchCoreTest(unittest.TestCase):
    def test_evolution_is_reproducible_and_normalized(self):
        left = generate_evolution_candidates(PARENTS, count=4, rng=Random(9))
        right = generate_evolution_candidates(PARENTS, count=4, rng=Random(9))
        self.assertEqual(left, right)
        self.assertEqual(len(left), 4)
        for child in left:
            self.assertAlmostEqual(sum(child["w"].values()), 1.0, places=6)
            self.assertTrue(all(value >= 0 for value in child["w"].values()))

    def test_evolution_requires_two_parents(self):
        self.assertEqual(generate_evolution_candidates(PARENTS[:1], count=4, rng=Random(1)), [])

    def test_mutation_stays_non_negative_and_normalized(self):
        child = mutate_child(PARENTS[0], rng=Random(2), mutation_rate=1.0, mutation_scale=10.0)
        self.assertAlmostEqual(sum(child["w"].values()), 1.0, places=6)
        self.assertTrue(all(value >= 0 for value in child["w"].values()))

    def test_random_search_is_reproducible(self):
        first = generate_random_candidates(count=5, rng=Random(4), inherited_filters={"max_con": 1})
        second = generate_random_candidates(count=5, rng=Random(4), inherited_filters={"max_con": 1})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_deduplication_uses_config_signature(self):
        duplicate = dict(PARENTS[0])
        unique = deduplicate_configs([PARENTS[0], duplicate, PARENTS[1]])
        self.assertEqual(len(unique), 2)
        self.assertNotEqual(config_signature(unique[0]), config_signature(unique[1]))

    def test_adaptation_fails_safe_with_short_history(self):
        self.assertFalse(build_evolution_adaptation([])["adaptive"])
        allocation = build_search_allocation([])
        self.assertFalse(allocation["adaptive"])
        self.assertEqual(sum(allocation["counts"].values()), allocation["total_count"])


if __name__ == "__main__":
    unittest.main()
