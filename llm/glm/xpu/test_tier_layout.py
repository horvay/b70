"""Run with python3 llm/glm/xpu/test_tier_layout.py. Does not import torch."""
import unittest

from glm_xpu.tier_layout import split_slots


class SlotLayoutTest(unittest.TestCase):
    def test_rare_largest_class_can_stage_a_full_layer(self):
        records = [12619776, 9474048]
        budget = int((22.8 - 3) * 2**30) - 48 * max(records)
        slots = split_slots(budget, [1, 41], records, [0, 0], [288, 0])
        self.assertEqual(slots[0], 288)
        self.assertGreater(slots[1], 288)
        self.assertLessEqual(sum(n * r for n, r in zip(slots, records)), budget)

    def test_unconstrained_split_preserves_proportions(self):
        self.assertEqual(split_slots(1000, [2, 3], [4, 2], [0, 0]), [142, 214])

    def test_existing_pool_large_enough_keeps_its_share(self):
        self.assertEqual(split_slots(1000, [2, 3], [4, 2], [0, 0], [100, 0]), [142, 214])

    def test_draft_only_fixed_pool(self):
        self.assertEqual(split_slots(1000, [0, 1, 41], [4, 3, 2], [10, 0, 0], [0, 100, 0]),
                         [10, 100, 330])

    def test_insufficient_budget_fails_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "cannot fit minimum pools"):
            split_slots(99, [1, 41], [4, 2], [0, 0], [25, 0])

    def test_several_minima(self):
        self.assertEqual(split_slots(100, [1, 1, 10], [2, 2, 1], [0, 0, 0], [10, 20, 0]),
                         [10, 20, 40])


if __name__ == "__main__":
    unittest.main()
