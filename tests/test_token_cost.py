import math
import unittest

from tinycodescaling.metrics.token_cost import (
    aggregate_token_usage,
    quality_per_1k_generated_tokens,
    tokens_per_solved,
)


class TokenCostTests(unittest.TestCase):
    def test_aggregate_token_usage(self):
        records = [
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        ]
        summary = aggregate_token_usage(records)
        self.assertEqual(summary["prompt_tokens"], 16)
        self.assertEqual(summary["completion_tokens"], 9)
        self.assertEqual(summary["total_tokens"], 25)
        self.assertEqual(summary["prompt_tokens_per_problem"], 8)
        self.assertEqual(summary["completion_tokens_per_problem"], 4.5)
        self.assertEqual(summary["total_tokens_per_problem"], 12.5)

    def test_tokens_per_solved_handles_zero(self):
        self.assertTrue(math.isinf(tokens_per_solved(100, 0)))

    def test_quality_per_1k_generated_tokens(self):
        self.assertAlmostEqual(quality_per_1k_generated_tokens(0.5, 100), 5.0)


if __name__ == "__main__":
    unittest.main()

