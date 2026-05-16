import unittest

from tinycodescaling.metrics.pass_at_k import estimate_pass_at_k


class PassAtKTests(unittest.TestCase):
    def test_zero_correct_returns_zero(self):
        self.assertEqual(estimate_pass_at_k(10, 0, 1), 0.0)

    def test_all_correct_returns_one(self):
        self.assertEqual(estimate_pass_at_k(10, 10, 1), 1.0)

    def test_pass_at_1_matches_fraction_correct(self):
        self.assertAlmostEqual(estimate_pass_at_k(10, 1, 1), 0.1)

    def test_pass_at_2_matches_closed_form(self):
        self.assertAlmostEqual(estimate_pass_at_k(5, 1, 2), 0.4)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            estimate_pass_at_k(0, 0, 1)
        with self.assertRaises(ValueError):
            estimate_pass_at_k(5, 6, 1)
        with self.assertRaises(ValueError):
            estimate_pass_at_k(5, 1, 0)


if __name__ == "__main__":
    unittest.main()

