"""Tests for generated-test extraction helpers."""

from __future__ import annotations

import unittest

from tinycodescaling.execution.test_extract import (
    extract_markdown_code_block,
    extract_test_assertions,
)


class GeneratedTestExtractionTests(unittest.TestCase):
    def test_extract_markdown_code_block_prefers_first_fenced_block(self):
        text = "intro\n```python\nassert f(1) == 2\n```\noutro"

        code, used_code_block = extract_markdown_code_block(text)

        self.assertTrue(used_code_block)
        self.assertEqual(code, "assert f(1) == 2")

    def test_extract_test_assertions_reads_plain_asserts(self):
        result = extract_test_assertions(
            "```python\nassert f(1) == 2\nassert g(1) == 3\n```",
            entry_point="f",
        )

        self.assertEqual(result.assertions, ["assert f(1) == 2"])
        self.assertEqual(result.parse_method, "ast_assert_walk")
        self.assertEqual(result.total_assertions_found, 2)
        self.assertEqual(result.rejected_assertions, 1)

    def test_extract_test_assertions_reads_asserts_inside_test_function(self):
        result = extract_test_assertions(
            "def test_basic():\n    assert f(2) == 3\n    assert helper(1) == 1\n",
            entry_point="f",
        )

        self.assertEqual(result.assertions, ["assert f(2) == 3"])

    def test_extract_test_assertions_converts_unittest_style_calls(self):
        raw_output = """
class TestF:
    def test_cases(self):
        self.assertEqual(f(1), 2)
        self.assertTrue(f(2) > 0)
        self.assertFalse(f(0))
"""

        result = extract_test_assertions(raw_output, entry_point="f")

        self.assertEqual(
            result.assertions,
            [
                "assert f(1) == 2",
                "assert f(2) > 0",
                "assert not f(0)",
            ],
        )

    def test_extract_test_assertions_falls_back_to_doctest_examples(self):
        raw_output = """
>>> f(1)
2
>>> f(2)
3
"""

        result = extract_test_assertions(raw_output, entry_point="f")

        self.assertEqual(
            result.assertions,
            [
                "assert f(1) == 2",
                "assert f(2) == 3",
            ],
        )
        self.assertEqual(result.parse_method, "doctest")

    def test_extract_test_assertions_rejects_entry_point_leakage(self):
        raw_output = "```python\ndef f(x):\n    return x + 1\nassert f(1) == 2\n```"

        result = extract_test_assertions(raw_output, entry_point="f")

        self.assertTrue(result.entry_point_leak_detected)
        self.assertEqual(result.assertions, [])
        self.assertEqual(result.parse_method, "rejected_leak")

    def test_extract_test_assertions_reports_parse_error_when_nothing_is_salvageable(self):
        result = extract_test_assertions("assert f(1) ==\n(", entry_point="f")

        self.assertEqual(result.assertions, [])
        self.assertEqual(result.parse_method, "parse_error")
        self.assertIsNotNone(result.parse_error)


if __name__ == "__main__":
    unittest.main()
