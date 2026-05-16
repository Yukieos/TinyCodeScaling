"""Tests for doctest extraction and public-test counting helpers."""

from __future__ import annotations

import importlib.util
import unittest

from tinycodescaling.execution.test_runner import (
    PublicTestCase,
    count_passing_public_tests,
    extract_doctests_from_prompt,
)


HAS_EVALPLUS = importlib.util.find_spec("evalplus") is not None


class PublicTestExtractionTests(unittest.TestCase):
    def test_extract_doctests_from_prompt_reads_function_docstring_examples(self):
        prompt = """def f(x):\n    \"\"\"Example.\n    >>> f(1)\n    2\n    >>> f(2)\n    3\n    \"\"\"\n"""

        test_cases = extract_doctests_from_prompt(prompt)

        self.assertEqual(
            test_cases,
            [
                PublicTestCase(source="f(1)", want="2\n"),
                PublicTestCase(source="f(2)", want="3\n"),
            ],
        )


@unittest.skipUnless(HAS_EVALPLUS, "evalplus runtime not installed")
class PublicTestExecutionTests(unittest.TestCase):
    def test_count_passing_public_tests_counts_successes(self):
        code = "def f(x):\n    return x + 1\n"
        public_tests = [
            PublicTestCase(source="f(1)", want="2\n"),
            PublicTestCase(source="f(2)", want="3\n"),
        ]

        passed = count_passing_public_tests(code, public_tests)

        self.assertEqual(passed, 2)


if __name__ == "__main__":
    unittest.main()
