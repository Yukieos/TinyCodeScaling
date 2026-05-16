import unittest
from unittest.mock import patch

from tinycodescaling.execution.code_extract import (
    build_benchmark_sample,
    build_evalplus_sample,
    extract_code,
    extract_python_solution,
)


class CodeExtractTests(unittest.TestCase):
    def test_evalplus_sample_uses_solution_when_entry_point_present(self):
        sample = build_evalplus_sample("HumanEval/0", "def f(x):\n    return x\n", entry_point="f")
        self.assertIn("solution", sample)
        self.assertNotIn("completion", sample)

    def test_evalplus_sample_uses_completion_for_body_only(self):
        sample = build_evalplus_sample("HumanEval/0", "return x + 1\n", entry_point="f")
        self.assertIn("completion", sample)
        self.assertNotIn("solution", sample)

    def test_build_benchmark_sample_dispatches_evalplus(self):
        sample = build_benchmark_sample(
            task_id="HumanEval/0",
            benchmark="humaneval_plus",
            extracted_code="def f(x):\n    return x\n",
            entry_point="f",
        )
        self.assertEqual(sample["task_id"], "HumanEval/0")
        self.assertIn("solution", sample)

    def test_extract_code_falls_back_to_raw_if_evalplus_missing(self):
        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "evalplus.sanitize":
                raise ImportError("evalplus unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = extract_code(
                raw_output="Here is my answer",
                benchmark="humaneval_plus",
                entry_point="f",
            )
        self.assertEqual(result["backend"], "evalplus")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["code"], "Here is my answer\n")

    @patch("tinycodescaling.execution.code_extract._extract_with_evalplus")
    def test_extract_code_dispatches_to_evalplus_backend(self, mock_extract):
        mock_extract.return_value = {
            "code": "def f(x):\n    return x\n",
            "backend": "evalplus",
            "method": "evalplus_sanitize",
            "fallback_used": False,
        }
        result = extract_code(
            raw_output="ignored",
            benchmark="humaneval_plus",
            entry_point="f",
            task_id="HumanEval/0",
        )
        self.assertEqual(result["method"], "evalplus_sanitize")
        mock_extract.assert_called_once()

    def test_extract_code_supports_raw_backend(self):
        result = extract_code(
            raw_output="def f(x):\n    return x\n",
            benchmark="humaneval_plus",
            extraction_backend="raw",
        )
        self.assertEqual(result["backend"], "raw")
        self.assertEqual(result["method"], "raw_passthrough")

    def test_extract_python_solution_alias_returns_code_string(self):
        self.assertEqual(extract_python_solution(""), "")

    def test_livecodebench_sample_shape(self):
        sample = build_benchmark_sample(
            task_id="q1",
            benchmark="livecodebench_subset",
            extracted_code="print(1)\n",
        )
        self.assertEqual(sample, {"question_id": "q1", "code_list": ["print(1)\n"]})


if __name__ == "__main__":
    unittest.main()
