"""Tests for the sandboxed execution helpers."""

from __future__ import annotations

import importlib.util
import unittest

from tinycodescaling.execution.sandbox import run_assertions_in_sandbox, run_in_sandbox


HAS_EVALPLUS = importlib.util.find_spec("evalplus") is not None


@unittest.skipUnless(HAS_EVALPLUS, "evalplus runtime not installed")
class SandboxTests(unittest.TestCase):
    def test_run_in_sandbox_passes_normal_code(self):
        result = run_in_sandbox(
            code="def f(x):\n    return x + 1\n",
            test_code="assert f(1) == 2",
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "pass")

    def test_run_in_sandbox_times_out_on_infinite_loop(self):
        result = run_in_sandbox(
            code="while True:\n    pass\n",
            test_code="",
            timeout_seconds=0.5,
        )
        self.assertEqual(result.status, "timeout")

    def test_run_in_sandbox_blocks_file_access(self):
        result = run_in_sandbox(
            code="",
            test_code="open('/etc/passwd').read()",
        )
        self.assertEqual(result.status, "blocked_operation")

    def test_run_in_sandbox_blocks_network_access(self):
        result = run_in_sandbox(
            code="",
            test_code="import urllib.request\nurllib.request.urlopen('http://example.com')",
        )
        self.assertEqual(result.status, "blocked_operation")

    def test_run_in_sandbox_blocks_subprocess_access(self):
        result = run_in_sandbox(
            code="import os",
            test_code="os.system('echo hi')",
        )
        self.assertEqual(result.status, "blocked_operation")

    def test_run_in_sandbox_reports_syntax_error(self):
        result = run_in_sandbox(
            code="def f(:\n    pass\n",
            test_code="",
        )
        self.assertEqual(result.status, "syntax_error")

    def test_run_in_sandbox_reports_runtime_error(self):
        result = run_in_sandbox(
            code="def f():\n    return missing_name\n",
            test_code="f()",
        )
        self.assertEqual(result.status, "runtime_error")

    def test_run_in_sandbox_reports_memory_error(self):
        result = run_in_sandbox(
            code="x = [0] * 10**10\n",
            test_code="",
            timeout_seconds=1.0,
            maximum_memory_bytes=128 * 1024 * 1024,
        )
        self.assertIn(result.status, {"memory_error", "runtime_error"})

    def test_run_assertions_in_sandbox_returns_per_assertion_results(self):
        result = run_assertions_in_sandbox(
            code="def f(x):\n    return x + 1\n",
            assertions=["assert f(1) == 2", "assert f(2) == 4"],
            timeout_seconds=1.0,
            per_assertion_timeout_seconds=0.5,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.details["assertion_results"], [True, False])


if __name__ == "__main__":
    unittest.main()
