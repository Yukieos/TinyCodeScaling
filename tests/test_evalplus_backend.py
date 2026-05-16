import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycodescaling.evaluators.evalplus_backend import evaluate_with_evalplus


class EvalPlusBackendTests(unittest.TestCase):
    @patch("tinycodescaling.evaluators.evalplus_backend.subprocess.run")
    def test_evaluate_with_evalplus_prefers_active_interpreter_module(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[sys.executable, "-m", "evalplus.evaluate"],
            returncode=0,
            stdout="Base\n{'pass@1': 0.5}\nBase + Extra\n{'pass@1': 0.25}\n",
            stderr="",
        )

        result = evaluate_with_evalplus(
            samples_path=Path("samples.jsonl"),
            dataset="humaneval",
            timeout_seconds=5,
            parallel=None,
        )

        self.assertEqual(result["pass_at_1_base"], 0.5)
        self.assertEqual(result["pass_at_1_plus"], 0.25)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "evalplus.evaluate"])

    @patch("tinycodescaling.evaluators.evalplus_backend.subprocess.run")
    def test_evaluate_with_evalplus_falls_back_to_console_script_when_module_missing(self, mock_run):
        mock_run.side_effect = [
            subprocess.CalledProcessError(
                returncode=1,
                cmd=[sys.executable, "-m", "evalplus.evaluate"],
                output="",
                stderr="/usr/bin/python: No module named evalplus.evaluate",
            ),
            subprocess.CompletedProcess(
                args=["evalplus.evaluate"],
                returncode=0,
                stdout="Base\n{'pass@1': 0.75}\nBase + Extra\n{'pass@1': 0.5}\n",
                stderr="",
            ),
        ]

        result = evaluate_with_evalplus(
            samples_path=Path("samples.jsonl"),
            dataset="humaneval",
            timeout_seconds=5,
            parallel=8,
        )

        self.assertEqual(result["pass_at_1_base"], 0.75)
        self.assertEqual(result["pass_at_1_plus"], 0.5)
        first_command = mock_run.call_args_list[0].args[0]
        second_command = mock_run.call_args_list[1].args[0]
        self.assertEqual(first_command[:3], [sys.executable, "-m", "evalplus.evaluate"])
        self.assertEqual(second_command[0], "evalplus.evaluate")


if __name__ == "__main__":
    unittest.main()
