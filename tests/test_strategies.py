"""Tests for the unified strategy interface and its first implementations."""

import unittest
from unittest.mock import patch

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.models.vllm_runner import GenerationResult
from tinycodescaling.strategies.best_of_n import BestOfNRandomPick
from tinycodescaling.strategies.base import StrategyConfig
from tinycodescaling.strategies.generated_test_selection import GeneratedTestSelectionStrategy
from tinycodescaling.strategies.greedy import GreedyStrategy
from tinycodescaling.strategies.public_test_selection import PublicTestSelectionStrategy
from tinycodescaling.strategies.temperature import TemperatureSamplingStrategy


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": prompts, "kwargs": kwargs})
        return self.outputs


class FakeSequentialRunner:
    def __init__(self, outputs_per_call):
        self.outputs_per_call = list(outputs_per_call)
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": prompts, "kwargs": kwargs})
        return self.outputs_per_call.pop(0)


class StrategyConfigTests(unittest.TestCase):
    def test_from_dict_preserves_extra_fields(self):
        config = StrategyConfig.from_dict(
            {"name": "temperature", "n": 4, "temperature": 0.8, "selector": "first"}
        )

        self.assertEqual(config.name, "temperature")
        self.assertEqual(config.n, 4)
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.extra, {"selector": "first"})
        self.assertEqual(config.as_dict()["selector"], "first")

    def test_from_dict_uses_generated_test_solution_count_defaults(self):
        config = StrategyConfig.from_dict(
            {
                "name": "generated_test_selection",
                "n_solutions": 6,
                "temperature_solutions": 0.7,
                "n_tests": 5,
            }
        )

        self.assertEqual(config.n, 6)
        self.assertEqual(config.temperature, 0.7)


class GreedyStrategyTests(unittest.TestCase):
    def test_greedy_strategy_returns_single_selected_candidate(self):
        task = CodeTask(task_id="HumanEval/0", prompt="return x", entry_point="f")
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=10,
                        completion_tokens=6,
                        finish_reason="stop",
                        latency_seconds=0.5,
                        cumulative_logprob=-1.0,
                    )
                ]
            ]
        )

        result = GreedyStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict({"name": "greedy"}),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.strategy_name, "greedy")
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.selected_code, "def f(x):\n    return x\n")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.total_completion_tokens, 6)
        self.assertEqual(result.selection_metadata["selection_method"], "greedy")
        self.assertEqual(runner.calls[0]["kwargs"]["temperature"], 0.0)


class TemperatureSamplingStrategyTests(unittest.TestCase):
    def test_temperature_strategy_keeps_all_candidates(self):
        task = CodeTask(task_id="HumanEval/1", prompt="return x", entry_point="f")
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=12,
                        completion_tokens=5,
                        finish_reason="stop",
                        latency_seconds=0.7,
                        cumulative_logprob=-1.1,
                    ),
                    GenerationResult(
                        text="def f(x):\n    return x + 1\n",
                        prompt_tokens=12,
                        completion_tokens=7,
                        finish_reason="length",
                        latency_seconds=0.7,
                        cumulative_logprob=-1.3,
                    ),
                ]
            ]
        )

        result = TemperatureSamplingStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict({"name": "temperature", "n": 2, "temperature": 0.8}),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.strategy_name, "temperature")
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.selected_code, "def f(x):\n    return x\n")
        self.assertEqual(result.total_completion_tokens, 12)
        self.assertEqual(result.selection_metadata["candidate_count"], 2)
        self.assertEqual(runner.calls[0]["kwargs"]["n"], 2)
        self.assertEqual(runner.calls[0]["kwargs"]["temperature"], 0.8)


class BestOfNRandomTests(unittest.TestCase):
    def test_best_of_n_random_keeps_all_candidates_and_selects_first(self):
        task = CodeTask(task_id="HumanEval/2", prompt="return x", entry_point="f")
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=14,
                        completion_tokens=4,
                        finish_reason="stop",
                        latency_seconds=0.9,
                        cumulative_logprob=-0.9,
                    ),
                    GenerationResult(
                        text="def f(x):\n    return x + 1\n",
                        prompt_tokens=14,
                        completion_tokens=6,
                        finish_reason="stop",
                        latency_seconds=0.9,
                        cumulative_logprob=-1.2,
                    ),
                ]
            ]
        )

        result = BestOfNRandomPick().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {"name": "best_of_n_random", "n": 2, "temperature": 0.8}
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.strategy_name, "best_of_n_random")
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.total_completion_tokens, 10)
        self.assertEqual(result.selection_metadata["selection_baseline"], "best_of_n_random")

    def test_best_of_n_random_passes_modern_sampling_kwargs_through_to_runner(self):
        task = CodeTask(task_id="HumanEval/2", prompt="return x", entry_point="f")
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=14,
                        completion_tokens=4,
                        finish_reason="stop",
                        latency_seconds=0.9,
                        cumulative_logprob=-0.9,
                    )
                ]
            ]
        )

        BestOfNRandomPick().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {
                    "name": "best_of_n_random",
                    "n": 1,
                    "temperature": 0.8,
                    "min_p": 0.1,
                    "top_p": 0.95,
                }
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(runner.calls[0]["kwargs"]["min_p"], 0.1)
        self.assertEqual(runner.calls[0]["kwargs"]["top_p"], 0.95)


class PublicTestSelectionStrategyTests(unittest.TestCase):
    @patch("tinycodescaling.strategies.public_test_selection.count_passing_public_tests")
    @patch("tinycodescaling.strategies.public_test_selection.extract_doctests_from_prompt")
    def test_public_test_selection_prefers_highest_pass_count(
        self, mock_extract_doctests, mock_count_passing
    ):
        task = CodeTask(task_id="HumanEval/3", prompt="prompt", entry_point="f")
        mock_extract_doctests.return_value = ["case-1", "case-2"]
        mock_count_passing.side_effect = [1, 2]
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=10,
                        completion_tokens=5,
                        finish_reason="stop",
                        latency_seconds=0.6,
                        cumulative_logprob=-2.0,
                    ),
                    GenerationResult(
                        text="def f(x):\n    return x + 1\n",
                        prompt_tokens=10,
                        completion_tokens=5,
                        finish_reason="stop",
                        latency_seconds=0.6,
                        cumulative_logprob=-1.0,
                    ),
                ]
            ]
        )

        result = PublicTestSelectionStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {"name": "public_test_selection", "n": 2, "temperature": 0.8}
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.selection_metadata["pass_counts"], [1, 2])
        self.assertEqual(result.selection_metadata["selection_method"], "public_test_pass_count")

    @patch("tinycodescaling.strategies.public_test_selection.extract_doctests_from_prompt")
    def test_public_test_selection_falls_back_when_no_public_tests(self, mock_extract_doctests):
        task = CodeTask(task_id="HumanEval/4", prompt="prompt", entry_point="f")
        mock_extract_doctests.return_value = []
        runner = FakeRunner(
            outputs=[
                [
                    GenerationResult(
                        text="def f(x):\n    return x\n",
                        prompt_tokens=10,
                        completion_tokens=5,
                        finish_reason="stop",
                        latency_seconds=0.6,
                        cumulative_logprob=-2.0,
                    )
                ]
            ]
        )

        result = PublicTestSelectionStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {"name": "public_test_selection", "n": 1, "temperature": 0.8}
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.selection_metadata["fallback_reason"], "no_public_tests")


class GeneratedTestSelectionStrategyTests(unittest.TestCase):
    @patch("tinycodescaling.strategies.generated_test_selection.run_assertions_in_sandbox")
    def test_generated_test_selection_prefers_highest_generated_test_score(
        self,
        mock_run_assertions,
    ):
        task = CodeTask(
            task_id="HumanEval/5",
            prompt='def f(x):\n    """doc"""\n',
            entry_point="f",
            canonical_solution="\n    return x + 1\n",
        )
        runner = FakeSequentialRunner(
            outputs_per_call=[
                [
                    [
                        GenerationResult(
                            text="def f(x):\n    return x\n",
                            prompt_tokens=10,
                            completion_tokens=5,
                            finish_reason="stop",
                            latency_seconds=0.6,
                            cumulative_logprob=-2.0,
                        ),
                        GenerationResult(
                            text="def f(x):\n    return x + 1\n",
                            prompt_tokens=10,
                            completion_tokens=7,
                            finish_reason="stop",
                            latency_seconds=0.6,
                            cumulative_logprob=-1.0,
                        ),
                    ]
                ],
                [
                    [
                        GenerationResult(
                            text="```python\nassert f(1) == 2\nassert f(2) == 3\n```",
                            prompt_tokens=14,
                            completion_tokens=8,
                            finish_reason="stop",
                            latency_seconds=0.4,
                            cumulative_logprob=-0.5,
                        )
                    ]
                ],
            ]
        )
        mock_run_assertions.side_effect = [
            type("Result", (), {"details": {"assertion_results": [False, False]}})(),
            type("Result", (), {"details": {"assertion_results": [True, True]}})(),
            type("Result", (), {"details": {"assertion_results": [True, True]}})(),
        ]

        result = GeneratedTestSelectionStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {
                    "name": "generated_test_selection",
                    "n_solutions": 2,
                    "n_tests": 2,
                    "temperature_solutions": 0.8,
                    "temperature_tests": 0.3,
                }
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.total_completion_tokens, 20)
        self.assertEqual(result.prompt_tokens, 24)
        self.assertEqual(result.selection_metadata["pass_counts"], [0, 2])
        self.assertEqual(result.selection_metadata["generated_test_canonical_pass_rate"], 1.0)
        self.assertEqual(result.selection_metadata["n_valid_generated_tests"], 2)
        self.assertEqual(
            result.candidates[1].extra["generated_test_assertion_results"],
            [1, 1],
        )

    def test_generated_test_selection_falls_back_when_generated_tests_are_invalid(self):
        task = CodeTask(
            task_id="HumanEval/6",
            prompt='def f(x):\n    """doc"""\n',
            entry_point="f",
        )
        runner = FakeSequentialRunner(
            outputs_per_call=[
                [
                    [
                        GenerationResult(
                            text="def f(x):\n    return x\n",
                            prompt_tokens=10,
                            completion_tokens=5,
                            finish_reason="stop",
                            latency_seconds=0.6,
                            cumulative_logprob=-2.0,
                        )
                    ]
                ],
                [
                    [
                        GenerationResult(
                            text="def f(x):\n    return x + 1\n",
                            prompt_tokens=14,
                            completion_tokens=9,
                            finish_reason="stop",
                            latency_seconds=0.4,
                            cumulative_logprob=-0.5,
                        )
                    ]
                ],
            ]
        )

        result = GeneratedTestSelectionStrategy().run(
            task=task,
            prompt="prompt",
            runner=runner,
            formatter=None,
            config=StrategyConfig.from_dict(
                {
                    "name": "generated_test_selection",
                    "n_solutions": 1,
                    "n_tests": 2,
                    "temperature_solutions": 0.8,
                    "temperature_tests": 0.3,
                }
            ),
            benchmark_name="humaneval_plus",
            extraction_backend="raw",
            max_tokens=64,
        )

        self.assertEqual(result.selected_index, 0)
        self.assertTrue(result.selection_metadata["generated_test_fallback_used"])
        self.assertEqual(
            result.selection_metadata["generated_test_fallback_reason"],
            "entry_point_leak_detected",
        )


if __name__ == "__main__":
    unittest.main()
