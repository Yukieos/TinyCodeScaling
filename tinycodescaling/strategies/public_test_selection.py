"""Select among N candidates by running prompt-derived public doctests."""

from __future__ import annotations

import math
import time

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.execution.test_runner import count_passing_public_tests, extract_doctests_from_prompt
from tinycodescaling.strategies.base import Strategy, StrategyConfig, StrategyResult


class PublicTestSelectionStrategy(Strategy):
    """Generate N candidates and select the one that best matches public doctests."""

    name = "public_test_selection"

    def run(
        self,
        task: CodeTask,
        prompt: str,
        runner,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> StrategyResult:
        """Run public-test-based selection for a single task."""
        return self.run_batch(
            tasks=[task],
            prompts=[prompt],
            runner=runner,
            config=config,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            max_tokens=max_tokens,
        )[0]

    def run_batch(
        self,
        tasks: list[CodeTask],
        prompts: list[str],
        runner,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> list[StrategyResult]:
        """Generate N candidates, score them on public tests, and select the best one."""
        candidate_batches = self._generate_candidate_batches(
            tasks=tasks,
            prompts=prompts,
            runner=runner,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            n=config.n,
            temperature=config.temperature,
            max_tokens=max_tokens,
        )
        results: list[StrategyResult] = []
        for task, candidates in zip(tasks, candidate_batches):
            public_tests = extract_doctests_from_prompt(task.prompt)
            generation_latency = candidates[0].latency_seconds if candidates else 0.0
            if not public_tests:
                results.append(
                    self._build_result(
                        task_id=task.task_id,
                        candidates=candidates,
                        selected_index=0,
                        strategy_config=config,
                        selection_metadata={
                            "selection_method": "first_candidate_fallback",
                            "fallback_reason": "no_public_tests",
                            "selected_index": 0,
                            "candidate_count": len(candidates),
                            "n_public_tests": 0,
                        },
                        total_latency_seconds=generation_latency,
                    )
                )
                continue

            started_at = time.perf_counter()
            pass_counts = [
                count_passing_public_tests(candidate.code, public_tests)
                for candidate in candidates
            ]
            selection_latency = time.perf_counter() - started_at
            selected_index = _select_best_candidate(pass_counts, candidates)
            results.append(
                self._build_result(
                    task_id=task.task_id,
                    candidates=candidates,
                    selected_index=selected_index,
                    strategy_config=config,
                    selection_metadata={
                        "selection_method": "public_test_pass_count",
                        "pass_counts": pass_counts,
                        "n_public_tests": len(public_tests),
                        "selected_index": selected_index,
                        "candidate_count": len(candidates),
                        "tie_breaker": "cumulative_logprob",
                        "public_test_discriminative": len(set(pass_counts)) > 1,
                    },
                    total_latency_seconds=generation_latency + selection_latency,
                )
            )
        return results


def _select_best_candidate(pass_counts: list[int], candidates) -> int:
    """Pick the highest-scoring candidate and break ties by cumulative logprob."""
    best_index = 0
    best_score = -1
    best_logprob = -math.inf
    for index, (pass_count, candidate) in enumerate(zip(pass_counts, candidates)):
        candidate_logprob = (
            candidate.cumulative_logprob
            if candidate.cumulative_logprob is not None
            else -math.inf
        )
        if pass_count > best_score:
            best_index = index
            best_score = pass_count
            best_logprob = candidate_logprob
            continue
        if pass_count == best_score and candidate_logprob > best_logprob:
            best_index = index
            best_logprob = candidate_logprob
    return best_index
