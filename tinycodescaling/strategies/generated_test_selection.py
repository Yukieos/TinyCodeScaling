"""Select among N candidates using model-generated unit tests."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.benchmarks.prompt_templates import render_generated_test_prompt
from tinycodescaling.execution.sandbox import run_assertions_in_sandbox
from tinycodescaling.execution.test_extract import extract_test_assertions
from tinycodescaling.strategies.base import CandidateSolution, Strategy, StrategyConfig, StrategyResult


class GeneratedTestSelectionStrategy(Strategy):
    """Generate candidate solutions plus verifier tests, then select by test pass count."""

    name = "generated_test_selection"

    def run(
        self,
        task: CodeTask,
        prompt: str,
        runner,
        formatter,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> StrategyResult:
        """Run generated-test selection for a single task."""
        return self.run_batch(
            tasks=[task],
            prompts=[prompt],
            runner=runner,
            formatter=formatter,
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
        formatter,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> list[StrategyResult]:
        """Generate N solutions, generate verifier tests, and select the best candidate."""
        n_solutions = int(config.extra.get("n_solutions", config.n))
        n_tests = int(config.extra.get("n_tests", 5))
        temperature_solutions = float(
            config.extra.get("temperature_solutions", config.temperature)
        )
        temperature_tests = float(config.extra.get("temperature_tests", 0.3))
        test_max_tokens = int(config.extra.get("test_max_tokens", 256))
        per_assertion_timeout_seconds = float(
            config.extra.get("per_assertion_timeout_seconds", 2.0)
        )
        generated_test_timeout_seconds = float(
            config.extra.get(
                "generated_test_timeout_seconds",
                max(5.0, n_tests * per_assertion_timeout_seconds + 1.0),
            )
        )

        candidate_batches = self._generate_candidate_batches(
            tasks=tasks,
            prompts=prompts,
            runner=runner,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            n=n_solutions,
            temperature=temperature_solutions,
            max_tokens=max_tokens,
        )
        test_prompts = [
            _format_generated_test_prompt(task, formatter=formatter, n_tests=n_tests)
            for task in tasks
        ]
        test_outputs = runner.generate(
            test_prompts,
            n=1,
            temperature=temperature_tests,
            max_tokens=test_max_tokens,
        )

        results: list[StrategyResult] = []
        for task, candidates, test_samples in zip(tasks, candidate_batches, test_outputs):
            test_generation = test_samples[0]
            generation_latency = candidates[0].latency_seconds if candidates else 0.0
            total_prompt_tokens = candidates[0].prompt_tokens + test_generation.prompt_tokens
            total_completion_tokens = (
                sum(candidate.completion_tokens for candidate in candidates)
                + test_generation.completion_tokens
            )

            if not task.entry_point:
                results.append(
                    self._build_result(
                        task_id=task.task_id,
                        candidates=candidates,
                        selected_index=0,
                        strategy_config=config,
                        selection_metadata={
                            "selection_method": "first_candidate_fallback",
                            "selected_index": 0,
                            "candidate_count": len(candidates),
                            "generated_test_fallback_used": True,
                            "generated_test_fallback_reason": "missing_entry_point",
                            "generated_tests_raw": test_generation.text,
                            "n_generated_tests_requested": n_tests,
                            "n_valid_generated_tests": 0,
                            "generated_test_parse_method": "skipped_missing_entry_point",
                            "test_generation_prompt_tokens": test_generation.prompt_tokens,
                            "test_generation_completion_tokens": test_generation.completion_tokens,
                        },
                        total_latency_seconds=(
                            generation_latency + test_generation.latency_seconds
                        ),
                        prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                    )
                )
                continue

            extracted_tests = extract_test_assertions(
                raw_output=test_generation.text,
                entry_point=task.entry_point,
            )
            assertions = extracted_tests.assertions
            if not assertions:
                results.append(
                    self._build_result(
                        task_id=task.task_id,
                        candidates=candidates,
                        selected_index=0,
                        strategy_config=config,
                        selection_metadata={
                            "selection_method": "first_candidate_fallback",
                            "selected_index": 0,
                            "candidate_count": len(candidates),
                            "generated_test_fallback_used": True,
                            "generated_test_fallback_reason": _fallback_reason(extracted_tests),
                            "generated_tests_raw": test_generation.text,
                            "generated_test_assertions": [],
                            "n_generated_tests_requested": n_tests,
                            "n_valid_generated_tests": 0,
                            "generated_test_parse_method": extracted_tests.parse_method,
                            "generated_test_parse_error": extracted_tests.parse_error,
                            "generated_test_entry_point_leak_detected": (
                                extracted_tests.entry_point_leak_detected
                            ),
                            "generated_test_total_assertions_found": (
                                extracted_tests.total_assertions_found
                            ),
                            "test_generation_prompt_tokens": test_generation.prompt_tokens,
                            "test_generation_completion_tokens": (
                                test_generation.completion_tokens
                            ),
                        },
                        total_latency_seconds=(
                            generation_latency + test_generation.latency_seconds
                        ),
                        prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                    )
                )
                continue

            started_at = time.perf_counter()
            pass_matrix: list[list[int]] = []
            pass_counts: list[int] = []
            updated_candidates: list[CandidateSolution] = []
            for candidate in candidates:
                assertion_run = run_assertions_in_sandbox(
                    code=candidate.code,
                    assertions=assertions,
                    timeout_seconds=generated_test_timeout_seconds,
                    per_assertion_timeout_seconds=per_assertion_timeout_seconds,
                )
                row = [
                    int(passed)
                    for passed in assertion_run.details.get(
                        "assertion_results",
                        [False for _ in assertions],
                    )
                ]
                if len(row) < len(assertions):
                    row.extend([0] * (len(assertions) - len(row)))
                pass_matrix.append(row)
                pass_count = sum(row)
                pass_counts.append(pass_count)
                updated_candidates.append(
                    replace(
                        candidate,
                        extra={
                            **candidate.extra,
                            "generated_test_pass_count": pass_count,
                            "generated_test_assertion_results": row,
                        },
                    )
                )

            selection_latency = time.perf_counter() - started_at
            selected_index, tie_breaker_used = _select_best_candidate(
                pass_counts,
                updated_candidates,
            )
            canonical_pass_count, canonical_pass_rate = _evaluate_canonical_solution(
                task=task,
                assertions=assertions,
                timeout_seconds=generated_test_timeout_seconds,
                per_assertion_timeout_seconds=per_assertion_timeout_seconds,
            )
            results.append(
                self._build_result(
                    task_id=task.task_id,
                    candidates=updated_candidates,
                    selected_index=selected_index,
                    strategy_config=config,
                    selection_metadata={
                        "selection_method": "generated_test_pass_count",
                        "selected_index": selected_index,
                        "candidate_count": len(updated_candidates),
                        "pass_matrix": pass_matrix,
                        "pass_counts": pass_counts,
                        "generated_tests_raw": test_generation.text,
                        "generated_test_assertions": assertions,
                        "n_generated_tests_requested": n_tests,
                        "n_valid_generated_tests": len(assertions),
                        "generated_test_parse_method": extracted_tests.parse_method,
                        "generated_test_parse_error": extracted_tests.parse_error,
                        "generated_test_entry_point_leak_detected": (
                            extracted_tests.entry_point_leak_detected
                        ),
                        "generated_test_total_assertions_found": (
                            extracted_tests.total_assertions_found
                        ),
                        "generated_test_discriminative": len(set(pass_counts)) > 1,
                        "generated_test_fallback_used": False,
                        "generated_test_canonical_pass_count": canonical_pass_count,
                        "generated_test_canonical_pass_rate": canonical_pass_rate,
                        "test_generation_prompt_tokens": test_generation.prompt_tokens,
                        "test_generation_completion_tokens": (
                            test_generation.completion_tokens
                        ),
                        "tie_breaker_used": tie_breaker_used,
                    },
                    total_latency_seconds=(
                        generation_latency
                        + test_generation.latency_seconds
                        + selection_latency
                    ),
                    prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                )
            )
        return results


def _format_generated_test_prompt(task: CodeTask, formatter, n_tests: int) -> str:
    """Render the generated-test prompt with the same chat-template path as solutions."""
    user_prompt = render_generated_test_prompt(
        problem_prompt=task.prompt,
        entry_point=task.entry_point or "candidate_solution",
        n_tests=n_tests,
    )
    if formatter is None:
        return user_prompt
    return formatter.format_user_text(user_prompt)


def _fallback_reason(extracted_tests) -> str:
    """Summarize why generated-test parsing failed for one task."""
    if extracted_tests.entry_point_leak_detected:
        return "entry_point_leak_detected"
    if extracted_tests.parse_method == "parse_error":
        return "parse_error"
    return "no_valid_generated_tests"


def _evaluate_canonical_solution(
    task: CodeTask,
    assertions: list[str],
    timeout_seconds: float,
    per_assertion_timeout_seconds: float,
) -> tuple[int | None, float | None]:
    """Measure how many generated tests pass on the benchmark canonical solution."""
    if not task.canonical_solution or not assertions:
        return None, None
    canonical_code = _build_canonical_code(task)
    result = run_assertions_in_sandbox(
        code=canonical_code,
        assertions=assertions,
        timeout_seconds=timeout_seconds,
        per_assertion_timeout_seconds=per_assertion_timeout_seconds,
    )
    assertion_results = result.details.get("assertion_results", [])
    if not assertion_results:
        return 0, 0.0
    passed = sum(bool(value) for value in assertion_results)
    return passed, passed / len(assertion_results)


def _build_canonical_code(task: CodeTask) -> str:
    """Reconstruct full canonical code from the benchmark prompt plus canonical body."""
    prompt_prefix = task.prompt.rstrip("\n")
    canonical_body = (task.canonical_solution or "").lstrip("\n")
    return f"{prompt_prefix}\n{canonical_body}".rstrip() + "\n"


def _select_best_candidate(
    pass_counts: list[int],
    candidates: list[CandidateSolution],
) -> tuple[int, str | None]:
    """Pick the best candidate by pass count, then break ties by logprob."""
    best_indices: list[int] = []
    best_score = -1
    for index, pass_count in enumerate(pass_counts):
        if pass_count > best_score:
            best_indices = [index]
            best_score = pass_count
        elif pass_count == best_score:
            best_indices.append(index)

    if len(best_indices) == 1:
        return best_indices[0], None

    selected_index = max(
        best_indices,
        key=lambda index: (
            candidates[index].cumulative_logprob
            if candidates[index].cumulative_logprob is not None
            else -math.inf
        ),
    )
    return selected_index, "cumulative_logprob"
