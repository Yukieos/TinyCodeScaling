"""Aggregate per-task and per-seed metrics into final experiment summaries."""

from __future__ import annotations

import math
import statistics

from tinycodescaling.metrics.latency import average_latency_seconds
from tinycodescaling.metrics.pass_at_k import estimate_pass_at_k
from tinycodescaling.metrics.token_cost import (
    aggregate_token_usage,
    quality_per_1k_generated_tokens,
    tokens_per_solved,
)


def summarize_seed_records(
    records: list[dict],
    n_problems: int,
    pass_at_1_base: float,
    pass_at_1_plus: float,
    seed: int,
    samples_path: str,
    evaluator_cache_path: str | None,
    evaluation: dict | None = None,
) -> dict:
    """Collapse one seed's raw task records into a single summary row."""
    if evaluation and evaluation.get("eval_results"):
        derived_metrics = derive_metrics_from_candidate_results(records, evaluation["eval_results"])
        pass_at_1_base = derived_metrics["pass_at_1_base"]
        pass_at_1_plus = derived_metrics["pass_at_1_plus"]
    else:
        derived_metrics = {}

    token_summary = aggregate_token_usage(records)
    latency_per_problem = average_latency_seconds(records)
    solved_plus = pass_at_1_plus * n_problems

    summary = {
        "seed": seed,
        "samples_path": samples_path,
        "evaluator_cache_path": evaluator_cache_path,
        "pass_at_1_base": pass_at_1_base,
        "pass_at_1_plus": pass_at_1_plus,
        "prompt_tokens_per_problem": token_summary["prompt_tokens_per_problem"],
        "completion_tokens_per_problem": token_summary["completion_tokens_per_problem"],
        "total_tokens_per_problem": token_summary["total_tokens_per_problem"],
        "latency_seconds_per_problem": latency_per_problem,
        "tokens_per_solved_plus": tokens_per_solved(
            token_summary["completion_tokens"], solved_plus
        ),
        "quality_per_1k_generated_tokens": quality_per_1k_generated_tokens(
            pass_at_1_plus,
            token_summary["completion_tokens_per_problem"],
        ),
    }
    summary.update(derived_metrics)
    summary.update(derive_selection_audit_metrics(records))
    return summary


def aggregate_seed_summaries(seed_summaries: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean/std aggregates across seed-level summaries."""
    if not seed_summaries:
        return {}

    aggregate: dict[str, dict[str, float]] = {}
    numeric_keys = [
        "pass_at_1_base",
        "pass_at_1_plus",
        "prompt_tokens_per_problem",
        "completion_tokens_per_problem",
        "total_tokens_per_problem",
        "latency_seconds_per_problem",
        "tokens_per_solved_plus",
        "quality_per_1k_generated_tokens",
    ]
    optional_numeric_keys = [
        "oracle_pass_at_n_base",
        "oracle_pass_at_n_plus",
        "candidate_pass_at_1_base",
        "candidate_pass_at_1_plus",
        "public_test_discrimination_rate",
        "public_test_fallback_rate",
        "generated_test_canonical_pass_rate",
        "generated_test_parse_failure_rate",
        "generated_test_entry_point_leak_rate",
        "generated_test_fallback_rate",
        "generated_test_discrimination_rate",
        "generated_test_valid_tests_per_task",
        "generated_test_tie_rate",
    ]
    numeric_keys.extend(
        key for key in optional_numeric_keys if all(key in summary for summary in seed_summaries)
    )

    for key in numeric_keys:
        values = [float(summary[key]) for summary in seed_summaries]
        aggregate[key] = _summarize_numeric_values(values)

    return aggregate


def attach_evalplus_results(records: list[dict], eval_results: dict | None) -> list[dict]:
    """Attach candidate-level official evaluator outcomes to raw task records."""
    if not eval_results:
        return records

    for record in records:
        task_results = eval_results.get(record["task_id"], [])
        candidate_base_statuses = [result["base_status"] for result in task_results]
        candidate_plus_statuses = [result["plus_status"] for result in task_results]
        record["candidate_base_statuses"] = candidate_base_statuses
        record["candidate_plus_statuses"] = candidate_plus_statuses
        if task_results:
            selected_result = task_results[record["strategy_selected_index"]]
            record["selected_base_status"] = selected_result["base_status"]
            record["selected_plus_status"] = selected_result["plus_status"]
            record["oracle_base_pass"] = any(
                result["base_status"] == "pass" for result in task_results
            )
            record["oracle_plus_pass"] = any(_result_passes_plus(result) for result in task_results)
    return records


def derive_metrics_from_candidate_results(records: list[dict], eval_results: dict) -> dict[str, float]:
    """Derive selected, candidate-average, and oracle metrics from official candidate results."""
    attach_evalplus_results(records, eval_results)
    if not records:
        return {}

    selected_base = 0
    selected_plus = 0
    oracle_base = 0
    oracle_plus = 0
    candidate_pass_rate_base_total = 0.0
    candidate_pass_rate_plus_total = 0.0
    candidate_counts: list[int] = []

    public_test_evaluated_tasks = 0
    public_test_fallbacks = 0
    public_test_discriminative = 0

    for record in records:
        base_statuses = record.get("candidate_base_statuses", [])
        plus_statuses = record.get("candidate_plus_statuses", [])
        if not base_statuses:
            continue

        candidate_counts.append(len(base_statuses))
        selected_index = int(record["strategy_selected_index"])
        selected_base += int(base_statuses[selected_index] == "pass")
        selected_plus += int(_statuses_pass_plus(base_statuses, plus_statuses, selected_index))
        oracle_base += int(any(status == "pass" for status in base_statuses))
        oracle_plus += int(
            any(_statuses_pass_plus(base_statuses, plus_statuses, index) for index in range(len(base_statuses)))
        )

        base_correct = sum(status == "pass" for status in base_statuses)
        plus_correct = sum(
            _statuses_pass_plus(base_statuses, plus_statuses, index)
            for index in range(len(base_statuses))
        )
        candidate_pass_rate_base_total += estimate_pass_at_k(len(base_statuses), base_correct, 1)
        candidate_pass_rate_plus_total += estimate_pass_at_k(len(base_statuses), plus_correct, 1)

        selection_metadata = record.get("strategy_selection_metadata", {}) or {}
        pass_counts = selection_metadata.get("pass_counts")
        if pass_counts is not None:
            public_test_evaluated_tasks += 1
            if selection_metadata.get("fallback_reason") == "no_public_tests":
                public_test_fallbacks += 1
            if len(set(pass_counts)) > 1:
                public_test_discriminative += 1
        elif selection_metadata.get("fallback_reason") == "no_public_tests":
            public_test_fallbacks += 1

    n_records = len(records)
    metrics = {
        "pass_at_1_base": selected_base / n_records,
        "pass_at_1_plus": selected_plus / n_records,
        "candidate_pass_at_1_base": candidate_pass_rate_base_total / n_records,
        "candidate_pass_at_1_plus": candidate_pass_rate_plus_total / n_records,
        "oracle_pass_at_n_base": oracle_base / n_records,
        "oracle_pass_at_n_plus": oracle_plus / n_records,
    }
    if candidate_counts and len(set(candidate_counts)) == 1:
        metrics["oracle_candidate_count"] = float(candidate_counts[0])
    if public_test_evaluated_tasks:
        metrics["public_test_discrimination_rate"] = (
            public_test_discriminative / public_test_evaluated_tasks
        )
    if public_test_fallbacks or any(
        (record.get("strategy_selection_metadata", {}) or {}).get("fallback_reason") == "no_public_tests"
        for record in records
    ):
        metrics["public_test_fallback_rate"] = public_test_fallbacks / n_records
    return metrics


def derive_selection_audit_metrics(records: list[dict]) -> dict[str, float]:
    """Aggregate strategy-specific audit metrics stored in raw selection metadata."""
    if not records:
        return {}

    generated_records = []
    for record in records:
        metadata = record.get("strategy_selection_metadata", {}) or {}
        if (
            "generated_test_parse_method" in metadata
            or "generated_test_fallback_used" in metadata
            or metadata.get("selection_method") == "generated_test_pass_count"
        ):
            generated_records.append(metadata)

    if not generated_records:
        return {}

    n_records = len(generated_records)
    canonical_pass_rates = [
        float(metadata["generated_test_canonical_pass_rate"])
        for metadata in generated_records
        if metadata.get("generated_test_canonical_pass_rate") is not None
    ]
    parse_failures = sum(
        metadata.get("generated_test_parse_method") == "parse_error"
        for metadata in generated_records
    )
    entry_point_leaks = sum(
        bool(metadata.get("generated_test_entry_point_leak_detected"))
        for metadata in generated_records
    )
    fallbacks = sum(
        bool(metadata.get("generated_test_fallback_used"))
        for metadata in generated_records
    )
    discriminative = sum(
        bool(metadata.get("generated_test_discriminative"))
        for metadata in generated_records
    )
    valid_tests_total = sum(
        int(metadata.get("n_valid_generated_tests", 0)) for metadata in generated_records
    )
    ties = sum(bool(metadata.get("tie_breaker_used")) for metadata in generated_records)

    metrics = {
        "generated_test_parse_failure_rate": parse_failures / n_records,
        "generated_test_entry_point_leak_rate": entry_point_leaks / n_records,
        "generated_test_fallback_rate": fallbacks / n_records,
        "generated_test_discrimination_rate": discriminative / n_records,
        "generated_test_valid_tests_per_task": valid_tests_total / n_records,
        "generated_test_tie_rate": ties / n_records,
    }
    if canonical_pass_rates:
        metrics["generated_test_canonical_pass_rate"] = sum(canonical_pass_rates) / len(
            canonical_pass_rates
        )
    return metrics


def _summarize_numeric_values(values: list[float]) -> dict[str, float]:
    """Summarize numeric values while preserving infinities instead of crashing."""
    finite_values = [value for value in values if math.isfinite(value)]
    if len(finite_values) == len(values):
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    if not finite_values:
        return {"mean": math.inf, "std": 0.0}

    return {"mean": math.inf, "std": math.inf}


def _statuses_pass_plus(base_statuses: list[str], plus_statuses: list[str | None], index: int) -> bool:
    """Return whether one candidate passed both base and plus tests."""
    return base_statuses[index] == "pass" and plus_statuses[index] == "pass"


def _result_passes_plus(result: dict) -> bool:
    """Return whether one EvalPlus result entry passed both base and plus tests."""
    return result["base_status"] == "pass" and result["plus_status"] == "pass"
