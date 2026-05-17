import math
import unittest

from tinycodescaling.metrics.aggregate import (
    aggregate_seed_summaries,
    derive_metrics_from_candidate_results,
    derive_selection_audit_metrics,
)


class AggregateTests(unittest.TestCase):
    def test_aggregate_seed_summaries_handles_infinite_tokens_per_solved(self):
        seed_summaries = [
            {
                "pass_at_1_base": 0.0,
                "pass_at_1_plus": 0.0,
                "prompt_tokens_per_problem": 10.0,
                "completion_tokens_per_problem": 20.0,
                "total_tokens_per_problem": 30.0,
                "latency_seconds_per_problem": 1.0,
                "tokens_per_solved_plus": math.inf,
                "quality_per_1k_generated_tokens": 0.0,
            },
            {
                "pass_at_1_base": 0.0,
                "pass_at_1_plus": 0.0,
                "prompt_tokens_per_problem": 10.0,
                "completion_tokens_per_problem": 20.0,
                "total_tokens_per_problem": 30.0,
                "latency_seconds_per_problem": 1.0,
                "tokens_per_solved_plus": math.inf,
                "quality_per_1k_generated_tokens": 0.0,
            },
        ]

        aggregate = aggregate_seed_summaries(seed_summaries)

        self.assertTrue(math.isinf(aggregate["tokens_per_solved_plus"]["mean"]))
        self.assertEqual(aggregate["tokens_per_solved_plus"]["std"], 0.0)

    def test_derive_metrics_from_candidate_results_computes_oracle_and_public_test_rates(self):
        records = [
            {
                "task_id": "HumanEval/0",
                "strategy_selected_index": 1,
                "strategy_selection_metadata": {"pass_counts": [0, 1]},
            },
            {
                "task_id": "HumanEval/1",
                "strategy_selected_index": 0,
                "strategy_selection_metadata": {"fallback_reason": "no_public_tests"},
            },
        ]
        eval_results = {
            "HumanEval/0": [
                {"base_status": "fail", "plus_status": "fail"},
                {"base_status": "pass", "plus_status": "pass"},
            ],
            "HumanEval/1": [
                {"base_status": "pass", "plus_status": "fail"},
                {"base_status": "fail", "plus_status": "fail"},
            ],
        }

        metrics = derive_metrics_from_candidate_results(records, eval_results)

        self.assertEqual(metrics["pass_at_1_base"], 1.0)
        self.assertEqual(metrics["pass_at_1_plus"], 0.5)
        self.assertEqual(metrics["oracle_pass_at_n_base"], 1.0)
        self.assertEqual(metrics["oracle_pass_at_n_plus"], 0.5)
        self.assertEqual(metrics["public_test_discrimination_rate"], 1.0)
        self.assertEqual(metrics["public_test_fallback_rate"], 0.5)

    def test_derive_selection_audit_metrics_summarizes_generated_test_quality(self):
        records = [
            {
                "strategy_selection_metadata": {
                    "selection_method": "generated_test_pass_count",
                    "generated_test_parse_method": "ast_assert_walk",
                    "generated_test_entry_point_leak_detected": False,
                    "generated_test_fallback_used": False,
                    "generated_test_discriminative": True,
                    "generated_test_canonical_pass_rate": 1.0,
                    "n_valid_generated_tests": 5,
                }
            },
            {
                "strategy_selection_metadata": {
                    "selection_method": "first_candidate_fallback",
                    "generated_test_parse_method": "parse_error",
                    "generated_test_entry_point_leak_detected": True,
                    "generated_test_fallback_used": True,
                    "generated_test_discriminative": False,
                    "generated_test_canonical_pass_rate": 0.5,
                    "n_valid_generated_tests": 0,
                    "tie_breaker_used": "cumulative_logprob",
                }
            },
        ]

        metrics = derive_selection_audit_metrics(records)

        self.assertEqual(metrics["generated_test_parse_failure_rate"], 0.5)
        self.assertEqual(metrics["generated_test_entry_point_leak_rate"], 0.5)
        self.assertEqual(metrics["generated_test_fallback_rate"], 0.5)
        self.assertEqual(metrics["generated_test_discrimination_rate"], 0.5)
        self.assertEqual(metrics["generated_test_valid_tests_per_task"], 2.5)
        self.assertEqual(metrics["generated_test_tie_rate"], 0.5)
        self.assertEqual(metrics["generated_test_canonical_pass_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
