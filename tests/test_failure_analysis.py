"""Tests for raw-record failure analysis helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinycodescaling.reports.failure_analysis import (
    analyze_failures,
    build_failure_report_markdown,
    infer_summary_path,
    write_failure_artifacts,
)


class FailureAnalysisTests(unittest.TestCase):
    def test_analyze_failures_groups_primary_and_tag_buckets(self):
        records = [
            {
                "seed": 11,
                "task_id": "HumanEval/0",
                "selected_plus_status": "fail",
                "oracle_plus_pass": True,
                "extraction_fallback_used": False,
                "finish_reason": "stop",
                "strategy_name": "generated_test_selection",
                "strategy_selected_index": 0,
                "candidate_count": 4,
                "strategy_selection_metadata": {
                    "selection_method": "first_candidate_fallback",
                    "generated_test_parse_method": "parse_error",
                    "generated_test_fallback_used": True,
                    "generated_test_fallback_reason": "parse_error",
                    "generated_test_entry_point_leak_detected": False,
                    "generated_test_canonical_pass_rate": 0.0,
                    "pass_counts": [0, 0, 0, 0],
                },
            },
            {
                "seed": 11,
                "task_id": "HumanEval/1",
                "selected_plus_status": "fail",
                "oracle_plus_pass": False,
                "extraction_fallback_used": True,
                "finish_reason": "length",
                "strategy_name": "best_of_n_random",
                "strategy_selected_index": 0,
                "candidate_count": 8,
                "strategy_selection_metadata": {
                    "selection_method": "first_candidate",
                },
            },
            {
                "seed": 12,
                "task_id": "HumanEval/2",
                "selected_plus_status": "pass",
                "oracle_plus_pass": True,
                "extraction_fallback_used": False,
                "finish_reason": "stop",
                "strategy_name": "generated_test_selection",
                "strategy_selected_index": 1,
                "candidate_count": 4,
                "strategy_selection_metadata": {
                    "selection_method": "generated_test_pass_count",
                    "generated_test_parse_method": "ast_assert_walk",
                    "generated_test_fallback_used": False,
                    "generated_test_entry_point_leak_detected": False,
                    "generated_test_canonical_pass_rate": 1.0,
                    "pass_counts": [3, 4, 4, 4],
                },
            },
            {
                "seed": 12,
                "task_id": "HumanEval/3",
                "selected_plus_status": "fail",
                "oracle_plus_pass": False,
                "extraction_fallback_used": False,
                "finish_reason": "stop",
                "strategy_name": "public_test_selection",
                "strategy_selected_index": 0,
                "candidate_count": 4,
                "strategy_selection_metadata": {
                    "selection_method": "first_candidate_fallback",
                    "fallback_reason": "no_public_tests",
                },
            },
        ]
        summary = {
            "metadata": {
                "experiment_name": "demo_run",
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "strategy": "generated_test_selection",
            },
            "benchmark": {"name": "humaneval_plus"},
        }

        analysis = analyze_failures(records, summary=summary, target_status="plus", example_limit=2)

        self.assertEqual(analysis["headline"]["total_records"], 4)
        self.assertEqual(analysis["headline"]["selected_passes"], 1)
        self.assertEqual(analysis["headline"]["selected_failures"], 3)
        self.assertEqual(analysis["headline"]["selection_miss_count"], 1)

        primary = {bucket["key"]: bucket["count"] for bucket in analysis["primary_failure_breakdown"]}
        self.assertEqual(primary["generated_test_parse_failure"], 1)
        self.assertEqual(primary["extraction_fallback"], 1)
        self.assertEqual(primary["public_test_no_public_tests"], 1)

        tags = {bucket["key"]: bucket["count"] for bucket in analysis["tag_breakdown"]}
        self.assertEqual(tags["selection_miss"], 1)
        self.assertEqual(tags["generated_test_selection_fallback"], 1)
        self.assertNotIn("generated_test_no_discrimination", tags)
        self.assertEqual(tags["public_test_no_public_tests"], 1)
        self.assertEqual(tags["generation_truncated"], 1)

        report = build_failure_report_markdown(analysis)
        self.assertIn("# Failure Analysis: demo_run", report)
        self.assertIn("Generated-test parse failure", report)
        self.assertIn("Selected candidate failed but oracle candidate exists", report)

    def test_infer_summary_path_matches_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "results" / "raw"
            processed_dir = Path(tmpdir) / "results" / "processed"
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            raw_path = raw_dir / "demo_run.jsonl"
            summary_path = processed_dir / "demo_run.summary.json"
            raw_path.write_text("", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")

            inferred = infer_summary_path(raw_path)

            self.assertEqual(inferred, summary_path)

    def test_write_failure_artifacts_emits_markdown_and_json(self):
        analysis = {
            "metadata": {
                "title": "demo_run",
                "target_status": "plus",
                "model": None,
                "benchmark": None,
                "strategy": None,
            },
            "headline": {
                "total_records": 1,
                "selected_passes": 0,
                "selected_failures": 1,
                "selected_pass_rate": 0.0,
                "selected_failure_rate": 1.0,
                "oracle_passes": 0,
                "oracle_pass_rate": 0.0,
                "selection_miss_count": 0,
                "selection_miss_rate_all": 0.0,
                "selection_miss_rate_failed": 0.0,
            },
            "primary_failure_breakdown": [],
            "tag_breakdown": [],
            "records": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "failure_report.md"

            markdown_path, json_path = write_failure_artifacts(analysis, output_path)

            self.assertTrue(markdown_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("Failure Analysis: demo_run", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
