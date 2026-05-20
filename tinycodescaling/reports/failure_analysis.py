"""Summarize raw experiment records into actionable failure buckets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PRIMARY_FAILURE_LABELS = {
    "missing_official_results": "Missing official evaluator results",
    "extraction_fallback": "Extraction fallback used",
    "generated_test_parse_failure": "Generated-test parse failure",
    "generated_test_entry_point_leak": "Generated-test entry-point leak",
    "generated_test_selection_fallback": "Generated-test selection fallback",
    "public_test_no_public_tests": "No public tests available",
    "generation_truncated": "Generation truncated at max tokens",
    "selection_miss": "Selection missed a passing candidate",
    "no_candidate_passed_official_tests": "No candidate passed official tests",
}

TAG_LABELS = {
    "selection_miss": "Selected candidate failed but oracle candidate exists",
    "no_candidate_passed_official_tests": "No sampled candidate passed official tests",
    "extraction_fallback": "Extraction fallback used",
    "generation_truncated": "Selected generation hit max tokens",
    "generated_test_parse_failure": "Generated tests failed to parse",
    "generated_test_entry_point_leak": "Generated tests leaked the entry point",
    "generated_test_selection_fallback": "Generated-test selection fell back to first candidate",
    "generated_test_no_discrimination": "Generated tests gave every candidate the same score",
    "public_test_no_public_tests": "Task had no public doctests",
    "public_test_no_discrimination": "Public tests gave every candidate the same score",
    "missing_official_results": "Official evaluator results missing from raw records",
}

PRIMARY_FAILURE_ORDER = [
    "missing_official_results",
    "extraction_fallback",
    "generated_test_parse_failure",
    "generated_test_entry_point_leak",
    "generated_test_selection_fallback",
    "public_test_no_public_tests",
    "generation_truncated",
    "selection_miss",
    "no_candidate_passed_official_tests",
]

TAG_ORDER = [
    "selection_miss",
    "no_candidate_passed_official_tests",
    "extraction_fallback",
    "generation_truncated",
    "generated_test_parse_failure",
    "generated_test_entry_point_leak",
    "generated_test_selection_fallback",
    "generated_test_no_discrimination",
    "public_test_no_public_tests",
    "public_test_no_discrimination",
    "missing_official_results",
]


def analyze_failures(
    records: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    target_status: str = "plus",
    example_limit: int = 8,
) -> dict[str, Any]:
    """Derive failure buckets and example slices from one raw-results JSONL payload."""
    if target_status not in {"base", "plus"}:
        raise ValueError("target_status must be either 'base' or 'plus'.")

    cases = [_build_failure_case(record, target_status=target_status) for record in records]
    total_records = len(cases)
    solved_cases = [case for case in cases if case["selected_pass"]]
    failed_cases = [case for case in cases if not case["selected_pass"]]

    primary_counter = Counter(case["primary_failure_type"] for case in failed_cases)
    tag_counter = Counter(tag for case in cases for tag in case["tags"])
    examples_by_primary = _collect_examples(
        failed_cases,
        key_name="primary_failure_type",
        keys=PRIMARY_FAILURE_ORDER,
        limit=example_limit,
    )
    examples_by_tag = _collect_examples(
        cases,
        key_name="tags",
        keys=TAG_ORDER,
        limit=example_limit,
    )

    analysis = {
        "metadata": _analysis_metadata(summary, target_status=target_status),
        "headline": {
            "total_records": total_records,
            "selected_passes": len(solved_cases),
            "selected_failures": len(failed_cases),
            "selected_pass_rate": _safe_rate(len(solved_cases), total_records),
            "selected_failure_rate": _safe_rate(len(failed_cases), total_records),
            "oracle_passes": sum(bool(case["oracle_pass"]) for case in cases),
            "oracle_pass_rate": _safe_rate(
                sum(bool(case["oracle_pass"]) for case in cases),
                total_records,
            ),
            "selection_miss_count": tag_counter["selection_miss"],
            "selection_miss_rate_all": _safe_rate(
                tag_counter["selection_miss"],
                total_records,
            ),
            "selection_miss_rate_failed": _safe_rate(
                tag_counter["selection_miss"],
                len(failed_cases),
            ),
        },
        "primary_failure_breakdown": [
            {
                "key": key,
                "label": PRIMARY_FAILURE_LABELS[key],
                "count": primary_counter[key],
                "rate_all": _safe_rate(primary_counter[key], total_records),
                "rate_failed": _safe_rate(primary_counter[key], len(failed_cases)),
                "examples": examples_by_primary.get(key, []),
            }
            for key in PRIMARY_FAILURE_ORDER
            if primary_counter[key] > 0
        ],
        "tag_breakdown": [
            {
                "key": key,
                "label": TAG_LABELS[key],
                "count": tag_counter[key],
                "rate_all": _safe_rate(tag_counter[key], total_records),
                "rate_failed": _safe_rate(tag_counter[key], len(failed_cases)),
                "examples": examples_by_tag.get(key, []),
            }
            for key in TAG_ORDER
            if tag_counter[key] > 0
        ],
        "records": cases,
    }
    return analysis


def build_failure_report_markdown(analysis: dict[str, Any]) -> str:
    """Render failure analysis output into a compact markdown report."""
    metadata = analysis["metadata"]
    headline = analysis["headline"]

    lines = [
        f"# Failure Analysis: {metadata['title']}",
        "",
        f"- target_status: `{metadata['target_status']}`",
    ]
    if metadata.get("model"):
        lines.append(f"- model: `{metadata['model']}`")
    if metadata.get("benchmark"):
        lines.append(f"- benchmark: `{metadata['benchmark']}`")
    if metadata.get("strategy"):
        lines.append(f"- strategy: `{metadata['strategy']}`")

    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- records: `{headline['total_records']}`",
            f"- selected pass rate: `{headline['selected_pass_rate']:.4f}`",
            f"- selected failure rate: `{headline['selected_failure_rate']:.4f}`",
            f"- oracle pass rate: `{headline['oracle_pass_rate']:.4f}`",
            f"- selection miss rate (all records): `{headline['selection_miss_rate_all']:.4f}`",
            f"- selection miss rate (failed records): `{headline['selection_miss_rate_failed']:.4f}`",
        ]
    )

    if analysis["primary_failure_breakdown"]:
        lines.extend(
            [
                "",
                "## Primary Failure Breakdown",
                "",
                "| bucket | count | rate/all | rate/failed |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for bucket in analysis["primary_failure_breakdown"]:
            lines.append(
                "| {label} | {count} | {rate_all:.4f} | {rate_failed:.4f} |".format(
                    label=bucket["label"],
                    count=bucket["count"],
                    rate_all=bucket["rate_all"],
                    rate_failed=bucket["rate_failed"],
                )
            )

    if analysis["tag_breakdown"]:
        lines.extend(
            [
                "",
                "## Important Slices",
                "",
            ]
        )
        for bucket in analysis["tag_breakdown"]:
            lines.append(
                "- {label}: `{count}` (`{rate_all:.4f}` of all records)".format(
                    label=bucket["label"],
                    count=bucket["count"],
                    rate_all=bucket["rate_all"],
                )
            )

    example_sections = [
        bucket
        for bucket in analysis["primary_failure_breakdown"] + analysis["tag_breakdown"]
        if bucket["examples"]
    ]
    if example_sections:
        lines.extend(["", "## Example Records", ""])
        for bucket in example_sections:
            lines.append(f"### {bucket['label']}")
            lines.append("")
            for example in bucket["examples"]:
                lines.append(f"- {format_failure_example(example)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_failure_artifacts(
    analysis: dict[str, Any],
    output_path: Path,
) -> tuple[Path, Path]:
    """Write markdown and JSON failure-analysis artifacts next to each other."""
    markdown_path = output_path
    json_path = output_path.with_suffix(".json")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_failure_report_markdown(analysis), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return markdown_path, json_path


def format_failure_example(example: dict[str, Any]) -> str:
    """Format one failure example into a short markdown bullet body."""
    parts = [example["record_id"]]
    if example.get("selected_status") is not None:
        parts.append(f"selected={example['selected_status']}")
    parts.append(f"oracle={example['oracle_pass']}")
    if example.get("finish_reason"):
        parts.append(f"finish={example['finish_reason']}")
    if example.get("fallback_reason"):
        parts.append(f"fallback={example['fallback_reason']}")
    if example.get("parse_method"):
        parts.append(f"parse={example['parse_method']}")
    if example.get("canonical_pass_rate") is not None:
        parts.append(f"canonical={example['canonical_pass_rate']:.2f}")
    if example.get("pass_counts") is not None:
        parts.append(f"pass_counts={example['pass_counts']}")
    return ", ".join(parts)


def infer_summary_path(raw_results_path: Path) -> Path | None:
    """Infer the matching processed summary path for one raw-results JSONL path."""
    if raw_results_path.suffix != ".jsonl":
        return None
    raw_dir = raw_results_path.parent
    if raw_dir.name != "raw":
        return None
    processed_dir = raw_dir.parent / "processed"
    candidate = processed_dir / f"{raw_results_path.stem}.summary.json"
    return candidate if candidate.exists() else None


def _analysis_metadata(
    summary: dict[str, Any] | None,
    target_status: str,
) -> dict[str, Any]:
    """Extract a small metadata block for failure reports."""
    metadata = (summary or {}).get("metadata", {})
    benchmark = (summary or {}).get("benchmark", {})
    title = metadata.get("experiment_name") or "TinyCodeScaling run"
    return {
        "title": title,
        "target_status": target_status,
        "model": metadata.get("model"),
        "benchmark": benchmark.get("name"),
        "strategy": metadata.get("strategy"),
    }


def _build_failure_case(record: dict[str, Any], target_status: str) -> dict[str, Any]:
    """Normalize one raw task record into a case used by failure summaries."""
    selected_status = record.get(f"selected_{target_status}_status")
    oracle_pass = bool(record.get(f"oracle_{target_status}_pass"))
    metadata = record.get("strategy_selection_metadata", {}) or {}
    pass_counts = metadata.get("pass_counts")
    generated_test_pass_counts = metadata.get("pass_counts")

    tags: list[str] = []
    if selected_status is None:
        tags.append("missing_official_results")
    elif selected_status != "pass":
        if oracle_pass:
            tags.append("selection_miss")
        else:
            tags.append("no_candidate_passed_official_tests")

    if record.get("extraction_fallback_used"):
        tags.append("extraction_fallback")
    if record.get("finish_reason") == "length":
        tags.append("generation_truncated")

    if metadata.get("generated_test_parse_method") == "parse_error":
        tags.append("generated_test_parse_failure")
    if metadata.get("generated_test_entry_point_leak_detected"):
        tags.append("generated_test_entry_point_leak")
    if metadata.get("generated_test_fallback_used"):
        tags.append("generated_test_selection_fallback")
    if (
        metadata.get("selection_method") == "generated_test_pass_count"
        and generated_test_pass_counts is not None
        and len(set(generated_test_pass_counts)) <= 1
    ):
        tags.append("generated_test_no_discrimination")

    if metadata.get("fallback_reason") == "no_public_tests":
        tags.append("public_test_no_public_tests")
    if (
        metadata.get("selection_method") == "public_test_pass_count"
        and pass_counts is not None
        and len(set(pass_counts)) <= 1
    ):
        tags.append("public_test_no_discrimination")

    primary_failure_type = "solved"
    if selected_status != "pass":
        primary_failure_type = _primary_failure_type(tags)

    seed = record.get("seed")
    record_id = f"seed{seed}:{record['task_id']}" if seed is not None else str(record["task_id"])
    return {
        "record_id": record_id,
        "task_id": record["task_id"],
        "seed": seed,
        "strategy_name": record.get("strategy_name"),
        "selected_status": selected_status,
        "selected_pass": selected_status == "pass",
        "oracle_pass": oracle_pass,
        "primary_failure_type": primary_failure_type,
        "tags": tags,
        "finish_reason": record.get("finish_reason"),
        "candidate_count": record.get("candidate_count"),
        "selected_index": record.get("strategy_selected_index"),
        "fallback_reason": metadata.get("generated_test_fallback_reason")
        or metadata.get("fallback_reason"),
        "parse_method": metadata.get("generated_test_parse_method"),
        "canonical_pass_rate": metadata.get("generated_test_canonical_pass_rate"),
        "pass_counts": metadata.get("pass_counts"),
    }


def _primary_failure_type(tags: list[str]) -> str:
    """Pick one primary bucket for a failed record using a fixed precedence order."""
    for key in PRIMARY_FAILURE_ORDER:
        if key in tags:
            return key
    return "no_candidate_passed_official_tests"


def _collect_examples(
    cases: list[dict[str, Any]],
    key_name: str,
    keys: list[str],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Collect a small number of representative examples for every bucket."""
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        case_keys = case[key_name] if key_name == "tags" else [case[key_name]]
        for key in case_keys:
            if key not in keys or len(examples[key]) >= limit:
                continue
            examples[key].append(
                {
                    "record_id": case["record_id"],
                    "task_id": case["task_id"],
                    "seed": case["seed"],
                    "selected_status": case["selected_status"],
                    "oracle_pass": case["oracle_pass"],
                    "finish_reason": case["finish_reason"],
                    "fallback_reason": case["fallback_reason"],
                    "parse_method": case["parse_method"],
                    "canonical_pass_rate": case["canonical_pass_rate"],
                    "pass_counts": case["pass_counts"],
                }
            )
    return dict(examples)


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a rate while handling empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
