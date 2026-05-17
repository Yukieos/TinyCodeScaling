"""Render experiment summaries into the project's markdown report format."""

from __future__ import annotations

from tinycodescaling.reports.leaderboard import format_mean_std


def build_markdown_report(summary: dict) -> str:
    """Convert a structured summary dictionary into a human-readable markdown table."""
    metadata = summary["metadata"]
    benchmark = summary["benchmark"]
    seed_results = summary["seed_results"]
    aggregate = summary["aggregate"]
    determinism = summary["determinism"]

    lines = [
        f"# {metadata['experiment_name']}",
        "",
        f"- benchmark: `{benchmark['name']}`",
        f"- model: `{metadata['model']}`",
        f"- backend: `{metadata['backend']}`",
        f"- extraction_backend: `{metadata['extraction_backend']}`",
        f"- evaluator_backend: `{metadata['evaluator_backend']}`",
        f"- problems: `{benchmark['n_problems']}`",
        f"- seeds: `{metadata['seeds']}`",
        f"- prompt_template_hash: `{metadata['prompt_template_hash']}`",
        "",
        "| seed | pass@1 base | pass@1 plus | prompt tok/problem | gen tok/problem | total tok/problem | latency/problem (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for seed_result in seed_results:
        lines.append(
            "| {seed} | {base:.4f} | {plus:.4f} | {prompt:.2f} | {completion:.2f} | {total:.2f} | {latency:.3f} |".format(
                seed=seed_result["seed"],
                base=seed_result["pass_at_1_base"],
                plus=seed_result["pass_at_1_plus"],
                prompt=seed_result["prompt_tokens_per_problem"],
                completion=seed_result["completion_tokens_per_problem"],
                total=seed_result["total_tokens_per_problem"],
                latency=seed_result["latency_seconds_per_problem"],
            )
        )

    lines.extend(
        [
            "| mean ± std | {base} | {plus} | {prompt} | {completion} | {total} | {latency} |".format(
                base=format_mean_std(
                    aggregate["pass_at_1_base"]["mean"],
                    aggregate["pass_at_1_base"]["std"],
                ),
                plus=format_mean_std(
                    aggregate["pass_at_1_plus"]["mean"],
                    aggregate["pass_at_1_plus"]["std"],
                ),
                prompt=format_mean_std(
                    aggregate["prompt_tokens_per_problem"]["mean"],
                    aggregate["prompt_tokens_per_problem"]["std"],
                    precision=2,
                ),
                completion=format_mean_std(
                    aggregate["completion_tokens_per_problem"]["mean"],
                    aggregate["completion_tokens_per_problem"]["std"],
                    precision=2,
                ),
                total=format_mean_std(
                    aggregate["total_tokens_per_problem"]["mean"],
                    aggregate["total_tokens_per_problem"]["std"],
                    precision=2,
                ),
                latency=format_mean_std(
                    aggregate["latency_seconds_per_problem"]["mean"],
                    aggregate["latency_seconds_per_problem"]["std"],
                    precision=3,
                ),
            ),
            "",
            "## Determinism",
            "",
            f"- identical solution hashes across seeds: `{determinism['all_equal']}`",
            f"- mismatch count: `{determinism['mismatch_count']}`",
        ]
    )

    if determinism["mismatch_task_ids"]:
        lines.append(f"- mismatch task ids: `{determinism['mismatch_task_ids']}`")

    oracle_plus = aggregate.get("oracle_pass_at_n_plus")
    oracle_base = aggregate.get("oracle_pass_at_n_base")
    if oracle_base and oracle_plus:
        oracle_k = metadata.get("strategy_config", {}).get("n")
        if oracle_k and int(oracle_k) > 1:
            lines.extend(
                [
                    "",
                    "## Oracle",
                    "",
                    f"- oracle pass@{int(oracle_k)} base: `{format_mean_std(oracle_base['mean'], oracle_base['std'])}`",
                    f"- oracle pass@{int(oracle_k)} plus: `{format_mean_std(oracle_plus['mean'], oracle_plus['std'])}`",
                ]
            )

    if "public_test_discrimination_rate" in aggregate:
        lines.extend(
            [
                "",
                "## Public Tests",
                "",
                "- discrimination rate: `{}`".format(
                    format_mean_std(
                        aggregate["public_test_discrimination_rate"]["mean"],
                        aggregate["public_test_discrimination_rate"]["std"],
                    )
                ),
            ]
        )
        if "public_test_fallback_rate" in aggregate:
            lines.append(
                "- fallback rate: `{}`".format(
                    format_mean_std(
                        aggregate["public_test_fallback_rate"]["mean"],
                        aggregate["public_test_fallback_rate"]["std"],
                    )
                )
            )

    if "generated_test_valid_tests_per_task" in aggregate:
        lines.extend(
            [
                "",
                "## Generated Tests",
                "",
                "- valid tests per task: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_valid_tests_per_task"]["mean"],
                        aggregate["generated_test_valid_tests_per_task"]["std"],
                    )
                ),
                "- discrimination rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_discrimination_rate"]["mean"],
                        aggregate["generated_test_discrimination_rate"]["std"],
                    )
                ),
                "- fallback rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_fallback_rate"]["mean"],
                        aggregate["generated_test_fallback_rate"]["std"],
                    )
                ),
                "- parse failure rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_parse_failure_rate"]["mean"],
                        aggregate["generated_test_parse_failure_rate"]["std"],
                    )
                ),
                "- entry-point leak rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_entry_point_leak_rate"]["mean"],
                        aggregate["generated_test_entry_point_leak_rate"]["std"],
                    )
                ),
                "- tie rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_tie_rate"]["mean"],
                        aggregate["generated_test_tie_rate"]["std"],
                    )
                ),
            ]
        )
        if "generated_test_canonical_pass_rate" in aggregate:
            lines.append(
                "- canonical pass rate: `{}`".format(
                    format_mean_std(
                        aggregate["generated_test_canonical_pass_rate"]["mean"],
                        aggregate["generated_test_canonical_pass_rate"]["std"],
                    )
                )
            )

    return "\n".join(lines) + "\n"
