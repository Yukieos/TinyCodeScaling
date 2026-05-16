"""Token accounting helpers for benchmark reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping


def aggregate_token_usage(records: list[Mapping[str, float | int]]) -> dict[str, float]:
    """Sum token counts and derive per-problem averages from raw task records."""
    prompt_tokens = sum(float(record.get("prompt_tokens", 0)) for record in records)
    completion_tokens = sum(float(record.get("completion_tokens", 0)) for record in records)
    total_tokens = sum(float(record.get("total_tokens", 0)) for record in records)
    n_records = max(len(records), 1)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_problem": prompt_tokens / n_records,
        "completion_tokens_per_problem": completion_tokens / n_records,
        "total_tokens_per_problem": total_tokens / n_records,
    }


def tokens_per_solved(total_tokens: float, solved_count: float) -> float:
    """Compute how many generated tokens were needed per solved problem."""
    if solved_count <= 0:
        return math.inf
    return total_tokens / solved_count


def quality_per_1k_generated_tokens(pass_rate: float, completion_tokens_per_problem: float) -> float:
    """Normalize task success by generated-token budget for cheap model comparison."""
    if completion_tokens_per_problem <= 0:
        return 0.0
    return pass_rate * 1000.0 / completion_tokens_per_problem
