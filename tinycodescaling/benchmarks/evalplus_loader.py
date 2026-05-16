"""Load HumanEval+ tasks through EvalPlus and normalize them into CodeTask objects."""

from __future__ import annotations

from typing import Any

from tinycodescaling.benchmarks.base import CodeTask


def load_humaneval_plus(max_tasks: int | None = None) -> list[CodeTask]:
    """Fetch HumanEval+ from EvalPlus and optionally truncate it for smoke runs."""
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required to load HumanEval+. Install runtime dependencies first."
        ) from exc

    raw_tasks: dict[str, dict[str, Any]] = get_human_eval_plus()
    ordered_ids = sorted(raw_tasks, key=_task_sort_key)
    if max_tasks is not None:
        ordered_ids = ordered_ids[:max_tasks]

    tasks: list[CodeTask] = []
    for task_id in ordered_ids:
        problem = raw_tasks[task_id]
        tasks.append(
            CodeTask(
                task_id=task_id,
                prompt=problem["prompt"],
                entry_point=problem.get("entry_point"),
                canonical_solution=problem.get("canonical_solution"),
                public_tests=tuple(problem.get("example_test", []) or ()),
                metadata={
                    "contract": problem.get("contract"),
                    "base_input": problem.get("base_input"),
                    "plus_input": problem.get("plus_input"),
                    "atol": problem.get("atol"),
                },
            )
        )

    return tasks


def _task_sort_key(task_id: str) -> tuple[str, int]:
    """Sort task ids numerically within each benchmark prefix when possible."""
    prefix, _, suffix = task_id.partition("/")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, 10**9
