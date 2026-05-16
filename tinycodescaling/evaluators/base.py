"""Dispatch benchmark samples to the selected official evaluation backend."""

from __future__ import annotations

from pathlib import Path

from tinycodescaling.evaluators.evalplus_backend import evaluate_with_evalplus
from tinycodescaling.evaluators.livecodebench_backend import evaluate_with_livecodebench


def evaluate_samples(
    samples_path: Path,
    evaluator_backend: str,
    dataset: str,
    timeout_seconds: int,
    parallel: int | None,
) -> dict:
    """Route one samples file to the configured evaluator backend."""
    if evaluator_backend == "evalplus":
        return evaluate_with_evalplus(
            samples_path=samples_path,
            dataset=dataset,
            timeout_seconds=timeout_seconds,
            parallel=parallel,
        )
    if evaluator_backend == "livecodebench":
        return evaluate_with_livecodebench(
            samples_path=samples_path,
            dataset=dataset,
            timeout_seconds=timeout_seconds,
            parallel=parallel,
        )
    raise ValueError(f"Unsupported evaluator backend: {evaluator_backend}")
