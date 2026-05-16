"""Placeholder LiveCodeBench evaluator backend."""

from __future__ import annotations

from pathlib import Path


def evaluate_with_livecodebench(
    samples_path: Path,
    dataset: str,
    timeout_seconds: int,
    parallel: int | None,
) -> dict:
    """Reserve the official LiveCodeBench evaluation hook for a later milestone."""
    raise NotImplementedError(
        "LiveCodeBench evaluation is intentionally delegated to the official LiveCodeBench runner. "
        "Week 1 only wires the official-backend interface; the actual integration lands when the "
        "LiveCodeBench benchmark is enabled."
    )
