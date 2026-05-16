"""Common benchmark task definitions shared across loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeTask:
    """Normalized benchmark task payload consumed by the rest of the pipeline."""
    task_id: str
    prompt: str
    entry_point: str | None = None
    canonical_solution: str | None = None
    public_tests: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
