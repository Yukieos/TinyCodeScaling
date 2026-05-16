"""Latency helpers for summary metrics."""

from __future__ import annotations


def average_latency_seconds(records: list[dict]) -> float:
    """Average the stored per-problem latency across a batch of raw records."""
    if not records:
        return 0.0
    return sum(float(record.get("latency_seconds", 0.0)) for record in records) / len(records)
