"""Small formatting helpers shared by reporting code."""

from __future__ import annotations


def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format a mean and standard deviation pair for report tables."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"
