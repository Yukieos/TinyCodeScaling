"""pass@k utility used when sampling-based strategies are added later."""

from __future__ import annotations


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimate pass@k from sampled outcomes using the standard unbiased formula."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if num_correct < 0 or num_correct > num_samples:
        raise ValueError("num_correct must be between 0 and num_samples.")
    if k <= 0:
        raise ValueError("k must be positive.")

    if num_samples - num_correct < k:
        return 1.0

    product = 1.0
    for i in range(k):
        product *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - product
