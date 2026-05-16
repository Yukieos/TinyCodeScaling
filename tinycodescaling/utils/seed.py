"""Helpers for seeding Python, NumPy, and torch random number generators."""

from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Best-effort seed all supported random generators used by the project."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
