"""Hash helpers used for determinism checks and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest for a UTF-8 text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: dict) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable dictionary."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return sha256_text(serialized)
