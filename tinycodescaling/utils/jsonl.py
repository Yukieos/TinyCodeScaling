"""Read and write newline-delimited JSON files used for benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write a list of dictionaries to a JSONL file, one record per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into memory while skipping blank lines."""
    source = Path(path)
    records: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
