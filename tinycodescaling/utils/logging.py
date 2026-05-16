"""Project-wide logging configuration helpers."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Install a basic timestamped logging format for CLI commands."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
