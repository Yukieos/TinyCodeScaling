"""Signal-based timeout helper for short local execution windows."""

import signal
from contextlib import contextmanager


@contextmanager
def time_limit(seconds: int):
    """Raise TimeoutError if the wrapped block exceeds the given wall-clock limit."""
    def _handle_timeout(signum, frame):
        """Convert SIGALRM into a Python TimeoutError."""
        raise TimeoutError(f"Timed out after {seconds} seconds.")

    original = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)
