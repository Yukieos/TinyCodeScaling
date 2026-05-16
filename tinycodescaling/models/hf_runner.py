"""Placeholder Hugging Face backend kept for future parity with the vLLM runner."""

class HFRunner:
    """Reserve the API shape for a future non-vLLM generation backend."""
    def __init__(self, *args, **kwargs):
        """Fail fast until the Hugging Face backend is implemented."""
        raise NotImplementedError("HF backend is not implemented in Week 1; use vLLM.")
