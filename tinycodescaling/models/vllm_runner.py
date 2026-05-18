"""Minimal vLLM generation wrapper used by the Week 1 experiment path."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    """Normalized output for one generated sample."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_seconds: float
    cumulative_logprob: float | None


class VLLMRunner:
    """Wrap vLLM so the rest of the codebase sees a stable generation interface."""
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        seed: int = 0,
        revision: str | None = None,
    ) -> None:
        """Create a vLLM engine for one model configuration and seed."""
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import vLLM runtime: {exc}. "
                "This usually means the Python environment is missing a working "
                "dependency required by vLLM, not necessarily that the 'vllm' "
                "package is absent."
            ) from exc

        self.model_name = model_name
        self.seed = seed
        self._sampling_params_cls = _load_sampling_params_cls()
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            revision=revision,
        )

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float | None = None,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[list[GenerationResult]]:
        """Generate one or more samples per prompt and normalize token accounting."""
        sampling_seed = self.seed if seed is None else seed
        params_kwargs = dict(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            seed=sampling_seed,
        )
        if min_p is not None:
            params_kwargs["min_p"] = min_p
        params = self._sampling_params_cls(
            **params_kwargs,
        )

        started_at = time.perf_counter()
        outputs = self.llm.generate(prompts, params)
        elapsed = time.perf_counter() - started_at
        per_prompt_latency = elapsed / max(len(prompts), 1)

        results: list[list[GenerationResult]] = []
        for output in outputs:
            prompt_token_count = len(output.prompt_token_ids or [])
            samples: list[GenerationResult] = []
            for sample in output.outputs:
                samples.append(
                    GenerationResult(
                        text=sample.text,
                        prompt_tokens=prompt_token_count,
                        completion_tokens=len(sample.token_ids),
                        finish_reason=getattr(sample, "finish_reason", None),
                        latency_seconds=per_prompt_latency,
                        cumulative_logprob=getattr(sample, "cumulative_logprob", None),
                    )
                )
            results.append(samples)

        return results


def _load_sampling_params_cls():
    """Import vLLM SamplingParams lazily so import errors stay localized."""
    from vllm import SamplingParams

    return SamplingParams
