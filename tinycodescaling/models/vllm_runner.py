"""Minimal vLLM generation wrapper used by the Week 1 experiment path."""

from __future__ import annotations

import re
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
        try:
            self.llm = LLM(
                model=model_name,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                seed=seed,
                revision=revision,
            )
        except ValueError as exc:
            raise RuntimeError(_format_vllm_startup_error(exc, gpu_memory_utilization)) from exc

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


def _format_vllm_startup_error(exc: ValueError, gpu_memory_utilization: float) -> str:
    """Rewrite common vLLM startup failures into a more actionable message."""
    message = str(exc)
    if "Free memory on device" not in message or "GPU memory utilization" not in message:
        return message

    recommended_utilization = _suggest_gpu_memory_utilization(message)
    guidance = (
        "vLLM could not start because the configured `gpu_memory_utilization` is higher than "
        "the free memory currently available on the GPU. "
        f"Configured value: {gpu_memory_utilization}."
    )
    if recommended_utilization is not None:
        guidance += (
            f" Try a value at or below {recommended_utilization:.2f} and consider lowering "
            "`batch_size` as well."
        )
    else:
        guidance += " Lower `gpu_memory_utilization` and consider lowering `batch_size` as well."
    return f"{guidance}\nOriginal vLLM error: {message}"


def _suggest_gpu_memory_utilization(message: str) -> float | None:
    """Estimate a safe utilization upper bound from a vLLM free-memory error string."""
    match = re.search(r"Free memory on device .* \(([\d.]+)/([\d.]+) GiB\)", message)
    if not match:
        return None
    free_gib = float(match.group(1))
    total_gib = float(match.group(2))
    if total_gib <= 0:
        return None
    return max(0.1, min(0.95, free_gib / total_gib * 0.95))
