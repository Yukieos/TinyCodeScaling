"""Sampling strategy interfaces shared by decoding implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.execution.code_extract import extract_code
from tinycodescaling.models.vllm_runner import GenerationResult


@dataclass(frozen=True)
class CandidateSolution:
    """One generated candidate plus the metadata needed for later analysis."""

    code: str
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    cumulative_logprob: float | None
    finish_reason: str | None
    latency_seconds: float
    extraction_backend: str
    extraction_method: str
    extraction_fallback_used: bool
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyResult:
    """Normalized output of running one strategy on one benchmark task."""

    task_id: str
    selected_code: str
    selected_index: int
    candidates: list[CandidateSolution]
    prompt_tokens: int
    total_completion_tokens: int
    total_latency_seconds: float
    strategy_name: str
    strategy_config: dict[str, Any]
    selection_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyConfig:
    """User-supplied decoding parameters parsed from experiment configs."""

    name: str
    n: int = 1
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyConfig":
        """Parse known strategy keys and preserve unknown ones for future strategies."""
        known_keys = {"name", "n", "temperature"}
        data = dict(payload)
        sample_count = data.get("n", data.get("n_solutions", 1))
        temperature = data.get("temperature", data.get("temperature_solutions", 0.0))
        return cls(
            name=str(data["name"]),
            n=int(sample_count),
            temperature=float(temperature),
            extra={key: value for key, value in data.items() if key not in known_keys},
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation suitable for metadata and JSON."""
        data = {
            "name": self.name,
            "n": self.n,
            "temperature": self.temperature,
        }
        data.update(self.extra)
        return data


class Strategy(ABC):
    """Base interface for generation strategies that return structured candidates."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        task: CodeTask,
        prompt: str,
        runner: Any,
        formatter: Any,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> StrategyResult:
        """Run the strategy on one task and return the selected solution plus candidates."""

    def run_batch(
        self,
        tasks: Sequence[CodeTask],
        prompts: Sequence[str],
        runner: Any,
        formatter: Any,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> list[StrategyResult]:
        """Default batch implementation that simply loops over single-task execution."""
        return [
            self.run(
                task=task,
                prompt=prompt,
                runner=runner,
                formatter=formatter,
                config=config,
                benchmark_name=benchmark_name,
                extraction_backend=extraction_backend,
                max_tokens=max_tokens,
            )
            for task, prompt in zip(tasks, prompts)
        ]

    def _normalize_candidates(
        self,
        task: CodeTask,
        generations: Sequence[GenerationResult],
        benchmark_name: str,
        extraction_backend: str,
    ) -> list[CandidateSolution]:
        """Extract benchmark-ready code for every generated sample."""
        candidates: list[CandidateSolution] = []
        for generation in generations:
            extraction = extract_code(
                raw_output=generation.text,
                benchmark=benchmark_name,
                entry_point=task.entry_point,
                task_id=task.task_id,
                extraction_backend=extraction_backend,
            )
            candidates.append(
                CandidateSolution(
                    code=extraction["code"],
                    raw_text=generation.text,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    cumulative_logprob=generation.cumulative_logprob,
                    finish_reason=generation.finish_reason,
                    latency_seconds=generation.latency_seconds,
                    extraction_backend=extraction["backend"],
                    extraction_method=extraction["method"],
                    extraction_fallback_used=bool(extraction["fallback_used"]),
                    extraction_metadata=extraction,
                )
            )
        return candidates

    def _generate_candidate_batches(
        self,
        tasks: Sequence[CodeTask],
        prompts: Sequence[str],
        runner: Any,
        benchmark_name: str,
        extraction_backend: str,
        n: int,
        temperature: float,
        max_tokens: int,
        sampling_kwargs: Mapping[str, Any] | None = None,
    ) -> list[list[CandidateSolution]]:
        """Generate and normalize candidate batches for one strategy invocation."""
        outputs = runner.generate(
            list(prompts),
            n=max(n, 1),
            temperature=temperature,
            max_tokens=max_tokens,
            **dict(sampling_kwargs or {}),
        )
        return [
            self._normalize_candidates(
                task=task,
                generations=samples,
                benchmark_name=benchmark_name,
                extraction_backend=extraction_backend,
            )
            for task, samples in zip(tasks, outputs)
        ]

    def _sampling_kwargs_from_config(
        self,
        config: StrategyConfig,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Extract shared vLLM sampling options from a strategy config."""
        sampling_kwargs: dict[str, Any] = {}
        for key in ("top_p", "top_k", "min_p"):
            scoped_key = f"{key}_{scope}" if scope else key
            value = config.extra.get(scoped_key, config.extra.get(key))
            if value is None:
                continue
            sampling_kwargs[key] = value
        return sampling_kwargs

    def _build_result(
        self,
        task_id: str,
        candidates: Sequence[CandidateSolution],
        selected_index: int,
        strategy_config: StrategyConfig,
        selection_metadata: dict[str, Any] | None = None,
        total_latency_seconds: float | None = None,
        prompt_tokens: int | None = None,
        total_completion_tokens: int | None = None,
    ) -> StrategyResult:
        """Assemble a StrategyResult with consistent token and selection accounting."""
        if not candidates:
            raise ValueError("Strategies must return at least one candidate solution.")

        selected_candidate = candidates[selected_index]
        return StrategyResult(
            task_id=task_id,
            selected_code=selected_candidate.code,
            selected_index=selected_index,
            candidates=list(candidates),
            prompt_tokens=candidates[0].prompt_tokens if prompt_tokens is None else prompt_tokens,
            total_completion_tokens=(
                sum(candidate.completion_tokens for candidate in candidates)
                if total_completion_tokens is None
                else total_completion_tokens
            ),
            total_latency_seconds=(
                selected_candidate.latency_seconds
                if total_latency_seconds is None
                else total_latency_seconds
            ),
            strategy_name=self.name,
            strategy_config=strategy_config.as_dict(),
            selection_metadata=selection_metadata or {},
        )
