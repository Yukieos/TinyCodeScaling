"""Greedy decoding strategy used by the Week 1 benchmark path."""

from __future__ import annotations

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.strategies.base import Strategy, StrategyConfig, StrategyResult


class GreedyStrategy(Strategy):
    """Force deterministic single-sample decoding."""

    name = "greedy"

    def run(
        self,
        task: CodeTask,
        prompt: str,
        runner,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> StrategyResult:
        """Run greedy decoding for a single task."""
        return self.run_batch(
            tasks=[task],
            prompts=[prompt],
            runner=runner,
            config=config,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            max_tokens=max_tokens,
        )[0]

    def run_batch(
        self,
        tasks: list[CodeTask],
        prompts: list[str],
        runner,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> list[StrategyResult]:
        """Run greedy decoding for a batch of prompts in one vLLM call."""
        outputs = runner.generate(
            prompts,
            n=1,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        results: list[StrategyResult] = []
        for task, samples in zip(tasks, outputs):
            candidates = self._normalize_candidates(
                task=task,
                generations=samples,
                benchmark_name=benchmark_name,
                extraction_backend=extraction_backend,
            )
            results.append(
                self._build_result(
                    task_id=task.task_id,
                    candidates=candidates,
                    selected_index=0,
                    strategy_config=config,
                    selection_metadata={
                        "selection_method": "greedy",
                        "selected_index": 0,
                    },
                )
            )
        return results
