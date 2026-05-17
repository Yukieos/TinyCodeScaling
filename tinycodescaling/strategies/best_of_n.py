"""Best-of-N generation baseline without any selection signal beyond sample order."""

from __future__ import annotations

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.strategies.base import Strategy, StrategyConfig, StrategyResult


class BestOfNRandomPick(Strategy):
    """Generate N candidates and deterministically keep the first as the baseline pick."""

    name = "best_of_n_random"

    def run(
        self,
        task: CodeTask,
        prompt: str,
        runner,
        formatter,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> StrategyResult:
        """Run best-of-N sampling for a single task."""
        return self.run_batch(
            tasks=[task],
            prompts=[prompt],
            runner=runner,
            formatter=formatter,
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
        formatter,
        config: StrategyConfig,
        benchmark_name: str,
        extraction_backend: str,
        max_tokens: int,
    ) -> list[StrategyResult]:
        """Generate N candidates per task and keep the first for reproducible selection."""
        candidate_batches = self._generate_candidate_batches(
            tasks=tasks,
            prompts=prompts,
            runner=runner,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            n=config.n,
            temperature=config.temperature,
            max_tokens=max_tokens,
        )
        results: list[StrategyResult] = []
        for task, candidates in zip(tasks, candidate_batches):
            results.append(
                self._build_result(
                    task_id=task.task_id,
                    candidates=candidates,
                    selected_index=0,
                    strategy_config=config,
                    selection_metadata={
                        "selection_method": "first_candidate",
                        "selection_baseline": "best_of_n_random",
                        "selected_index": 0,
                        "candidate_count": len(candidates),
                    },
                )
            )
        return results
