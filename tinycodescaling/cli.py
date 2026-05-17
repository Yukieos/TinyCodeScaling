"""CLI entrypoint for running experiments and rendering reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from tinycodescaling.benchmarks.base import CodeTask
from tinycodescaling.benchmarks.evalplus_loader import load_humaneval_plus
from tinycodescaling.evaluators import evaluate_samples
from tinycodescaling.execution.code_extract import build_benchmark_sample
from tinycodescaling.metrics.aggregate import (
    aggregate_seed_summaries,
    attach_evalplus_results,
    summarize_seed_records,
)
from tinycodescaling.models.tokenizer_utils import ChatPromptFormatter
from tinycodescaling.models.vllm_runner import VLLMRunner
from tinycodescaling.reports.markdown import build_markdown_report
from tinycodescaling.reports.pareto import (
    build_pareto_dataset,
    load_summary_files,
    write_pareto_artifacts,
)
from tinycodescaling.strategies import (
    BestOfNRandomPick,
    GeneratedTestSelectionStrategy,
    GreedyStrategy,
    PublicTestSelectionStrategy,
    TemperatureSamplingStrategy,
)
from tinycodescaling.strategies.base import Strategy, StrategyConfig
from tinycodescaling.utils.hashing import sha256_text
from tinycodescaling.utils.jsonl import write_jsonl
from tinycodescaling.utils.logging import configure_logging
from tinycodescaling.utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    configure_logging()
    parser = argparse.ArgumentParser(prog="tinycodescaling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment config.")
    run_parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")

    report_parser = subparsers.add_parser("report", help="Render a markdown report from a summary.")
    report_parser.add_argument("--summary", help="Path to a summary JSON file.")
    report_parser.add_argument("--run-dir", help="Directory containing summary.json.")
    report_parser.add_argument("--results", help="Path to a raw JSONL file; used to infer summary.")

    pareto_parser = subparsers.add_parser("pareto", help="Build a Pareto plot from summary files.")
    pareto_parser.add_argument(
        "--summary",
        dest="summaries",
        action="append",
        help="Path to one summary JSON file. Repeat to compare multiple runs.",
    )
    pareto_parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        help="Directory containing one or more *.summary.json files. Repeat as needed.",
    )
    pareto_parser.add_argument("--output", required=True, help="Output SVG path.")
    pareto_parser.add_argument(
        "--title",
        default="TinyCodeScaling Pareto Frontier",
        help="Chart title written into the SVG and companion JSON.",
    )
    pareto_parser.add_argument(
        "--x-metric",
        default="completion_tokens_per_problem",
        choices=[
            "completion_tokens_per_problem",
            "total_tokens_per_problem",
            "tokens_per_solved_plus",
        ],
        help="Cost metric for the x-axis.",
    )
    pareto_parser.add_argument(
        "--y-metric",
        default="pass_at_1_plus",
        choices=["pass_at_1_plus", "pass_at_1_base"],
        help="Quality metric for the y-axis.",
    )
    pareto_parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear x-axis instead of the default log scale.",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        run_experiment(Path(args.config))
        return
    if args.command == "report":
        render_report(summary_path=_resolve_summary_path(args))
        return
    if args.command == "pareto":
        render_pareto(
            summary_paths=_resolve_pareto_summary_paths(args),
            output_path=Path(args.output),
            title=args.title,
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            log_scale_x=not args.linear_x,
        )
        return


def run_experiment(config_path: Path) -> None:
    """Run one configured benchmark experiment from generation through reporting."""
    experiment_config = _load_yaml(config_path)
    model_config = _load_yaml(_resolve_config_path(config_path, experiment_config["model_config"]))
    benchmark_config = _load_yaml(
        _resolve_config_path(config_path, experiment_config["benchmark_config"])
    )
    strategy_config = StrategyConfig.from_dict(experiment_config["strategy"])
    strategy = _build_strategy(strategy_config.name)

    seeds = list(experiment_config["seeds"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_config['experiment_name']}_{timestamp}"
    results_root = _resolve_results_root(experiment_config.get("results_root", "results"))
    raw_results_path = results_root / "raw" / f"{run_id}.jsonl"
    summary_path = results_root / "processed" / f"{run_id}.summary.json"
    report_path = results_root / "reports" / f"{run_id}.md"

    tasks = load_humaneval_plus(max_tasks=benchmark_config.get("max_tasks"))
    formatter = ChatPromptFormatter(
        model_name=model_config["model_name"],
        system_prompt=model_config.get("system_prompt", "You are a helpful coding assistant."),
        use_chat_template=model_config.get("use_chat_template", True),
        revision=model_config.get("revision"),
    )

    metadata = _collect_metadata(
        experiment_name=experiment_config["experiment_name"],
        model_config=model_config,
        benchmark_config=benchmark_config,
        seeds=seeds,
        formatter=formatter,
        n_problems=len(tasks),
        strategy_config=strategy_config,
    )

    raw_records: list[dict] = []
    seed_summaries: list[dict] = []
    extraction_backend = benchmark_config.get("extraction_backend", "evalplus")
    evaluator_backend = benchmark_config.get("evaluator_backend", "evalplus")

    for seed in seeds:
        seed_everything(seed)
        runner = VLLMRunner(
            model_name=model_config["model_name"],
            dtype=model_config.get("dtype", "bfloat16"),
            max_model_len=model_config.get("max_model_len", 4096),
            gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9),
            seed=seed,
            revision=model_config.get("revision"),
        )
        seed_records, samples = _run_seed(
            seed=seed,
            tasks=tasks,
            benchmark_name=benchmark_config["benchmark"],
            extraction_backend=extraction_backend,
            formatter=formatter,
            runner=runner,
            strategy=strategy,
            strategy_config=strategy_config,
            batch_size=model_config.get("batch_size", 16),
            max_tokens=model_config["max_new_tokens"],
        )
        samples_path = results_root / "raw" / f"{run_id}_seed{seed}_samples.jsonl"
        write_jsonl(samples_path, samples)
        evaluation = evaluate_samples(
            samples_path=samples_path,
            evaluator_backend=evaluator_backend,
            dataset=benchmark_config.get("dataset", "humaneval"),
            timeout_seconds=int(benchmark_config.get("timeout_seconds", 5)),
            parallel=benchmark_config.get("parallel"),
        )
        attach_evalplus_results(seed_records, evaluation.get("eval_results"))
        raw_records.extend(seed_records)
        seed_summaries.append(
            summarize_seed_records(
                records=seed_records,
                n_problems=len(tasks),
                pass_at_1_base=evaluation["pass_at_1_base"],
                pass_at_1_plus=evaluation["pass_at_1_plus"],
                seed=seed,
                samples_path=str(samples_path),
                evaluator_cache_path=evaluation.get("cache_path"),
                evaluation=evaluation,
            )
        )

    write_jsonl(raw_results_path, raw_records)
    determinism = _compute_determinism(raw_records)
    summary = {
        "metadata": metadata,
        "benchmark": {"name": benchmark_config["benchmark"], "n_problems": len(tasks)},
        "seed_results": seed_summaries,
        "aggregate": aggregate_seed_summaries(seed_summaries),
        "determinism": determinism,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_text = build_markdown_report(summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Raw records: {raw_results_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


def render_report(summary_path: Path) -> None:
    """Load a saved summary JSON file and print its markdown report."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = build_markdown_report(summary)
    print(report_text)


def render_pareto(
    summary_paths: list[Path],
    output_path: Path,
    title: str,
    x_metric: str,
    y_metric: str,
    log_scale_x: bool,
) -> None:
    """Load summaries, build a Pareto dataset, and write plot artifacts to disk."""
    summaries = load_summary_files(summary_paths)
    points = build_pareto_dataset(summaries, x_metric=x_metric, y_metric=y_metric)
    svg_path, dataset_path = write_pareto_artifacts(
        points,
        output_path=output_path,
        title=title,
        log_scale_x=log_scale_x,
    )
    print(f"Pareto plot: {svg_path}")
    print(f"Pareto dataset: {dataset_path}")


def _run_seed(
    seed: int,
    tasks: list[CodeTask],
    benchmark_name: str,
    extraction_backend: str,
    formatter: ChatPromptFormatter,
    runner: VLLMRunner,
    strategy: Strategy,
    strategy_config: StrategyConfig,
    batch_size: int,
    max_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """Run one seed across all tasks and return raw records plus evaluator samples."""
    records: list[dict] = []
    samples: list[dict] = []

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        prompts = [formatter.format_problem_prompt(task.prompt) for task in batch]
        strategy_results = strategy.run_batch(
            tasks=batch,
            prompts=prompts,
            runner=runner,
            formatter=formatter,
            config=strategy_config,
            benchmark_name=benchmark_name,
            extraction_backend=extraction_backend,
            max_tokens=max_tokens,
        )
        for task, prompt, strategy_result in zip(batch, prompts, strategy_results):
            selected_candidate = strategy_result.candidates[strategy_result.selected_index]
            candidate_samples = [
                build_benchmark_sample(
                    task_id=task.task_id,
                    benchmark=benchmark_name,
                    extracted_code=candidate.code,
                    entry_point=task.entry_point,
                )
                for candidate in strategy_result.candidates
            ]
            selected_sample = candidate_samples[strategy_result.selected_index]
            samples.extend(candidate_samples)
            records.append(
                {
                    "seed": seed,
                    "task_id": task.task_id,
                    "entry_point": task.entry_point,
                    "prompt": prompt,
                    "raw_generation": selected_candidate.raw_text,
                    "extracted_code": strategy_result.selected_code,
                    "extraction_backend": selected_candidate.extraction_backend,
                    "extraction_method": selected_candidate.extraction_method,
                    "extraction_fallback_used": selected_candidate.extraction_fallback_used,
                    "extraction_metadata": selected_candidate.extraction_metadata,
                    "output_schema": (
                        "solution" if "solution" in selected_sample else "completion"
                    ),
                    "strategy_name": strategy_result.strategy_name,
                    "strategy_config": strategy_result.strategy_config,
                    "strategy_selected_index": strategy_result.selected_index,
                    "strategy_selection_metadata": strategy_result.selection_metadata,
                    "candidate_count": len(strategy_result.candidates),
                    "candidates": [asdict(candidate) for candidate in strategy_result.candidates],
                    "prompt_tokens": strategy_result.prompt_tokens,
                    "completion_tokens": strategy_result.total_completion_tokens,
                    "total_tokens": (
                        strategy_result.prompt_tokens + strategy_result.total_completion_tokens
                    ),
                    "selected_completion_tokens": selected_candidate.completion_tokens,
                    "latency_seconds": strategy_result.total_latency_seconds,
                    "finish_reason": selected_candidate.finish_reason,
                    "cumulative_logprob": selected_candidate.cumulative_logprob,
                    "solution_hash": sha256_text(strategy_result.selected_code),
                }
            )

    return records, samples


def _compute_determinism(records: list[dict]) -> dict:
    """Check whether extracted solutions are identical across seeds for each task."""
    grouped: dict[str, set[str]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], set()).add(record["solution_hash"])
    mismatch_task_ids = sorted(task_id for task_id, hashes in grouped.items() if len(hashes) > 1)
    return {
        "all_equal": not mismatch_task_ids,
        "mismatch_count": len(mismatch_task_ids),
        "mismatch_task_ids": mismatch_task_ids,
    }


def _collect_metadata(
    experiment_name: str,
    model_config: dict,
    benchmark_config: dict,
    seeds: list[int],
    formatter: ChatPromptFormatter,
    n_problems: int,
    strategy_config: StrategyConfig,
) -> dict:
    """Collect reproducibility metadata for the current experiment run."""
    strategy_metadata = strategy_config.as_dict()
    strategy_metadata["max_tokens"] = model_config["max_new_tokens"]
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_output(["git", "status", "--porcelain"])),
        "model": model_config["model_name"],
        "model_revision": formatter.model_revision,
        "backend": model_config.get("backend", "vllm"),
        "vllm_version": _package_version("vllm"),
        "evalplus_version": _package_version("evalplus"),
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
        "cuda_version": _cuda_version(),
        "gpu": _gpu_name(),
        "seeds": seeds,
        "strategy": strategy_config.name,
        "strategy_config": strategy_metadata,
        "prompt_template_hash": formatter.prompt_template_hash,
        "benchmark": benchmark_config["benchmark"],
        "extraction_backend": benchmark_config.get("extraction_backend", "evalplus"),
        "evaluator_backend": benchmark_config.get("evaluator_backend", "evalplus"),
        "n_problems": n_problems,
    }
    return metadata


def _build_strategy(name: str) -> Strategy:
    """Instantiate a strategy implementation from its config name."""
    if name == "greedy":
        return GreedyStrategy()
    if name in {"temperature", "temp_sample"}:
        return TemperatureSamplingStrategy()
    if name in {"best_of_n_random", "best_of_n"}:
        return BestOfNRandomPick()
    if name == "public_test_selection":
        return PublicTestSelectionStrategy()
    if name == "generated_test_selection":
        return GeneratedTestSelectionStrategy()
    raise ValueError(f"Unsupported strategy: {name}")


def _cuda_version() -> str | None:
    """Return the CUDA version reported by torch, if torch is available."""
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


def _gpu_name() -> str | None:
    """Return the primary GPU name reported by torch, when available."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _package_version(name: str) -> str | None:
    """Return an installed package version or None if the package is missing."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_output(command: list[str]) -> str:
    """Run a git command at the project root and return stripped stdout."""
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _resolve_summary_path(args: argparse.Namespace) -> Path:
    """Resolve a summary path from explicit CLI flags or inferred results paths."""
    if args.summary:
        return Path(args.summary)
    if args.run_dir:
        candidates = sorted(Path(args.run_dir).glob("*.summary.json"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No summary file found under: {args.run_dir}")
        raise FileNotFoundError(f"Multiple summary files found under: {args.run_dir}")
    if args.results:
        raw_path = Path(args.results)
        inferred = PROJECT_ROOT / "results" / "processed" / f"{raw_path.stem}.summary.json"
        if inferred.exists():
            return inferred
        raise FileNotFoundError(f"Could not infer summary file from raw results: {raw_path}")
    raise ValueError("One of --summary, --run-dir, or --results is required.")


def _resolve_pareto_summary_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve a unique ordered list of summary JSON paths for Pareto rendering."""
    summary_paths: list[Path] = []
    for summary in args.summaries or ():
        summary_paths.append(Path(summary))
    for run_dir in args.run_dirs or ():
        summary_paths.extend(sorted(Path(run_dir).glob("*.summary.json")))
    ordered_unique: list[Path] = []
    seen: set[Path] = set()
    for path in summary_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_unique.append(resolved)
    if not ordered_unique:
        raise ValueError("At least one --summary or --run-dir is required for pareto.")
    return ordered_unique


def _load_yaml(path: Path) -> dict:
    """Read and parse a YAML file into a Python dictionary."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_config_path(config_path: Path, maybe_relative_path: str) -> Path:
    """Resolve a config reference relative to the caller config or project root."""
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return candidate
    from_config_dir = (config_path.parent / candidate).resolve()
    if from_config_dir.exists():
        return from_config_dir
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_results_root(maybe_relative_path: str) -> Path:
    """Resolve the results directory against the project root when needed."""
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()
