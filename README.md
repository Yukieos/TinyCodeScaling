# TinyCodeScaling

A reproducible benchmark for measuring how far small code LLMs can go with inference-time scaling under fixed token budgets.

## Scope

TinyCodeScaling is intentionally narrow. It is not a training project and not a general-purpose eval framework. The initial target is a single end-to-end path with a small number of strategy variants:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- HumanEval+ (all 164 tasks)
- greedy decoding
- temperature-based multi-sample generation
- token and latency accounting
- EvalPlus evaluation
- raw JSONL plus markdown summary
- 3 seeds to validate determinism

## Why EvalPlus / LiveCodeBench

- HumanEval+ and MBPP+ add substantially more tests than the original datasets, making code evaluation less fragile.
- LiveCodeBench is contamination-aware and useful for later code-generation subsets without blowing up v0.1 scope.

## Week 1 Status

This repository now contains the minimal infrastructure for:

- experiment configs
- Qwen chat-template prompt formatting
- EvalPlus HumanEval+ loading
- vLLM generation wrapper
- benchmark-official extraction/sanitization wrapper
- benchmark-official evaluator wrapper
- token/latency aggregation
- markdown report generation

The intended runtime environment is Python 3.10/3.11 with recent `transformers`, `vllm`, and the latest EvalPlus from GitHub.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

All project dependencies should live inside the repo-local `.venv`. Do not install benchmark/runtime libraries into a shared global interpreter.

## Quickstart

```bash
python -m tinycodescaling run --config configs/experiments/v01_humaneval_qwen15b.yaml
```

That command writes:

- raw records to `results/raw/`
- processed summaries to `results/processed/`
- markdown reports to `results/reports/`

To compare several finished runs on one internal Pareto chart:

```bash
python -m tinycodescaling pareto \
  --summary results/processed/run_a.summary.json \
  --summary results/processed/run_b.summary.json \
  --output results/reports/internal_pareto.svg
```

## Methods

The CLI currently supports:

- `greedy`
- `temperature`
- `best_of_n_random`
- `public_test_selection`
- `generated_test_selection`

All strategies now emit a common result schema with candidate-level metadata so later selection methods and oracle metrics can reuse the same raw artifacts.

Current selection semantics:

- `greedy`: single deterministic sample
- `temperature`: stochastic sampling, first candidate selected
- `best_of_n_random`: `n` stochastic samples, first candidate selected as the no-selection baseline
- `public_test_selection`: optional `n`-sample baseline that selects by prompt-derived public doctests when available
- `generated_test_selection`: `n` stochastic samples plus one generated verifier block, selecting by generated-test pass count

Generated-test selection now uses a stricter test-generation prompt that asks for exactly one Python code block containing only top-level `assert` statements. This keeps extraction auditable and makes canonical-pass-rate analysis less noisy.

Evaluation policy:

- HumanEval+ / MBPP+: default to EvalPlus sanitize + EvalPlus evaluator
- LiveCodeBench: reserved for official LiveCodeBench extraction + evaluator
- OpenCompass / lm-eval-harness: optional compatibility or cross-check only, not the primary correctness backend

Important note for later v0.1 comparisons:

- Best-of-N oracle is an upper bound, not a deployable method.
- HumanEval public-test selection only applies to tasks whose prompt docstring contains doctest examples.
- Public-test selection is useful as a baseline, but generated-test selection is the more general verifier-style path when authoritative prompt examples are weak or missing.

## Metrics

Primary metrics are token-based, not dollar-based:

- `pass@1`
- generated tokens per problem
- total tokens per problem
- latency per problem
- tokens per solved problem
- quality per 1K generated tokens

Dollar cost is intentionally secondary and should remain an appendix estimate.

## Limitations

- The current local machine needs a Python 3.10/3.11 runtime to install `vllm`.
- Per-prompt latency is batch-apportioned wall-clock time, not deployment latency.
- Official EvalPlus scoring is still intended for full HumanEval+ runs, not partial-task smoke checks.
- LiveCodeBench is still not wired in yet.

## Acknowledgement

- EvalPlus: HumanEval+ / MBPP+ datasets and evaluation tooling.
- Qwen team for Qwen2.5-Coder model releases and chat templates.
