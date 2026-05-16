# Methodology

## Week 1

Week 1 is intentionally narrow:

- benchmark: HumanEval+
- model: Qwen2.5-Coder-1.5B-Instruct
- baseline strategy: greedy
- seeds: 11 / 12 / 13

The codebase now also exposes a unified strategy interface plus a temperature-sampling strategy. That interface records all generated candidates, total completion-token usage, and selection metadata so later Week 2 strategies can share the same output schema.

The current Week 2 strategy set is:

- greedy
- temperature single-selection baseline
- best-of-N random-pick baseline
- public-test selection

Prompt formatting uses the model's Hugging Face chat template. This matters for Qwen2.5-Coder Instruct models and is expected to materially affect pass rates.

## Evaluator Policy

Correctness is delegated to benchmark-official tooling whenever possible.

- HumanEval+ / MBPP+: EvalPlus sanitize plus EvalPlus evaluator
- LiveCodeBench: official LiveCodeBench extraction plus evaluator

TinyCodeScaling may wrap those tools to record metadata and normalize interfaces, but it should not silently replace them with a custom correctness implementation.

For multi-candidate strategies, TinyCodeScaling writes every candidate to the EvalPlus samples file, runs the official evaluator once, and then derives:

- selected-solution pass@1 from the configured `selected_index`
- oracle pass@N from whether any candidate passed the official tests

## Token Accounting

Each generation record stores:

- prompt tokens
- total completion tokens across all generated candidates for that task
- total tokens
- per-prompt apportioned latency
- strategy metadata and candidate-level extraction outputs

For batched vLLM inference, `latency_seconds` is total batch wall-clock divided by batch size. This is a reproducibility metric, not a deployment SLA metric.
