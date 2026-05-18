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
- public-test selection baseline
- generated-test selection

Prompt formatting uses the model's Hugging Face chat template. This matters for Qwen2.5-Coder Instruct models and is expected to materially affect pass rates.

Sampling truncation is treated as a decoding configuration rather than a separate strategy family. The current code can pass through `top_p`, `top_k`, and `min_p`, which keeps comparisons like best-of-N top-p versus best-of-N Min-P inside one shared result schema.

## Evaluator Policy

Correctness is delegated to benchmark-official tooling whenever possible.

- HumanEval+ / MBPP+: EvalPlus sanitize plus EvalPlus evaluator
- LiveCodeBench: official LiveCodeBench extraction plus evaluator

TinyCodeScaling may wrap those tools to record metadata and normalize interfaces, but it should not silently replace them with a custom correctness implementation.

For multi-candidate strategies, TinyCodeScaling writes every candidate to the EvalPlus samples file, runs the official evaluator once, and then derives:

- selected-solution pass@1 from the configured `selected_index`
- oracle pass@N from whether any candidate passed the official tests

The reporting layer now also supports an internal Pareto plot builder. It reads one or more saved summary files, projects each run into a cost-quality point, adds oracle companion points when those metrics exist, and renders a lightweight SVG for quick comparison.

## Token Accounting

Each generation record stores:

- prompt tokens
- total completion tokens across all generated candidates for that task
- total tokens
- per-prompt apportioned latency
- strategy metadata and candidate-level extraction outputs

For batched vLLM inference, `latency_seconds` is total batch wall-clock divided by batch size. This is a reproducibility metric, not a deployment SLA metric.

## Week 3

Generated-test selection now follows this path:

- generate `n_solutions` candidate solutions from the same solution prompt used by other strategies
- generate one verifier block with a stricter prompt that asks for exactly one Python code block containing only top-level `assert` statements
- extract valid assertions, rejecting outputs that redefine the benchmark entry point
- run every candidate against the extracted assertions in the sandbox
- select by generated-test pass count, breaking ties by cumulative logprob

The run metadata also records:

- full generated-test raw output
- extracted assertions
- pass matrix and pass counts
- fallback reason when no valid verifier tests survive extraction
- canonical pass rate for the generated tests when the benchmark exposes a canonical solution
