# Limitations

- Official full scoring still assumes full HumanEval+ sample coverage.
- EvalPlus runtime and vLLM depend on a Python 3.10/3.11 environment; this repo does not attempt to paper over incompatible local interpreters.
- Code extraction is heuristic even with EvalPlus sanitization wrapped in front.
- Per-prompt latency is batch-apportioned wall-clock latency.
- HumanEval prompt doctests only exist for a subset of tasks, so public-test selection can fall back to first-candidate selection on many problems.
- The local sandbox follows EvalPlus-style guards, but it is still a pragmatic benchmark sandbox rather than a hardened OS-level isolation boundary.
- LiveCodeBench, MBPP+, generated-test selection, and Pareto plots are not yet wired in.
