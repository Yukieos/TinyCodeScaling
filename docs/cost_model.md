# Cost Model

Primary benchmarking metrics should be token-based, not dollar-based.

Why:

- token budgets transfer across providers and deployment setups
- dollar pricing changes over time
- local inference does not map cleanly to hosted pricing

Week 1 therefore reports token and latency metrics first. Dollar estimates can be added later as an appendix:

`estimated_cost = total_tokens * provider_price_per_token`

