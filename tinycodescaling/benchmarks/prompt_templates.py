"""Prompt templates used to wrap raw benchmark problems before generation."""

from __future__ import annotations

WEEK1_USER_PROMPT_TEMPLATE = """Solve the following Python programming task.

Return only valid Python code. Do not include explanations.

```python
{problem_prompt}
```"""

GENERATED_TEST_PROMPT_TEMPLATE = """You are given the following function specification:

{problem_prompt}

Your task is to write {n_tests} diverse unit tests for the function `{entry_point}`.
Output format requirements:
- Return exactly one Python code block
- Inside the code block, write only top-level `assert` statements
- Write one assert per line
- Do not define any functions, classes, helpers, or variables
- Do not use `unittest`, `pytest`, or doctest `>>>` syntax
- Do not include imports
- Do not include the implementation of `{entry_point}`
- Do not assign to `{entry_point}`

Test requirements:
- Cover normal inputs, edge cases, and boundary conditions
- Keep the tests self-contained and runnable

Example format:
```python
assert {entry_point}(...) == ...
assert {entry_point}(...) == ...
```
"""


def render_user_prompt(problem_prompt: str) -> str:
    """Render one benchmark problem into the Week 1 user prompt template."""
    return WEEK1_USER_PROMPT_TEMPLATE.format(problem_prompt=problem_prompt.rstrip())


def render_generated_test_prompt(problem_prompt: str, entry_point: str, n_tests: int) -> str:
    """Render the Week 3 prompt used to ask the model for verifier-style unit tests."""
    return GENERATED_TEST_PROMPT_TEMPLATE.format(
        problem_prompt=problem_prompt.rstrip(),
        entry_point=entry_point,
        n_tests=n_tests,
    )
