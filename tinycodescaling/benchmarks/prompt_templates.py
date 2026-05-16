"""Prompt templates used to wrap raw benchmark problems before generation."""

from __future__ import annotations

WEEK1_USER_PROMPT_TEMPLATE = """Solve the following Python programming task.

Return only valid Python code. Do not include explanations.

```python
{problem_prompt}
```"""


def render_user_prompt(problem_prompt: str) -> str:
    """Render one benchmark problem into the Week 1 user prompt template."""
    return WEEK1_USER_PROMPT_TEMPLATE.format(problem_prompt=problem_prompt.rstrip())
