"""Helpers for extracting and running prompt-level public tests."""

from __future__ import annotations

import ast
import doctest
from dataclasses import dataclass

from tinycodescaling.execution.sandbox import run_test_cases_in_sandbox


@dataclass(frozen=True)
class PublicTestCase:
    """One doctest-style public example extracted from a benchmark prompt."""

    source: str
    want: str


def extract_doctests_from_prompt(prompt: str) -> list[PublicTestCase]:
    """Extract doctest examples from function docstrings inside a benchmark prompt."""
    try:
        module = ast.parse(prompt)
    except SyntaxError:
        return []

    docstrings: list[str] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)

    parser = doctest.DocTestParser()
    test_cases: list[PublicTestCase] = []
    for docstring in docstrings:
        for example in parser.get_examples(docstring):
            source = example.source.strip()
            want = example.want if example.want.endswith("\n") else example.want + "\n"
            if source:
                test_cases.append(PublicTestCase(source=source, want=want))
    return test_cases


def count_passing_public_tests(
    code: str,
    public_tests: list[PublicTestCase],
    timeout_seconds: float = 5.0,
    per_test_timeout_seconds: float = 1.0,
) -> int:
    """Run public doctests against candidate code and return the number of passing cases."""
    if not public_tests:
        return 0
    result = run_test_cases_in_sandbox(
        code=code,
        test_cases=[{"source": case.source, "want": case.want} for case in public_tests],
        timeout_seconds=timeout_seconds,
        per_test_timeout_seconds=per_test_timeout_seconds,
    )
    case_results = result.details.get("case_results", [])
    return sum(bool(passed) for passed in case_results)


def run_tests(code: str, test_code: str, timeout_seconds: float = 5.0):
    """Execute arbitrary test code against candidate code in the shared sandbox."""
    from tinycodescaling.execution.sandbox import run_in_sandbox

    return run_in_sandbox(code=code, test_code=test_code, timeout_seconds=timeout_seconds)
