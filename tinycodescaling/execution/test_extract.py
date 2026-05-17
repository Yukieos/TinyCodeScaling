"""Extract runnable generated unit tests from raw model output."""

from __future__ import annotations

import ast
import doctest
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedTestExtractionResult:
    """Structured output of parsing generated tests for later audit and selection."""

    assertions: list[str]
    extracted_code: str
    parse_method: str
    used_markdown_code_block: bool
    parse_error: str | None = None
    entry_point_leak_detected: bool = False
    total_assertions_found: int = 0
    kept_assertions: int = 0
    rejected_assertions: int = 0


def extract_test_assertions(raw_output: str, entry_point: str) -> GeneratedTestExtractionResult:
    """Extract assert statements that reference the task entry point."""
    extracted_code, used_markdown_code_block = extract_markdown_code_block(raw_output)
    if _contains_entry_point_definition(extracted_code, entry_point):
        return GeneratedTestExtractionResult(
            assertions=[],
            extracted_code=extracted_code,
            parse_method="rejected_leak",
            used_markdown_code_block=used_markdown_code_block,
            entry_point_leak_detected=True,
        )

    try:
        tree = ast.parse(extracted_code)
    except SyntaxError as exc:
        doctest_assertions = _extract_doctest_assertions(extracted_code, entry_point)
        if doctest_assertions:
            return GeneratedTestExtractionResult(
                assertions=doctest_assertions,
                extracted_code=extracted_code,
                parse_method="doctest",
                used_markdown_code_block=used_markdown_code_block,
                total_assertions_found=len(doctest_assertions),
                kept_assertions=len(doctest_assertions),
            )
        return GeneratedTestExtractionResult(
            assertions=[],
            extracted_code=extracted_code,
            parse_method="parse_error",
            used_markdown_code_block=used_markdown_code_block,
            parse_error=str(exc),
        )

    assertions: list[str] = []
    total_assertions_found = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            total_assertions_found += 1
            if _node_mentions_entry_point(node.test, entry_point):
                assertions.append(ast.unparse(node))
            continue

        converted = _convert_unittest_assertion(node, entry_point)
        if converted is not None:
            total_assertions_found += 1
            assertions.append(converted)

    return GeneratedTestExtractionResult(
        assertions=assertions,
        extracted_code=extracted_code,
        parse_method="ast_assert_walk",
        used_markdown_code_block=used_markdown_code_block,
        total_assertions_found=total_assertions_found,
        kept_assertions=len(assertions),
        rejected_assertions=max(total_assertions_found - len(assertions), 0),
    )


def extract_markdown_code_block(text: str) -> tuple[str, bool]:
    """Return the first fenced code block, or the raw text if no block is present."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def _contains_entry_point_definition(code: str, entry_point: str) -> bool:
    """Reject generated tests that redefine or assign the benchmark entry point."""
    definition_pattern = re.compile(
        rf"^\s*(?:async\s+def|def)\s+{re.escape(entry_point)}\b",
        re.MULTILINE,
    )
    assignment_pattern = re.compile(
        rf"^\s*{re.escape(entry_point)}\s*=",
        re.MULTILINE,
    )
    return bool(definition_pattern.search(code) or assignment_pattern.search(code))


def _extract_doctest_assertions(code: str, entry_point: str) -> list[str]:
    """Convert doctest examples into plain assert statements when possible."""
    parser = doctest.DocTestParser()
    assertions: list[str] = []
    for example in parser.get_examples(code):
        source = example.source.strip()
        want = example.want.strip()
        if not source or not want or entry_point not in source:
            continue
        assertions.append(f"assert {source} == {want}")
    return assertions


def _convert_unittest_assertion(node: ast.AST, entry_point: str) -> str | None:
    """Translate a small subset of unittest-style assertions into plain asserts."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    if not isinstance(call.func, ast.Attribute):
        return None
    method_name = call.func.attr
    supported_methods = {"assertEqual", "assertTrue", "assertFalse"}
    if method_name not in supported_methods:
        return None
    if not any(_node_mentions_entry_point(argument, entry_point) for argument in call.args):
        return None

    if method_name == "assertEqual" and len(call.args) >= 2:
        return f"assert {ast.unparse(call.args[0])} == {ast.unparse(call.args[1])}"
    if method_name == "assertTrue" and call.args:
        return f"assert {ast.unparse(call.args[0])}"
    if method_name == "assertFalse" and call.args:
        return f"assert not {ast.unparse(call.args[0])}"
    return None


def _node_mentions_entry_point(node: ast.AST, entry_point: str) -> bool:
    """Check whether one AST node references the target entry point anywhere inside it."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == entry_point:
            return True
        if isinstance(child, ast.Attribute) and child.attr == entry_point:
            return True
    return False
