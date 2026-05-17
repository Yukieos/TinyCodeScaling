"""Sandboxed Python execution helpers used by selection-time test runners."""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import os
import platform
import shutil
import socket
import subprocess
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class SandboxResult:
    """Structured result returned by sandboxed execution helpers."""

    passed: bool
    status: str
    error_type: str | None = None
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def run_in_sandbox(
    code: str,
    test_code: str,
    timeout_seconds: float = 5.0,
    maximum_memory_bytes: int | None = 512 * 1024 * 1024,
) -> SandboxResult:
    """Execute code plus test code inside a sandboxed subprocess."""
    if _looks_like_memory_bomb(code) or _looks_like_memory_bomb(test_code):
        return SandboxResult(
            passed=False,
            status="memory_error",
            error_type="MemoryError",
            error_message="blocked likely memory-exhausting expression before execution.",
        )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_sandbox_code_worker,
        args=(queue, code, test_code, maximum_memory_bytes),
    )
    process.start()
    process.join(timeout_seconds + 1.0)
    if process.is_alive():
        process.terminate()
        process.join(0.1)
    if process.is_alive():
        process.kill()
        process.join(0.1)
    if process.is_alive() or process.exitcode is None:
        return SandboxResult(
            passed=False,
            status="timeout",
            error_type="TimeoutError",
            error_message=f"Timed out after {timeout_seconds} seconds.",
        )

    if process.exitcode != 0 and queue.empty():
        return SandboxResult(
            passed=False,
            status="runtime_error",
            error_type="ProcessExitError",
            error_message=f"Sandbox subprocess exited with code {process.exitcode}.",
        )

    result = queue.get()
    if result.get("status") == "running":
        return SandboxResult(
            passed=False,
            status="timeout",
            error_type="TimeoutError",
            error_message=f"Timed out after {timeout_seconds} seconds.",
        )
    return SandboxResult(**result)


def run_test_cases_in_sandbox(
    code: str,
    test_cases: Sequence[dict[str, str]],
    timeout_seconds: float = 5.0,
    per_test_timeout_seconds: float = 1.0,
    maximum_memory_bytes: int | None = 512 * 1024 * 1024,
) -> SandboxResult:
    """Execute expression-based public tests and return per-case pass/fail booleans."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_sandbox_test_case_worker,
        args=(
            queue,
            code,
            list(test_cases),
            per_test_timeout_seconds,
            maximum_memory_bytes,
        ),
    )
    process.start()
    process.join(timeout_seconds + 1.0)
    if process.is_alive():
        process.terminate()
        process.join(0.1)
    if process.is_alive():
        process.kill()
        process.join(0.1)
    if process.is_alive() or process.exitcode is None:
        return SandboxResult(
            passed=False,
            status="timeout",
            error_type="TimeoutError",
            error_message=f"Timed out after {timeout_seconds} seconds.",
            details={"case_results": [False for _ in test_cases]},
        )

    if process.exitcode != 0 and queue.empty():
        return SandboxResult(
            passed=False,
            status="runtime_error",
            error_type="ProcessExitError",
            error_message=f"Sandbox subprocess exited with code {process.exitcode}.",
            details={"case_results": [False for _ in test_cases]},
        )

    result = queue.get()
    return SandboxResult(**result)


def run_assertions_in_sandbox(
    code: str,
    assertions: Sequence[str],
    timeout_seconds: float = 5.0,
    per_assertion_timeout_seconds: float = 1.0,
    maximum_memory_bytes: int | None = 512 * 1024 * 1024,
) -> SandboxResult:
    """Execute one code snippet against multiple standalone assert statements."""
    if _looks_like_memory_bomb(code) or any(_looks_like_memory_bomb(assertion) for assertion in assertions):
        return SandboxResult(
            passed=False,
            status="memory_error",
            error_type="MemoryError",
            error_message="blocked likely memory-exhausting expression before execution.",
            details={"assertion_results": [False for _ in assertions]},
        )

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_sandbox_assertion_worker,
        args=(
            queue,
            code,
            list(assertions),
            per_assertion_timeout_seconds,
            maximum_memory_bytes,
        ),
    )
    process.start()
    process.join(timeout_seconds + 1.0)
    if process.is_alive():
        process.terminate()
        process.join(0.1)
    if process.is_alive():
        process.kill()
        process.join(0.1)
    if process.is_alive() or process.exitcode is None:
        return SandboxResult(
            passed=False,
            status="timeout",
            error_type="TimeoutError",
            error_message=f"Timed out after {timeout_seconds} seconds.",
            details={"assertion_results": [False for _ in assertions]},
        )

    if process.exitcode != 0 and queue.empty():
        return SandboxResult(
            passed=False,
            status="runtime_error",
            error_type="ProcessExitError",
            error_message=f"Sandbox subprocess exited with code {process.exitcode}.",
            details={"assertion_results": [False for _ in assertions]},
        )

    result = queue.get()
    return SandboxResult(**result)


def _sandbox_code_worker(
    queue: multiprocessing.Queue,
    code: str,
    test_code: str,
    maximum_memory_bytes: int | None,
) -> None:
    """Run code and test code in an isolated subprocess and emit a structured result."""
    create_tempdir, _, swallow_io, _ = _load_evalplus_sandbox_utils()
    queue.put({"passed": False, "status": "running"})
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        _apply_sandbox_guards(maximum_memory_bytes)
        exec_globals: dict[str, Any] = {"__builtins__": builtins.__dict__}
        try:
            with swallow_io():
                exec(code, exec_globals)
                exec(test_code, exec_globals)
        except BaseException as exc:  # noqa: BLE001
            queue.get_nowait()
            queue.put(_exception_to_result(exc))
        else:
            queue.get_nowait()
            queue.put({"passed": True, "status": "pass", "details": {}})
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


def _sandbox_test_case_worker(
    queue: multiprocessing.Queue,
    code: str,
    test_cases: list[dict[str, str]],
    per_test_timeout_seconds: float,
    maximum_memory_bytes: int | None,
) -> None:
    """Run a batch of public tests in one isolated subprocess."""
    create_tempdir, _, swallow_io, time_limit = _load_evalplus_sandbox_utils()
    with create_tempdir():
        import doctest
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        _apply_sandbox_guards(maximum_memory_bytes)
        exec_globals: dict[str, Any] = {"__builtins__": builtins.__dict__}
        checker = doctest.OutputChecker()
        case_results: list[bool] = []
        try:
            with swallow_io():
                exec(code, exec_globals)
            for case in test_cases:
                try:
                    with time_limit(per_test_timeout_seconds):
                        with swallow_io():
                            value = eval(case["source"], exec_globals)
                    got = repr(value) + "\n"
                    case_results.append(checker.check_output(case["want"], got, 0))
                except BaseException:  # noqa: BLE001
                    case_results.append(False)
        except BaseException as exc:  # noqa: BLE001
            queue.put(
                {
                    **_exception_to_result(exc),
                    "details": {"case_results": [False for _ in test_cases]},
                }
            )
        else:
            queue.put(
                {
                    "passed": all(case_results),
                    "status": "pass" if all(case_results) else "fail",
                    "details": {"case_results": case_results},
                }
            )
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


def _sandbox_assertion_worker(
    queue: multiprocessing.Queue,
    code: str,
    assertions: list[str],
    per_assertion_timeout_seconds: float,
    maximum_memory_bytes: int | None,
) -> None:
    """Run multiple assert statements against one candidate in one isolated subprocess."""
    create_tempdir, _, swallow_io, time_limit = _load_evalplus_sandbox_utils()
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        _apply_sandbox_guards(maximum_memory_bytes)
        exec_globals: dict[str, Any] = {"__builtins__": builtins.__dict__}
        assertion_results: list[bool] = []
        try:
            with swallow_io():
                exec(code, exec_globals)
            for assertion in assertions:
                try:
                    with time_limit(per_assertion_timeout_seconds):
                        with swallow_io():
                            exec(assertion, exec_globals)
                    assertion_results.append(True)
                except BaseException:  # noqa: BLE001
                    assertion_results.append(False)
        except BaseException as exc:  # noqa: BLE001
            queue.put(
                {
                    **_exception_to_result(exc),
                    "details": {"assertion_results": [False for _ in assertions]},
                }
            )
        else:
            queue.put(
                {
                    "passed": all(assertion_results),
                    "status": "pass" if all(assertion_results) else "fail",
                    "details": {"assertion_results": assertion_results},
                }
            )
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


def _apply_sandbox_guards(maximum_memory_bytes: int | None) -> None:
    """Apply EvalPlus-style guards plus clearer blocking stubs for risky operations."""
    _, reliability_guard, _, _ = _load_evalplus_sandbox_utils()
    _apply_memory_limits(maximum_memory_bytes)
    reliability_guard(maximum_memory_bytes=None)
    _install_blocking_stubs()


def _apply_memory_limits(maximum_memory_bytes: int | None) -> None:
    """Apply conservative memory limits while respecting platform hard limits."""
    if maximum_memory_bytes is None or platform.system() == "Windows":
        return
    try:
        import resource
    except ImportError:
        return

    limits = [resource.RLIMIT_AS, resource.RLIMIT_DATA]
    if platform.system() != "Darwin":
        limits.append(resource.RLIMIT_STACK)

    for limit_name in limits:
        try:
            soft, hard = resource.getrlimit(limit_name)
            target = maximum_memory_bytes
            if hard not in (-1, resource.RLIM_INFINITY):
                target = min(target, hard)
            resource.setrlimit(limit_name, (target, target))
        except (OSError, ValueError):
            continue


def _install_blocking_stubs() -> None:
    """Replace dangerous operations with explicit PermissionError stubs."""
    def _blocked(name: str):
        def _raiser(*args, **kwargs):
            raise PermissionError(f"blocked operation: {name}")

        return _raiser

    builtins.open = _blocked("open")
    os.system = _blocked("os.system")
    os.remove = _blocked("os.remove")
    os.removedirs = _blocked("os.removedirs")
    os.rmdir = _blocked("os.rmdir")
    os.rename = _blocked("os.rename")
    os.renames = _blocked("os.renames")
    os.replace = _blocked("os.replace")
    os.unlink = _blocked("os.unlink")
    shutil.rmtree = _blocked("shutil.rmtree")
    shutil.move = _blocked("shutil.move")
    subprocess.Popen = _blocked("subprocess.Popen")  # type: ignore[assignment]
    socket.socket = _blocked("socket.socket")  # type: ignore[assignment]
    socket.create_connection = _blocked("socket.create_connection")  # type: ignore[assignment]
    socket.getaddrinfo = _blocked("socket.getaddrinfo")  # type: ignore[assignment]
    urllib.request.urlopen = _blocked("urllib.request.urlopen")


def _exception_to_result(exc: BaseException) -> dict[str, Any]:
    """Convert an execution exception into a stable sandbox result payload."""
    if isinstance(exc, SyntaxError):
        status = "syntax_error"
    elif isinstance(exc, MemoryError):
        status = "memory_error"
    elif isinstance(exc, PermissionError):
        status = "blocked_operation"
    elif type(exc).__name__ in {"TimeoutException", "TimeoutError"}:
        status = "timeout"
    else:
        status = "runtime_error"
    return {
        "passed": False,
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip(),
        "details": {},
    }


def _load_evalplus_sandbox_utils():
    """Import EvalPlus sandbox helpers lazily so non-runtime tests can still import modules."""
    from evalplus.eval.utils import create_tempdir, reliability_guard, swallow_io, time_limit

    return create_tempdir, reliability_guard, swallow_io, time_limit


def _looks_like_memory_bomb(source: str) -> bool:
    """Heuristically block obvious huge repeated allocations before execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, (ast.List, ast.Tuple, ast.Set, ast.Constant)):
                multiplier = _safe_int_value(node.right)
                if multiplier is not None and multiplier >= 10**8:
                    return True
            if isinstance(node.right, (ast.List, ast.Tuple, ast.Set, ast.Constant)):
                multiplier = _safe_int_value(node.left)
                if multiplier is not None and multiplier >= 10**8:
                    return True
    return False


def _safe_int_value(node: ast.AST) -> int | None:
    """Best-effort evaluator for simple integer expressions used in guards."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _safe_int_value(node.operand)
        return -operand if operand is not None else None
    if isinstance(node, ast.BinOp):
        left = _safe_int_value(node.left)
        right = _safe_int_value(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Pow):
            return left**right
    return None
