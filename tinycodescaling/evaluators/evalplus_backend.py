"""Run the EvalPlus evaluator and normalize its stdout into structured metrics."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def evaluate_with_evalplus(
    samples_path: Path,
    dataset: str,
    timeout_seconds: int,
    parallel: int | None,
) -> dict:
    """Execute EvalPlus for one samples file and parse pass@1 metrics from stdout."""
    command_args = [
        "--dataset",
        dataset,
        "--samples",
        str(samples_path),
        "--i-just-wanna-run",
    ]
    if parallel is not None:
        command_args.extend(["--parallel", str(parallel)])

    env = os.environ.copy()
    env["EVALPLUS_TIMEOUT_PER_TASK"] = str(timeout_seconds)

    try:
        completed = _run_evalplus(command_args, env)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "EvalPlus was not found in the active interpreter or on PATH. "
            "Install runtime dependencies in the same Python environment used to launch tinycodescaling."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"EvalPlus evaluation failed:\n{_format_subprocess_output(exc)}") from exc

    stdout = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    pass_base, pass_plus = _parse_evalplus_stdout(stdout)
    result_path = _find_evalplus_result_path(samples_path)
    result_payload = _load_evalplus_result_payload(result_path)
    return {
        "backend": "evalplus",
        "dataset": dataset,
        "pass_at_1_base": pass_base,
        "pass_at_1_plus": pass_plus,
        "pass_at_k_base": (
            result_payload.get("pass_at_k", {}).get("base", {}) if result_payload else {}
        ),
        "pass_at_k_plus": (
            result_payload.get("pass_at_k", {}).get("plus", {}) if result_payload else {}
        ),
        "eval_results": result_payload.get("eval") if result_payload else None,
        "stdout": stdout,
        "cache_path": str(result_path) if result_path else None,
        "result_path": str(result_path) if result_path else None,
    }


def _run_evalplus(command_args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Prefer the active interpreter's EvalPlus module and only then try PATH."""
    attempts = [
        [sys.executable, "-m", "evalplus.evaluate", *command_args],
        ["evalplus.evaluate", *command_args],
    ]
    last_error: FileNotFoundError | subprocess.CalledProcessError | None = None

    for index, command in enumerate(attempts):
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
        except FileNotFoundError as exc:
            last_error = exc
            continue
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if index == 0 and _is_missing_evalplus_module(exc):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("EvalPlus invocation failed before running any subprocess.")


def _is_missing_evalplus_module(exc: subprocess.CalledProcessError) -> bool:
    """Detect the specific failure mode where the module is missing."""
    output = _format_subprocess_output(exc)
    missing_markers = (
        "No module named evalplus.evaluate",
        "No module named 'evalplus.evaluate'",
        "No module named evalplus",
        "No module named 'evalplus'",
    )
    return any(marker in output for marker in missing_markers)


def _format_subprocess_output(process: subprocess.CalledProcessError) -> str:
    """Join subprocess stdout and stderr into one readable error string."""
    return "\n".join(part for part in [process.stdout, process.stderr] if part)


def _parse_evalplus_stdout(stdout: str) -> tuple[float, float]:
    """Extract Base and Base + Extra pass@1 values from EvalPlus text output."""
    base_match = _search_block(stdout, "Base")
    plus_match = _search_block(stdout, "Base + Extra")
    if base_match is None or plus_match is None:
        raise RuntimeError(f"Could not parse EvalPlus output:\n{stdout}")

    base = ast.literal_eval(base_match)
    plus = ast.literal_eval(plus_match)
    return float(base["pass@1"]), float(plus["pass@1"])


def _search_block(text: str, header: str) -> str | None:
    """Find the first JSON-like metrics block that follows a named stdout header."""
    import re

    pattern = re.compile(rf"{re.escape(header)}\s*\n(\{{.*?\}})")
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def _find_evalplus_result_path(samples_path: Path) -> Path | None:
    """Locate the EvalPlus result file emitted next to the samples file, if any."""
    stem = samples_path.stem
    candidates = list(samples_path.parent.glob(f"{stem}*_eval_results.json*"))
    return candidates[0] if candidates else None


def _load_evalplus_result_payload(result_path: Path | None) -> dict | None:
    """Load the EvalPlus result JSON file when evaluation produced one."""
    if result_path is None or not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))
