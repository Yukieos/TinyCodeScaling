"""Normalize raw model output into benchmark-specific code samples."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def extract_code(
    raw_output: str,
    benchmark: str,
    entry_point: str | None = None,
    task_id: str | None = None,
    extraction_backend: str | None = None,
    fallback_mode: str = "raw",
) -> dict:
    """Run the configured extraction backend and record how the code was derived."""
    backend = extraction_backend or _default_extraction_backend(benchmark)
    if backend == "evalplus":
        return _extract_with_evalplus(
            raw_output=raw_output,
            entry_point=entry_point,
            task_id=task_id,
            fallback_mode=fallback_mode,
        )
    if backend == "livecodebench":
        return _extract_with_livecodebench(
            raw_output=raw_output,
            task_id=task_id,
            fallback_mode=fallback_mode,
        )
    if backend == "raw":
        return _raw_fallback_result(raw_output, backend="raw")
    raise ValueError(f"Unsupported extraction backend: {backend}")


def build_benchmark_sample(
    task_id: str,
    benchmark: str,
    extracted_code: str,
    entry_point: str | None = None,
) -> dict:
    """Convert extracted code into the JSON shape expected by the evaluator."""
    if benchmark in {"humaneval_plus", "mbpp_plus"}:
        return build_evalplus_sample(task_id, extracted_code, entry_point)
    if benchmark == "livecodebench_subset":
        return {"question_id": task_id, "code_list": [extracted_code]}
    raise ValueError(f"Unsupported benchmark for sample building: {benchmark}")


def build_evalplus_sample(task_id: str, extracted_code: str, entry_point: str | None = None) -> dict:
    """Build one EvalPlus sample using `solution` or `completion` as required."""
    normalized = extracted_code.rstrip() + ("\n" if extracted_code.strip() else "")
    if entry_point and _contains_entry_point(normalized, entry_point):
        return {"task_id": task_id, "solution": normalized}
    return {"task_id": task_id, "completion": normalized}


def extract_python_solution(
    text: str,
    entry_point: str | None = None,
    use_evalplus_sanitize: bool = True,
) -> str:
    """Return only the extracted code string for older call sites."""
    backend = "evalplus" if use_evalplus_sanitize else "raw"
    result = extract_code(
        raw_output=text,
        benchmark="humaneval_plus",
        entry_point=entry_point,
        extraction_backend=backend,
    )
    return result["code"]


def _default_extraction_backend(benchmark: str) -> str:
    """Choose the default official extraction backend for a benchmark name."""
    if benchmark in {"humaneval_plus", "mbpp_plus"}:
        return "evalplus"
    if benchmark == "livecodebench_subset":
        return "livecodebench"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _extract_with_evalplus(
    raw_output: str,
    entry_point: str | None,
    task_id: str | None,
    fallback_mode: str,
) -> dict:
    """Use EvalPlus sanitization when available and preserve fallback metadata."""
    evalplus_version = _package_version("evalplus")
    try:
        from evalplus.sanitize import sanitize
    except ImportError:
        return _raw_fallback_result(
            raw_output,
            backend="evalplus",
            method="fallback_raw_due_to_missing_evalplus",
            extra={"evalplus_version": evalplus_version, "task_id": task_id},
        )

    attempts = (
        {"entrypoint": entry_point},
        {"entry_point": entry_point},
        {},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        payload = {key: value for key, value in kwargs.items() if value is not None}
        try:
            cleaned = sanitize(raw_output, **payload)
            if isinstance(cleaned, str):
                return {
                    "code": cleaned.rstrip() + ("\n" if cleaned.strip() else ""),
                    "backend": "evalplus",
                    "method": "evalplus_sanitize",
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "input_length": len(raw_output),
                    "output_length": len(cleaned),
                    "evalplus_version": evalplus_version,
                    "fallback_used": False,
                }
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            break

    if fallback_mode == "raw":
        return _raw_fallback_result(
            raw_output,
            backend="evalplus",
            method=f"fallback_raw_due_to_{type(last_error).__name__ if last_error else 'empty_output'}",
            extra={"evalplus_version": evalplus_version, "task_id": task_id},
        )
    if fallback_mode == "empty":
        return {
            "code": "",
            "backend": "evalplus",
            "method": f"empty_due_to_{type(last_error).__name__ if last_error else 'empty_output'}",
            "task_id": task_id,
            "entry_point": entry_point,
            "input_length": len(raw_output),
            "output_length": 0,
            "evalplus_version": evalplus_version,
            "fallback_used": True,
        }
    raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")


def _extract_with_livecodebench(
    raw_output: str,
    task_id: str | None,
    fallback_mode: str,
) -> dict:
    """Reserve the official LiveCodeBench extraction path for a later milestone."""
    lcb_version = _package_version("livecodebench")
    if fallback_mode == "raw":
        return _raw_fallback_result(
            raw_output,
            backend="livecodebench",
            method="fallback_raw_due_to_unimplemented_livecodebench_wrapper",
            extra={"livecodebench_version": lcb_version, "task_id": task_id},
        )
    if fallback_mode == "empty":
        return {
            "code": "",
            "backend": "livecodebench",
            "method": "empty_due_to_unimplemented_livecodebench_wrapper",
            "task_id": task_id,
            "input_length": len(raw_output),
            "output_length": 0,
            "livecodebench_version": lcb_version,
            "fallback_used": True,
        }
    raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")


def _raw_fallback_result(
    raw_output: str,
    backend: str,
    method: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Return raw model text as code while recording that a fallback was used."""
    result = {
        "code": raw_output.rstrip() + ("\n" if raw_output.strip() else ""),
        "backend": backend,
        "method": method or "raw_passthrough",
        "input_length": len(raw_output),
        "output_length": len(raw_output),
        "fallback_used": True,
    }
    if extra:
        result.update(extra)
    return result


def _contains_entry_point(code: str, entry_point: str) -> bool:
    """Check whether extracted code already defines the benchmark entry function."""
    import re

    pattern = re.compile(
        rf"^\s*(?:async\s+def|def)\s+{re.escape(entry_point)}\b",
        re.MULTILINE,
    )
    return bool(pattern.search(code))


def _package_version(package_name: str) -> str | None:
    """Return an installed package version or None if the package is unavailable."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None
