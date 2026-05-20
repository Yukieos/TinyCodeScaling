"""Public report builder exports."""

from tinycodescaling.reports.failure_analysis import (
    analyze_failures,
    build_failure_report_markdown,
    infer_summary_path,
    write_failure_artifacts,
)
from tinycodescaling.reports.markdown import build_markdown_report
from tinycodescaling.reports.pareto import (
    ParetoPoint,
    build_pareto_dataset,
    build_pareto_frontier,
    load_summary_files,
    render_pareto_svg,
    write_pareto_artifacts,
)

__all__ = [
    "ParetoPoint",
    "analyze_failures",
    "build_failure_report_markdown",
    "build_markdown_report",
    "build_pareto_dataset",
    "build_pareto_frontier",
    "infer_summary_path",
    "load_summary_files",
    "render_pareto_svg",
    "write_failure_artifacts",
    "write_pareto_artifacts",
]
