"""Public report builder exports."""

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
    "build_markdown_report",
    "build_pareto_dataset",
    "build_pareto_frontier",
    "load_summary_files",
    "render_pareto_svg",
    "write_pareto_artifacts",
]
