"""Build internal Pareto-frontier datasets and lightweight SVG plots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ParetoPoint:
    """One experiment summary projected into a cost-quality chart."""

    experiment_name: str
    label: str
    strategy_name: str
    benchmark_name: str
    model_name: str
    series: str
    x_metric: str
    y_metric: str
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    candidate_count: int | None = None
    summary_path: str | None = None
    on_frontier: bool = False


def build_pareto_dataset(
    summaries: Sequence[dict],
    x_metric: str = "completion_tokens_per_problem",
    y_metric: str = "pass_at_1_plus",
) -> list[ParetoPoint]:
    """Project one or more experiment summaries into chart-ready Pareto points."""
    points: list[ParetoPoint] = []
    for summary in summaries:
        metadata = summary["metadata"]
        aggregate = summary["aggregate"]
        label = _strategy_label(metadata)
        points.append(
            ParetoPoint(
                experiment_name=metadata["experiment_name"],
                label=label,
                strategy_name=metadata["strategy"],
                benchmark_name=summary["benchmark"]["name"],
                model_name=metadata["model"],
                series="selected",
                x_metric=x_metric,
                y_metric=y_metric,
                x_mean=float(aggregate[x_metric]["mean"]),
                x_std=float(aggregate[x_metric]["std"]),
                y_mean=float(aggregate[y_metric]["mean"]),
                y_std=float(aggregate[y_metric]["std"]),
                candidate_count=_candidate_count(metadata),
                summary_path=metadata.get("summary_path"),
            )
        )

        if _oracle_metric_name(y_metric) in aggregate:
            points.append(
                ParetoPoint(
                    experiment_name=metadata["experiment_name"],
                    label=f"{label} oracle",
                    strategy_name=metadata["strategy"],
                    benchmark_name=summary["benchmark"]["name"],
                    model_name=metadata["model"],
                    series="oracle",
                    x_metric=x_metric,
                    y_metric=_oracle_metric_name(y_metric),
                    x_mean=float(aggregate[x_metric]["mean"]),
                    x_std=float(aggregate[x_metric]["std"]),
                    y_mean=float(aggregate[_oracle_metric_name(y_metric)]["mean"]),
                    y_std=float(aggregate[_oracle_metric_name(y_metric)]["std"]),
                    candidate_count=_candidate_count(metadata),
                    summary_path=metadata.get("summary_path"),
                )
            )
    return _mark_frontier_points(points)


def build_pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated subset when minimizing x and maximizing y."""
    finite_points = [
        point for point in points if point.series == "selected" and _is_finite_point(point)
    ]
    ordered = sorted(finite_points, key=lambda point: (point.x_mean, -point.y_mean, point.label))
    frontier: list[ParetoPoint] = []
    best_y = -math.inf
    for point in ordered:
        if point.y_mean > best_y:
            frontier.append(point)
            best_y = point.y_mean
    return frontier


def render_pareto_svg(
    points: Sequence[ParetoPoint],
    title: str = "TinyCodeScaling Pareto Frontier",
    log_scale_x: bool = True,
) -> str:
    """Render a small self-contained SVG scatter plot for internal experiment review."""
    finite_points = [point for point in points if _is_finite_point(point)]
    if not finite_points:
        raise ValueError("Cannot render a Pareto plot without at least one finite point.")

    width = 1200
    height = 760
    margin_left = 110
    margin_right = 70
    margin_top = 80
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [point.x_mean for point in finite_points]
    y_values = [point.y_mean for point in finite_points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = 0.0
    y_max = max(max(y_values) * 1.1, 1.0)
    if log_scale_x:
        x_floor = math.floor(math.log10(max(min(x_values), 1e-6)))
        x_ceil = math.ceil(math.log10(max(x_values)))
        x_ticks = [10**power for power in range(x_floor, x_ceil + 1)]
        x_domain_min = min(x_ticks)
        x_domain_max = max(x_ticks)
    else:
        x_domain_min = x_min * 0.9
        x_domain_max = x_max * 1.05 if x_max > 0 else 1.0
        x_ticks = _linear_ticks(x_domain_min, x_domain_max, 5)
    y_ticks = _linear_ticks(y_min, y_max, 5)

    def scale_x(value: float) -> float:
        if log_scale_x:
            numerator = math.log10(value) - math.log10(x_domain_min)
            denominator = math.log10(x_domain_max) - math.log10(x_domain_min)
            return margin_left + plot_width * (numerator / max(denominator, 1e-9))
        return margin_left + plot_width * ((value - x_domain_min) / max(x_domain_max - x_domain_min, 1e-9))

    def scale_y(value: float) -> float:
        return margin_top + plot_height * (1.0 - ((value - y_min) / max(y_max - y_min, 1e-9)))

    frontier = build_pareto_frontier(points)
    frontier_points = sorted(frontier, key=lambda point: point.x_mean)
    frontier_path = " ".join(
        f"L {scale_x(point.x_mean):.1f} {scale_y(point.y_mean):.1f}"
        for point in frontier_points
    )
    if frontier_path:
        first = frontier_points[0]
        frontier_path = (
            f"M {scale_x(first.x_mean):.1f} {scale_y(first.y_mean):.1f} "
            + frontier_path[1:]
        )
    y_axis_label = next((point.y_metric for point in points if point.series == "selected"), "y")

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8" />',
        f'<text x="{margin_left}" y="40" font-size="26" font-family="Menlo, Monaco, monospace" fill="#1f2933">{_escape(title)}</text>',
        f'<text x="{margin_left}" y="62" font-size="14" font-family="Menlo, Monaco, monospace" fill="#52606d">{_escape(_axis_subtitle(points, log_scale_x))}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#9aa5b1" stroke-width="1.5" />',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#9aa5b1" stroke-width="1.5" />',
    ]

    for y_tick in y_ticks:
        y = scale_y(y_tick)
        svg_lines.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e4e7eb" stroke-width="1" />',
                f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13" font-family="Menlo, Monaco, monospace" fill="#52606d">{y_tick:.2f}</text>',
            ]
        )

    for x_tick in x_ticks:
        x = scale_x(x_tick)
        tick_label = _format_x_tick(x_tick, log_scale_x)
        svg_lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#f0f2f4" stroke-width="1" />',
                f'<text x="{x:.1f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="13" font-family="Menlo, Monaco, monospace" fill="#52606d">{tick_label}</text>',
            ]
        )

    if frontier_path:
        svg_lines.append(
            f'<path d="{frontier_path}" fill="none" stroke="#102a43" stroke-width="2.5" stroke-dasharray="8 6" />'
        )

    for point in finite_points:
        x = scale_x(point.x_mean)
        y = scale_y(point.y_mean)
        color = _series_color(point)
        radius = 8 if point.series == "selected" else 6
        stroke = "#102a43" if point.on_frontier else "#243b53"
        fill = "#ffffff" if point.series == "oracle" else color
        svg_lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2" />'
        )
        svg_lines.append(
            f'<text x="{x + 12:.1f}" y="{y - 10:.1f}" font-size="13" font-family="Menlo, Monaco, monospace" fill="#102a43">{_escape(point.label)}</text>'
        )

    svg_lines.extend(
        [
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 34}" text-anchor="middle" font-size="15" font-family="Menlo, Monaco, monospace" fill="#1f2933">{_escape(points[0].x_metric)}</text>',
            f'<text transform="translate(28 {margin_top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="15" font-family="Menlo, Monaco, monospace" fill="#1f2933">{_escape(y_axis_label)}</text>',
        ]
    )

    skipped_count = len(points) - len(finite_points)
    if skipped_count:
        svg_lines.append(
            f'<text x="{width - margin_right}" y="{height - 18}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#7b8794">skipped non-finite points: {skipped_count}</text>'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def write_pareto_artifacts(
    points: Sequence[ParetoPoint],
    output_path: Path,
    title: str = "TinyCodeScaling Pareto Frontier",
    log_scale_x: bool = True,
) -> tuple[Path, Path]:
    """Write both the plot SVG and the companion JSON dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_text = render_pareto_svg(points, title=title, log_scale_x=log_scale_x)
    output_path.write_text(svg_text, encoding="utf-8")

    dataset_path = output_path.with_suffix(".json")
    dataset_payload = {
        "title": title,
        "points": [asdict(point) for point in points],
        "frontier_labels": [point.label for point in build_pareto_frontier(points)],
    }
    dataset_path.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, dataset_path


def load_summary_files(summary_paths: Iterable[Path]) -> list[dict]:
    """Read one or more summary JSON files from disk."""
    summaries: list[dict] = []
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary.setdefault("metadata", {})["summary_path"] = str(path)
        summaries.append(summary)
    return summaries


def _mark_frontier_points(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Attach `on_frontier` flags to selected-series points for plotting and exports."""
    frontier_keys = {
        (point.experiment_name, point.series, point.label)
        for point in build_pareto_frontier(points)
    }
    return [
        ParetoPoint(**{**asdict(point), "on_frontier": (point.experiment_name, point.series, point.label) in frontier_keys})
        for point in points
    ]


def _candidate_count(metadata: Mapping[str, Any]) -> int | None:
    """Read the configured candidate count when a strategy uses multiple samples."""
    strategy_config = metadata.get("strategy_config", {})
    value = strategy_config.get("n", strategy_config.get("n_solutions"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _oracle_metric_name(y_metric: str) -> str:
    """Map selected-solution pass metrics to the matching oracle metric name."""
    if y_metric == "pass_at_1_base":
        return "oracle_pass_at_n_base"
    return "oracle_pass_at_n_plus"


def _strategy_label(metadata: Mapping[str, Any]) -> str:
    """Format a compact point label from strategy metadata."""
    strategy = str(metadata["strategy"])
    strategy_config = metadata.get("strategy_config", {})
    candidate_count = _candidate_count(metadata)
    if candidate_count and candidate_count > 1:
        return f"{strategy} (n={candidate_count})"
    if strategy == "temperature":
        return f"{strategy} (temp={strategy_config.get('temperature', 0.0)})"
    return strategy


def _series_color(point: ParetoPoint) -> str:
    """Choose a stable color per strategy family."""
    palette = {
        "greedy": "#197278",
        "temperature": "#bc4749",
        "best_of_n_random": "#7f5539",
        "public_test_selection": "#386641",
        "generated_test_selection": "#5f0f40",
    }
    return palette.get(point.strategy_name, "#486581")


def _axis_subtitle(points: Sequence[ParetoPoint], log_scale_x: bool) -> str:
    """Describe the selected axes and scaling mode in one compact subtitle."""
    x_metric = points[0].x_metric if points else "x"
    y_metric = next((point.y_metric for point in points if point.series == "selected"), "y")
    x_scale = "log-x" if log_scale_x else "linear-x"
    return f"{x_metric} vs {y_metric} ({x_scale})"


def _is_finite_point(point: ParetoPoint) -> bool:
    """Return whether both coordinates are finite and chartable."""
    return math.isfinite(point.x_mean) and math.isfinite(point.y_mean) and point.x_mean > 0


def _linear_ticks(start: float, end: float, count: int) -> list[float]:
    """Build evenly spaced tick values for a numeric axis."""
    if count <= 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def _format_x_tick(value: float, log_scale_x: bool) -> str:
    """Format x-axis ticks without cluttering the chart."""
    if log_scale_x:
        if value >= 1000:
            return f"{int(value / 1000)}k"
        return str(int(value))
    return f"{value:.0f}"


def _escape(text: str) -> str:
    """Escape a small amount of text for safe SVG embedding."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
