"""Tests for Pareto dataset building and lightweight plot rendering."""

from __future__ import annotations

import unittest

from tinycodescaling.reports.pareto import (
    ParetoPoint,
    build_pareto_dataset,
    build_pareto_frontier,
    render_pareto_svg,
)


class ParetoDatasetTests(unittest.TestCase):
    def test_build_pareto_dataset_adds_oracle_series_when_available(self):
        summary = {
            "metadata": {
                "experiment_name": "bestof8",
                "strategy": "best_of_n_random",
                "strategy_config": {"n": 8, "temperature": 0.8},
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            },
            "benchmark": {"name": "humaneval_plus"},
            "aggregate": {
                "completion_tokens_per_problem": {"mean": 3200.0, "std": 120.0},
                "pass_at_1_plus": {"mean": 0.21, "std": 0.01},
                "oracle_pass_at_n_plus": {"mean": 0.38, "std": 0.02},
            },
        }

        points = build_pareto_dataset([summary])

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].series, "selected")
        self.assertEqual(points[1].series, "oracle")
        self.assertEqual(points[1].label, "best_of_n_random (n=8) oracle")


class ParetoFrontierTests(unittest.TestCase):
    def test_build_pareto_frontier_keeps_only_non_dominated_selected_points(self):
        points = [
            ParetoPoint(
                experiment_name="a",
                label="greedy",
                strategy_name="greedy",
                benchmark_name="humaneval_plus",
                model_name="model",
                series="selected",
                x_metric="completion_tokens_per_problem",
                y_metric="pass_at_1_plus",
                x_mean=100.0,
                x_std=0.0,
                y_mean=0.30,
                y_std=0.0,
            ),
            ParetoPoint(
                experiment_name="b",
                label="dominated",
                strategy_name="temperature",
                benchmark_name="humaneval_plus",
                model_name="model",
                series="selected",
                x_metric="completion_tokens_per_problem",
                y_metric="pass_at_1_plus",
                x_mean=140.0,
                x_std=0.0,
                y_mean=0.28,
                y_std=0.0,
            ),
            ParetoPoint(
                experiment_name="c",
                label="public",
                strategy_name="public_test_selection",
                benchmark_name="humaneval_plus",
                model_name="model",
                series="selected",
                x_metric="completion_tokens_per_problem",
                y_metric="pass_at_1_plus",
                x_mean=180.0,
                x_std=0.0,
                y_mean=0.42,
                y_std=0.0,
            ),
        ]

        frontier = build_pareto_frontier(points)

        self.assertEqual([point.label for point in frontier], ["greedy", "public"])

    def test_render_pareto_svg_returns_svg_markup(self):
        points = build_pareto_dataset(
            [
                {
                    "metadata": {
                        "experiment_name": "greedy",
                        "strategy": "greedy",
                        "strategy_config": {"n": 1},
                        "model": "model",
                    },
                    "benchmark": {"name": "humaneval_plus"},
                    "aggregate": {
                        "completion_tokens_per_problem": {"mean": 180.0, "std": 0.0},
                        "pass_at_1_plus": {"mean": 0.20, "std": 0.01},
                    },
                },
                {
                    "metadata": {
                        "experiment_name": "bestof4",
                        "strategy": "best_of_n_random",
                        "strategy_config": {"n": 4, "temperature": 0.8},
                        "model": "model",
                    },
                    "benchmark": {"name": "humaneval_plus"},
                    "aggregate": {
                        "completion_tokens_per_problem": {"mean": 720.0, "std": 5.0},
                        "pass_at_1_plus": {"mean": 0.27, "std": 0.01},
                        "oracle_pass_at_n_plus": {"mean": 0.41, "std": 0.02},
                    },
                },
            ]
        )

        svg = render_pareto_svg(points, title="Internal Pareto")

        self.assertIn("<svg", svg)
        self.assertIn("Internal Pareto", svg)
        self.assertIn("best_of_n_random (n=4)", svg)


if __name__ == "__main__":
    unittest.main()
