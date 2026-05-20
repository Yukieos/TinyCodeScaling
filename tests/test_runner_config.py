"""Tests for runner startup guidance and CLI model-config overrides."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinycodescaling.cli import _resolve_model_config_path, _resolve_seed_override
from tinycodescaling.models.vllm_runner import (
    _format_vllm_startup_error,
    _suggest_gpu_memory_utilization,
)


class RunnerConfigTests(unittest.TestCase):
    def test_suggest_gpu_memory_utilization_estimates_safe_ratio(self):
        message = (
            "Free memory on device cuda:0 (25.67/47.4 GiB) on startup is less than desired "
            "GPU memory utilization (0.9, 42.66 GiB)."
        )

        suggested = _suggest_gpu_memory_utilization(message)

        self.assertIsNotNone(suggested)
        self.assertLess(suggested, 0.9)
        self.assertGreater(suggested, 0.4)

    def test_format_vllm_startup_error_adds_actionable_guidance(self):
        error = ValueError(
            "Free memory on device cuda:0 (25.67/47.4 GiB) on startup is less than desired "
            "GPU memory utilization (0.9, 42.66 GiB)."
        )

        message = _format_vllm_startup_error(error, gpu_memory_utilization=0.9)

        self.assertIn("gpu_memory_utilization", message)
        self.assertIn("Try a value at or below", message)

    def test_resolve_model_config_path_uses_override_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiment_path = root / "configs" / "experiments" / "demo.yaml"
            experiment_path.parent.mkdir(parents=True)
            experiment_path.write_text("", encoding="utf-8")
            override_path = root / "configs" / "models" / "lowmem.yaml"
            override_path.parent.mkdir(parents=True)
            override_path.write_text("", encoding="utf-8")

            resolved = _resolve_model_config_path(
                config_path=experiment_path,
                experiment_config={"model_config": "configs/models/default.yaml"},
                model_config_override=override_path,
            )

            self.assertEqual(resolved, override_path)

    def test_resolve_seed_override_uses_configured_seeds_by_default(self):
        self.assertEqual(_resolve_seed_override([11, 12, 13], None), [11, 12, 13])

    def test_resolve_seed_override_parses_comma_separated_values(self):
        self.assertEqual(_resolve_seed_override([11, 12, 13], "11"), [11])
        self.assertEqual(_resolve_seed_override([11, 12, 13], "11, 15"), [11, 15])


if __name__ == "__main__":
    unittest.main()
