from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gentexlab.experiments.runner import ExperimentRunner
from gentexlab.experiments.schema import ExperimentConfig, ModelConfig


class RunnerTests(unittest.TestCase):
    def test_runner_creates_summary_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="unit-test-run",
                models=[ModelConfig(name="procedural-baseline", provider="procedural")],
                prompt_categories=["wood"],
                num_samples=1,
                metrics=["seam_score", "tiling_score", "fft_score", "stability_score"],
                width=32,
                height=32,
                output_root=tmpdir,
            )
            summary = ExperimentRunner().run(config)
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "images").exists())


if __name__ == "__main__":
    unittest.main()

