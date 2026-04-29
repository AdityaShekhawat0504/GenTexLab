from __future__ import annotations

from gentexlab.evaluation.base import Metric
from gentexlab.evaluation.clip import CLIPScoreMetric
from gentexlab.evaluation.fft import FFTScoreMetric
from gentexlab.evaluation.seam import SeamScoreMetric
from gentexlab.evaluation.stability import StabilityMetric
from gentexlab.evaluation.tiling import TilingConsistencyMetric


METRIC_REGISTRY = {
    "clip_score": CLIPScoreMetric,
    "seam_score": SeamScoreMetric,
    "tiling_score": TilingConsistencyMetric,
    "tiling_consistency": TilingConsistencyMetric,
    "fft_score": FFTScoreMetric,
    "stability_score": StabilityMetric,
}


def build_metrics(metric_names: list[str]) -> list[Metric]:
    metrics: list[Metric] = []
    for name in metric_names:
        normalized = name.lower()
        if normalized not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'.")
        metrics.append(METRIC_REGISTRY[normalized]())
    return metrics

