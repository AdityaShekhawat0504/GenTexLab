from __future__ import annotations

from gentexlab.evaluation.base import Metric, MetricResult
from gentexlab.image_utils import mean_rgb_difference


class SeamScoreMetric(Metric):
    name = "seam_score"

    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        edges = record.image.edge_pixels()
        lr_diff = mean_rgb_difference(edges["left"], edges["right"])
        tb_diff = mean_rgb_difference(edges["top"], edges["bottom"])
        mismatch = (lr_diff + tb_diff) / 2.0
        score = max(0.0, 1.0 - mismatch)
        return MetricResult(
            score=score,
            details={"left_right_diff": lr_diff, "top_bottom_diff": tb_diff, "mismatch": mismatch},
        )

