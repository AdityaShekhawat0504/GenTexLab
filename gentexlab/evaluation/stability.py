from __future__ import annotations

from collections import defaultdict

from gentexlab.evaluation.base import Metric, MetricResult
from gentexlab.image_utils import matrix_rmse


class StabilityMetric(Metric):
    name = "stability_score"
    scope = "group"

    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        if not records:
            return MetricResult(score=None, available=False, message="No records supplied.")

        grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for item in records:
            grouped[(item.model_name, item.prompt)].append(item)

        siblings = grouped[(record.model_name, record.prompt)]
        if len(siblings) < 2:
            return MetricResult(score=None, available=True, message="Only one sample for this prompt/model.")

        matrices = [item.image.grayscale_matrix(sample_size=24) for item in siblings]
        distances = []
        for index, matrix_a in enumerate(matrices):
            for matrix_b in matrices[index + 1 :]:
                distances.append(matrix_rmse(matrix_a, matrix_b))

        mean_distance = sum(distances) / len(distances) if distances else 0.0
        score = max(0.0, 1.0 - mean_distance)
        return MetricResult(
            score=score,
            details={
                "pairwise_rmse": mean_distance,
                "group_size": len(siblings),
                "comparisons": len(distances),
            },
        )

