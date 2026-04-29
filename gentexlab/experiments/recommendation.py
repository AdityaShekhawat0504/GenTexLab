from __future__ import annotations

from collections import defaultdict
from typing import Any


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _composite_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    total_weight = 0.0
    weighted = 0.0
    for metric_name, weight in weights.items():
        value = row.get(metric_name)
        if isinstance(value, (int, float)):
            weighted += float(value) * weight
            total_weight += weight
    if total_weight == 0:
        return None
    return weighted / total_weight


def build_recommendations(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    category_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        category_groups[row["prompt_category"]][row["model_name"]].append(row)
        model_groups[row["model_name"]].append(row)

    category_rankings: dict[str, list[dict[str, Any]]] = {}
    recommendations: dict[str, str] = {}

    for category, model_map in category_groups.items():
        ranking: list[dict[str, Any]] = []
        for model_name, model_rows in model_map.items():
            aggregate: dict[str, Any] = {"model_name": model_name, "sample_count": len(model_rows)}
            for metric_name in weights:
                aggregate[metric_name] = _mean(_numeric_values(model_rows, metric_name))
            aggregate["composite_score"] = _composite_score(aggregate, weights)
            ranking.append(aggregate)
        ranking.sort(key=lambda item: item.get("composite_score") or -1.0, reverse=True)
        category_rankings[category] = ranking
        if ranking:
            best = ranking[0]
            highlights = []
            for metric_name in ("clip_score", "seam_score", "tiling_score", "fft_score", "stability_score"):
                value = best.get(metric_name)
                if isinstance(value, float):
                    highlights.append(f"{metric_name}={value:.3f}")
            recommendations[category] = (
                f"For {category} textures: {best['model_name']} performs best "
                f"with composite score {best['composite_score']:.3f} ({', '.join(highlights)})."
            )

    overall_ranking: list[dict[str, Any]] = []
    for model_name, model_rows in model_groups.items():
        aggregate = {"model_name": model_name, "sample_count": len(model_rows)}
        for metric_name in weights:
            aggregate[metric_name] = _mean(_numeric_values(model_rows, metric_name))
        aggregate["composite_score"] = _composite_score(aggregate, weights)
        overall_ranking.append(aggregate)
    overall_ranking.sort(key=lambda item: item.get("composite_score") or -1.0, reverse=True)

    return {
        "category_rankings": category_rankings,
        "overall_ranking": overall_ranking,
        "recommendations": recommendations,
    }

