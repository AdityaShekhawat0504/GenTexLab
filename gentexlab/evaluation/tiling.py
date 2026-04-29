from __future__ import annotations

from gentexlab.evaluation.base import Metric, MetricResult
from gentexlab.image_utils import average, mean_rgb_difference


class TilingConsistencyMetric(Metric):
    name = "tiling_score"

    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        image = record.image
        tiled = image.tile_2x2()
        edges = image.edge_pixels()
        seam_vertical = mean_rgb_difference(edges["left"], edges["right"])
        seam_horizontal = mean_rgb_difference(edges["top"], edges["bottom"])

        near_vertical = []
        near_horizontal = []
        for y in range(image.height):
            left_inner = image.pixels[y][1]
            right_inner = image.pixels[y][-2]
            near_vertical.append(sum(abs(a - b) for a, b in zip(left_inner, right_inner)) / (3 * 255.0))
        for x in range(image.width):
            top_inner = image.pixels[1][x]
            bottom_inner = image.pixels[-2][x]
            near_horizontal.append(sum(abs(a - b) for a, b in zip(top_inner, bottom_inner)) / (3 * 255.0))

        seam_penalty = average([seam_vertical, seam_horizontal, average(near_vertical), average(near_horizontal)])
        edge_penalty, edge_impl = self._edge_penalty(tiled)
        score = max(0.0, 1.0 - average([seam_penalty, edge_penalty]))
        return MetricResult(
            score=score,
            details={
                "vertical_seam_diff": seam_vertical,
                "horizontal_seam_diff": seam_horizontal,
                "near_vertical_diff": average(near_vertical),
                "near_horizontal_diff": average(near_horizontal),
                "edge_penalty": edge_penalty,
                "edge_implementation": edge_impl,
            },
        )

    def _edge_penalty(self, tiled: Any) -> tuple[float, str]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            matrix = tiled.grayscale_matrix()
            height = len(matrix)
            width = len(matrix[0]) if height else 0
            seam_x = width // 2
            seam_y = height // 2
            vertical = [abs(matrix[y][seam_x] - matrix[y][max(0, seam_x - 1)]) for y in range(height)]
            horizontal = [abs(matrix[seam_y][x] - matrix[max(0, seam_y - 1)][x]) for x in range(width)]
            return average(vertical + horizontal), "grayscale-fallback"

        matrix = tiled.grayscale_matrix()
        grayscale = (np.array(matrix, dtype=np.float32) * 255.0).astype("uint8")
        edges = cv2.Canny(grayscale, 100, 200)
        height, width = edges.shape
        seam_x = width // 2
        seam_y = height // 2
        vertical_band = edges[:, max(0, seam_x - 1) : min(width, seam_x + 2)]
        horizontal_band = edges[max(0, seam_y - 1) : min(height, seam_y + 2), :]
        band_mean = float(vertical_band.mean() + horizontal_band.mean()) / (2.0 * 255.0)
        return band_mean, "opencv-canny"
