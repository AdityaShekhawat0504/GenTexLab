from __future__ import annotations

import cmath
import math

from gentexlab.evaluation.base import Metric, MetricResult


class FFTScoreMetric(Metric):
    name = "fft_score"

    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        matrix = record.image.grayscale_matrix(sample_size=16)
        try:
            import numpy as np  # type: ignore
        except ImportError:
            spectrum = self._pure_python_spectrum(matrix)
            return self._score_from_spectrum(spectrum, source="dft-fallback")

        array = np.array(matrix, dtype=float)
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(array)))
        spectrum = magnitude.tolist()
        return self._score_from_spectrum(spectrum, source="numpy-fft")

    def _score_from_spectrum(self, spectrum: list[list[float]], source: str) -> MetricResult:
        height = len(spectrum)
        width = len(spectrum[0]) if height else 0
        center_y = height // 2
        center_x = width // 2
        values: list[float] = []
        peaks: list[float] = []

        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance <= 1.5:
                    continue
                value = spectrum[y][x]
                values.append(value)
                peaks.append(value)

        peaks.sort(reverse=True)
        selected_peaks = peaks[: min(12, len(peaks))]
        mean_energy = sum(values) / len(values) if values else 1.0
        peak_energy = sum(selected_peaks) / len(selected_peaks) if selected_peaks else 0.0
        artifact_ratio = peak_energy / (mean_energy + 1e-9)
        score = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, artifact_ratio - 1.0) / 4.0)))
        return MetricResult(
            score=score,
            details={
                "artifact_ratio": artifact_ratio,
                "mean_energy": mean_energy,
                "peak_energy": peak_energy,
                "implementation": source,
            },
        )

    def _pure_python_spectrum(self, matrix: list[list[float]]) -> list[list[float]]:
        size_y = len(matrix)
        size_x = len(matrix[0]) if size_y else 0
        spectrum = [[0.0 for _ in range(size_x)] for _ in range(size_y)]
        for ky in range(size_y):
            for kx in range(size_x):
                total = 0j
                for y in range(size_y):
                    for x in range(size_x):
                        angle = -2.0 * math.pi * ((kx * x / size_x) + (ky * y / size_y))
                        total += matrix[y][x] * cmath.exp(1j * angle)
                spectrum[ky][kx] = abs(total)
        return spectrum

