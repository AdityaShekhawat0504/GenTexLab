from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import zlib
from pathlib import Path
from typing import Iterable


RGBPixel = tuple[int, int, int]


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def blend_color(color_a: RGBPixel, color_b: RGBPixel, t: float) -> RGBPixel:
    return tuple(
        clamp_channel(a + (b - a) * t) for a, b in zip(color_a, color_b)
    )  # type: ignore[return-value]


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass(slots=True)
class ImageData:
    width: int
    height: int
    pixels: list[list[RGBPixel]]

    def __post_init__(self) -> None:
        if len(self.pixels) != self.height:
            raise ValueError("Image height does not match pixel rows.")
        for row in self.pixels:
            if len(row) != self.width:
                raise ValueError("Image width does not match pixel columns.")

    def save_png(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = []
        for row in self.pixels:
            raw_rows.append(b"\x00" + bytes(channel for pixel in row for channel in pixel))
        compressed = zlib.compress(b"".join(raw_rows), level=9)

        def chunk(tag: bytes, payload: bytes) -> bytes:
            crc = zlib.crc32(tag)
            crc = zlib.crc32(payload, crc)
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", crc & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        png = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", ihdr),
                chunk(b"IDAT", compressed),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(png)

    def grayscale_matrix(self, sample_size: int | None = None) -> list[list[float]]:
        if sample_size is None or sample_size >= min(self.width, self.height):
            xs = list(range(self.width))
            ys = list(range(self.height))
        else:
            xs = self._sample_indices(self.width, sample_size)
            ys = self._sample_indices(self.height, sample_size)

        matrix: list[list[float]] = []
        for y in ys:
            row: list[float] = []
            for x in xs:
                r, g, b = self.pixels[y][x]
                row.append((0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0)
            matrix.append(row)
        return matrix

    def edge_pixels(self) -> dict[str, list[RGBPixel]]:
        left = [self.pixels[y][0] for y in range(self.height)]
        right = [self.pixels[y][-1] for y in range(self.height)]
        top = list(self.pixels[0])
        bottom = list(self.pixels[-1])
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    def center_crop(self, size: int) -> "ImageData":
        size = min(size, self.width, self.height)
        x0 = (self.width - size) // 2
        y0 = (self.height - size) // 2
        cropped = [row[x0 : x0 + size] for row in self.pixels[y0 : y0 + size]]
        return ImageData(width=size, height=size, pixels=cropped)

    def tile_2x2(self) -> "ImageData":
        tiled_rows: list[list[RGBPixel]] = []
        for source_y in range(self.height * 2):
            base_row = self.pixels[source_y % self.height]
            tiled_rows.append(base_row + base_row)
        return ImageData(width=self.width * 2, height=self.height * 2, pixels=tiled_rows)

    @staticmethod
    def _sample_indices(length: int, sample_size: int) -> list[int]:
        if sample_size <= 1:
            return [0]
        if sample_size >= length:
            return list(range(length))
        return [int(round(index * (length - 1) / (sample_size - 1))) for index in range(sample_size)]


def mean_rgb_difference(first: Iterable[RGBPixel], second: Iterable[RGBPixel]) -> float:
    distances: list[float] = []
    for pixel_a, pixel_b in zip(first, second):
        distances.append(sum(abs(a - b) for a, b in zip(pixel_a, pixel_b)) / (3 * 255.0))
    return average(distances)


def matrix_rmse(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> float:
    height = min(len(matrix_a), len(matrix_b))
    width = min(len(matrix_a[0]), len(matrix_b[0])) if height else 0
    error = 0.0
    count = 0
    for y in range(height):
        for x in range(width):
            delta = matrix_a[y][x] - matrix_b[y][x]
            error += delta * delta
            count += 1
    return math.sqrt(error / count) if count else 0.0

