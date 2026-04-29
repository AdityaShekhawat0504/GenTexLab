from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gentexlab.image_utils import ImageData


@dataclass(slots=True)
class GeneratedSampleRecord:
    artifact_id: str
    experiment_name: str
    model_name: str
    model_id: str
    prompt: str
    raw_prompt: str
    prompt_category: str
    sample_index: int
    seed: int
    image_path: str
    tile_path: str
    generation_seconds: float
    image: ImageData
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    metric_details: dict[str, Any] = field(default_factory=dict)

    def base_row(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "raw_prompt": self.raw_prompt,
            "prompt_category": self.prompt_category,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "image_path": self.image_path,
            "tile_path": self.tile_path,
            "generation_seconds": round(self.generation_seconds, 4),
        }
