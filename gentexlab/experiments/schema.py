from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "clip_score",
    "seam_score",
    "tiling_score",
    "fft_score",
    "stability_score",
]


@dataclass(slots=True)
class ModelConfig:
    name: str
    provider: str = "auto"
    model_id: str | None = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    precision: str = "float16"
    device: str = "auto"
    disable_safety_checker: bool = True

    @classmethod
    def from_raw(cls, raw: str | dict[str, Any]) -> "ModelConfig":
        if isinstance(raw, str):
            provider = "procedural" if raw in {"procedural-baseline", "procedural", "lightweight-diffusion"} else "diffusers"
            return cls(name=raw, provider=provider)
        return cls(
            name=raw["name"],
            provider=raw.get("provider", "auto"),
            model_id=raw.get("model_id"),
            guidance_scale=float(raw.get("guidance_scale", 7.5)),
            num_inference_steps=int(raw.get("num_inference_steps", 30)),
            precision=raw.get("precision", "float16"),
            device=raw.get("device", "auto"),
            disable_safety_checker=bool(raw.get("disable_safety_checker", True)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    models: list[ModelConfig]
    prompts: list[str] = field(default_factory=list)
    prompt_categories: list[str] = field(default_factory=list)
    num_samples: int = 1
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    seed: int = 42
    width: int = 512
    height: int = 512
    negative_prompt: str = ""
    output_root: str = "results/experiments"
    recommendation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "clip_score": 0.35,
            "seam_score": 0.25,
            "tiling_score": 0.20,
            "fft_score": 0.10,
            "stability_score": 0.10,
        }
    )
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_name: str = "gentexlab-experiment") -> "ExperimentConfig":
        raw_models = raw.get("models")
        if raw_models is None and "model" in raw:
            raw_models = [raw["model"]]
        if not raw_models:
            raw_models = ["procedural-baseline"]

        models = [ModelConfig.from_raw(item) for item in raw_models]
        image_size = raw.get("image_size")
        width = int(raw.get("width", image_size or 512))
        height = int(raw.get("height", image_size or 512))

        return cls(
            experiment_name=raw.get("experiment_name", source_name),
            models=models,
            prompts=list(raw.get("prompts", [])),
            prompt_categories=list(raw.get("prompt_categories", raw.get("categories", []))),
            num_samples=int(raw.get("num_samples", 1)),
            metrics=list(raw.get("metrics", DEFAULT_METRICS)),
            seed=int(raw.get("seed", 42)),
            width=width,
            height=height,
            negative_prompt=raw.get("negative_prompt", ""),
            output_root=raw.get("output_root", "results/experiments"),
            recommendation_weights=dict(raw.get("recommendation_weights", {}))
            or {
                "clip_score": 0.35,
                "seam_score": 0.25,
                "tiling_score": 0.20,
                "fft_score": 0.10,
                "stability_score": 0.10,
            },
            notes=raw.get("notes", ""),
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs. Use JSON or install PyYAML.") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    return ExperimentConfig.from_dict(payload, source_name=path.stem)
