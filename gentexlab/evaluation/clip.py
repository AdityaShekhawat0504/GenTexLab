from __future__ import annotations

from typing import Any

from gentexlab.evaluation.base import Metric, MetricResult


class CLIPScoreMetric(Metric):
    name = "clip_score"
    _backend: Any = None

    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        try:
            backend = self._load_backend()
        except RuntimeError as exc:
            return MetricResult(score=None, available=False, message=str(exc))

        try:
            score = backend.score(record.prompt, record.image)
        except RuntimeError as exc:
            return MetricResult(score=None, available=False, message=str(exc))
        return MetricResult(score=score, details={"implementation": "clip-vit-base-patch32"})

    @classmethod
    def _load_backend(cls) -> Any:
        if cls._backend is not None:
            return cls._backend

        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("CLIP dependencies missing. Install torch, transformers, and Pillow.") from exc

        class _Backend:
            def __init__(self) -> None:
                self.torch = torch
                self.Image = Image
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.model.eval()

            def score(self, prompt: str, image: Any) -> float:
                pil_image = self._to_pil(image)
                inputs = self.processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True)
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    similarity = self.torch.nn.functional.cosine_similarity(image_features, text_features)
                return float(similarity.cpu().item())

            def _to_pil(self, image: Any) -> Any:
                flat = [pixel for row in image.pixels for pixel in row]
                pil = self.Image.new("RGB", (image.width, image.height))
                pil.putdata(flat)
                return pil

        cls._backend = _Backend()
        return cls._backend

