from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetricResult:
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    message: str = ""


class Metric(ABC):
    name: str
    scope: str = "image"

    @abstractmethod
    def evaluate(self, record: Any, records: list[Any] | None = None) -> MetricResult:
        raise NotImplementedError

