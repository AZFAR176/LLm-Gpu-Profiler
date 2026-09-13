from __future__ import annotations

from typing import Protocol

from gpu_profiler.models import DetectorFinding, StructuredPerformanceState


class Detector(Protocol):
    detector_id: str

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        ...
