from __future__ import annotations

from gpu_profiler.detectors.communication import CommunicationDetector
from gpu_profiler.detectors.cpu import CPUBottleneckDetector
from gpu_profiler.detectors.imbalance import RankImbalanceDetector
from gpu_profiler.detectors.memory import MemoryBandwidthDetector
from gpu_profiler.detectors.underfed import UnderfedGPUDetector
from gpu_profiler.models import DetectorFinding, StructuredPerformanceState


class DetectorEngine:
    """Runs deterministic detectors and returns aggregated findings."""

    def __init__(self) -> None:
        self.detectors = [
            MemoryBandwidthDetector(),
            CommunicationDetector(),
            UnderfedGPUDetector(),
            RankImbalanceDetector(),
            CPUBottleneckDetector(),
        ]

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []
        for detector in self.detectors:
            findings.extend(detector.detect(state))
        return findings
