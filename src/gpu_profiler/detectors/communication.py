from __future__ import annotations

from gpu_profiler.detectors.utils import norm, run_ref
from gpu_profiler.models import DetectorFinding, FindingEvidence, StructuredPerformanceState


class CommunicationDetector:
    detector_id = "communication_detector_v1"

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        nccl = norm(state.nccl_time_pct)
        if nccl <= 0.5:
            return []
        return [
            DetectorFinding(
                finding_id=self.detector_id,
                finding="communication_bound",
                score=0.81,
                evidence=[
                    FindingEvidence(
                        metric="nccl_time_pct",
                        observed=float(state.nccl_time_pct or 0.0),
                        relation=">",
                        threshold=50.0,
                        window="global",
                        reference=run_ref(state, "finding/communication_detector_v1/nccl"),
                    )
                ],
            )
        ]
