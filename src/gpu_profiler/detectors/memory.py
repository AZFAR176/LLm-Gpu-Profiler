from __future__ import annotations

from gpu_profiler.detectors.utils import norm, run_ref
from gpu_profiler.models import DetectorFinding, FindingEvidence, StructuredPerformanceState


class MemoryBandwidthDetector:
    detector_id = "memory_bandwidth_detector_v1"

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        mfu = norm(state.mfu_pct)
        hbm = norm(state.hbm_bw_pct)
        if not (mfu < 0.4 and hbm > 0.75):
            return []
        return [
            DetectorFinding(
                finding_id=self.detector_id,
                finding="memory_bandwidth_bound",
                score=0.78,
                evidence=[
                    FindingEvidence(
                        metric="mfu_pct",
                        observed=float(state.mfu_pct or 0.0),
                        relation="<",
                        threshold=40.0,
                        window="global",
                        reference=run_ref(state, "finding/memory_bandwidth_detector_v1/mfu"),
                    ),
                    FindingEvidence(
                        metric="hbm_bw_pct",
                        observed=float(state.hbm_bw_pct or 0.0),
                        relation=">",
                        threshold=75.0,
                        window="global",
                        reference=run_ref(state, "finding/memory_bandwidth_detector_v1/hbm"),
                    ),
                ],
            )
        ]
