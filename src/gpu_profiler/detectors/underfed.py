from __future__ import annotations

from gpu_profiler.detectors.utils import norm, run_ref
from gpu_profiler.models import DetectorFinding, FindingEvidence, StructuredPerformanceState


class UnderfedGPUDetector:
    detector_id = "gpu_underfeeding_detector_v1"

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        mfu = norm(state.mfu_pct)
        hbm = norm(state.hbm_bw_pct)
        nccl = norm(state.nccl_time_pct)
        idle = norm(state.gpu_idle_pct)
        if not (mfu < 0.4 and hbm < 0.5 and nccl < 0.3 and idle > 0.1):
            return []
        return [
            DetectorFinding(
                finding_id=self.detector_id,
                finding="gpu_underfeeding",
                score=0.72,
                evidence=[
                    FindingEvidence(
                        metric="gpu_idle_pct",
                        observed=float(state.gpu_idle_pct or 0.0),
                        relation=">",
                        threshold=10.0,
                        window="global",
                        reference=run_ref(state, "finding/gpu_underfeeding_detector_v1/idle"),
                    )
                ],
            )
        ]
