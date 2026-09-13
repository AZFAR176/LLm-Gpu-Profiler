from __future__ import annotations

from gpu_profiler.detectors.utils import norm, run_ref
from gpu_profiler.models import DetectorFinding, FindingEvidence, StructuredPerformanceState


class CPUBottleneckDetector:
    detector_id = "cpu_starvation_detector_v1"

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        cpu = norm(state.cpu_dataloader_time_pct)
        if cpu <= 0.35:
            return []
        return [
            DetectorFinding(
                finding_id=self.detector_id,
                finding="cpu_data_pipeline_bound",
                score=0.68,
                evidence=[
                    FindingEvidence(
                        metric="cpu_dataloader_time_pct",
                        observed=float(state.cpu_dataloader_time_pct or 0.0),
                        relation=">",
                        threshold=35.0,
                        window="global",
                        reference=run_ref(state, "finding/cpu_starvation_detector_v1/cpu"),
                    )
                ],
            )
        ]
