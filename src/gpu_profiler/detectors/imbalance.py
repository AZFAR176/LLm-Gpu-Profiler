from __future__ import annotations

from gpu_profiler.detectors.utils import run_ref
from gpu_profiler.models import DetectorFinding, FindingEvidence, StructuredPerformanceState


class RankImbalanceDetector:
    detector_id = "rank_imbalance_detector_v1"

    def detect(self, state: StructuredPerformanceState) -> list[DetectorFinding]:
        rank_nccl = state.metric_distributions.get("nccl_time")
        if not rank_nccl or not rank_nccl.by_rank:
            return []

        values = list(rank_nccl.by_rank.values())
        min_v = min(values)
        max_v = max(values)
        if min_v <= 0 or (max_v / min_v) < 1.8:
            return []

        worst_rank = max(rank_nccl.by_rank, key=rank_nccl.by_rank.get)
        return [
            DetectorFinding(
                finding_id=self.detector_id,
                finding="rank_communication_imbalance",
                score=0.74,
                evidence=[
                    FindingEvidence(
                        metric="nccl_time_by_rank",
                        observed=max_v,
                        relation="ratio_to_min>=",
                        threshold=1.8,
                        window="global",
                        ranks=[int(worst_rank)],
                        reference=run_ref(state, "finding/rank_imbalance_detector_v1/nccl_by_rank"),
                    )
                ],
                affected_ranks=[int(worst_rank)],
            )
        ]
