from __future__ import annotations

from gpu_profiler.models import (
    DetectorFinding,
    DiagnosticOutput,
    ExperimentProposal,
    ProfilingActionRequest,
    StructuredPerformanceState,
)


class BaselineClassifier:
    """Deterministic diagnosis over detector findings."""

    def reason(
        self, state: StructuredPerformanceState, findings: list[DetectorFinding]
    ) -> DiagnosticOutput:
        if not findings:
            return DiagnosticOutput(
                primary_bottleneck="insufficient_evidence",
                confidence=0.2,
                evidence=[],
                likely_root_causes=["No detector fired with current signals."],
                missing_evidence=[
                    "NCCL collective breakdown",
                    "per-rank GPU idle timeline",
                    "phase-aware step segmentation",
                ],
                recommended_actions=["Collect additional profiling windows."],
                ask_for_more_profiling=True,
                profiling_request=(
                    "Cannot isolate bottleneck. Collect collective-level NCCL and per-rank idle traces "
                    "for 100 steady-state steps."
                ),
                requested_profiling_actions=[
                    ProfilingActionRequest(
                        action_id="collect_nccl_breakdown_100_steps",
                        reason="No strong detector evidence for communication behavior.",
                        params={"steps": 100, "collectives": True},
                    ),
                    ProfilingActionRequest(
                        action_id="collect_gpu_idle_timeline_100_steps",
                        reason="Need rank-level idle/wait breakdown.",
                        params={"steps": 100, "per_rank": True},
                    ),
                ],
            )

        top = max(findings, key=lambda f: f.score)
        evidence = [f"{ev.metric}={ev.observed} ({ev.relation} {ev.threshold})" for ev in top.evidence]
        evidence_ids = [ev.reference for ev in top.evidence]

        output = DiagnosticOutput(
            primary_bottleneck=top.finding,
            confidence=round(min(0.95, max(0.0, top.score)), 2),
            evidence=evidence,
            evidence_ids=evidence_ids,
            likely_root_causes=self._root_causes(top.finding),
            missing_evidence=self._missing_for(top.finding),
            recommended_actions=self._actions_for(top.finding),
            next_experiment=self._experiment_for(top.finding),
        )

        if top.finding in {"communication_bound", "rank_communication_imbalance"}:
            output.ask_for_more_profiling = True
            output.profiling_request = "Collect collective-level NCCL per-rank latency for 100 steps."
            output.requested_profiling_actions = [
                ProfilingActionRequest(
                    action_id="collect_nccl_collective_breakdown_100_steps",
                    reason="Need finer attribution of communication hotspots.",
                    params={"steps": 100, "collective_breakdown": True, "per_rank": True},
                )
            ]
        return output

    @staticmethod
    def _root_causes(label: str) -> list[str]:
        mapping = {
            "memory_bandwidth_bound": [
                "Memory movement dominates kernel runtime.",
                "Arithmetic intensity is low relative to bandwidth demand.",
            ],
            "gpu_underfeeding": [
                "Input pipeline likely cannot saturate device work queues.",
                "Batching strategy may be too conservative.",
            ],
            "communication_bound": [
                "Collective overhead dominates step critical path.",
                "Parallelism plan likely induces high synchronization cost.",
            ],
            "rank_communication_imbalance": [
                "One or more ranks are slower in collective operations.",
                "Topology or mapping asymmetry may be causing straggler behavior.",
            ],
            "cpu_data_pipeline_bound": [
                "Host-side preprocessing and loading dominate stage runtime."
            ],
        }
        return mapping.get(label, ["Multiple weak signals; bottleneck unclear."])

    @staticmethod
    def _missing_for(label: str) -> list[str]:
        mapping = {
            "memory_bandwidth_bound": ["Kernel roofline evidence by phase."],
            "gpu_underfeeding": ["CPU stage trace and queue backlog metrics."],
            "communication_bound": ["Collective split by all-reduce/all-gather/reduce-scatter."],
            "rank_communication_imbalance": ["NCCL latency histogram by rank and topology mapping."],
            "cpu_data_pipeline_bound": ["Per-stage dataloader/tokenization timing."],
        }
        return mapping.get(label, ["Longer phase-segmented telemetry window."])

    @staticmethod
    def _actions_for(label: str) -> list[str]:
        mapping = {
            "memory_bandwidth_bound": [
                "Enable fused kernels and FlashAttention.",
                "Reduce unnecessary memory movement and activation churn.",
            ],
            "gpu_underfeeding": [
                "Increase effective batch while keeping memory safe.",
                "Increase dataloader workers and prefetch depth.",
            ],
            "communication_bound": [
                "Increase compute/communication overlap.",
                "Compare TP/FSDP configurations while keeping global batch constant.",
            ],
            "rank_communication_imbalance": [
                "Investigate rank-to-device/NIC mapping.",
                "Try topology-aware rank remapping.",
            ],
            "cpu_data_pipeline_bound": [
                "Optimize tokenization and move heavy preprocessing off critical path."
            ],
        }
        return mapping.get(label, ["Collect more evidence before tuning."])

    @staticmethod
    def _experiment_for(label: str) -> ExperimentProposal:
        mapping = {
            "memory_bandwidth_bound": ExperimentProposal(
                hypothesis="Memory bandwidth is limiting throughput.",
                action="Enable fused kernels and compare against baseline.",
                measure=["step_latency_ms", "mfu_pct", "hbm_bw_pct", "throughput_toks_per_sec"],
                stop_condition="Accept if latency decreases >=8% without memory regressions.",
            ),
            "gpu_underfeeding": ExperimentProposal(
                hypothesis="Host/input path starves GPUs.",
                action="Increase dataloader workers and batch by 20%.",
                measure=["gpu_idle_pct", "throughput_toks_per_sec", "step_latency_ms"],
                stop_condition="Accept if idle decreases and throughput increases.",
            ),
            "communication_bound": ExperimentProposal(
                hypothesis="Collectives dominate step time.",
                action="Run TP=4 vs TP=2 with constant global batch.",
                measure=["nccl_time_pct", "step_latency_ms", "mfu_pct", "memory_used_pct"],
                stop_condition="Accept if NCCL fraction decreases with net latency gain.",
            ),
            "rank_communication_imbalance": ExperimentProposal(
                hypothesis="One rank has topology-induced communication slowdown.",
                action="Remap ranks to topology-local NIC/GPU placement.",
                measure=["nccl_time_pct", "rank_step_time_skew", "step_latency_ms"],
                stop_condition="Accept if worst-rank NCCL latency approaches median rank.",
            ),
            "cpu_data_pipeline_bound": ExperimentProposal(
                hypothesis="CPU preprocessing blocks forward progress.",
                action="Cache preprocessed inputs and increase pipeline parallelism.",
                measure=["cpu_dataloader_time_pct", "gpu_idle_pct", "throughput_toks_per_sec"],
                stop_condition="Accept if CPU time share decreases and throughput increases.",
            ),
        }
        return mapping.get(
            label,
            ExperimentProposal(
                hypothesis="Primary bottleneck is unclear.",
                action="Collect broader telemetry and rerun diagnosis.",
                measure=["step_latency_ms", "mfu_pct", "nccl_time_pct"],
                stop_condition="Stop when one detector score exceeds 0.7.",
            ),
        )
