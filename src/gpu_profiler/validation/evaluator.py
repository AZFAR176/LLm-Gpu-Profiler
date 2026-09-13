from __future__ import annotations

from gpu_profiler.models import StructuredPerformanceState, ValidationReport


class ValidationEvaluator:
    """Compares before/after runs to validate recommendation impact."""

    def compare(
        self, before: StructuredPerformanceState, after: StructuredPerformanceState
    ) -> ValidationReport:
        deltas = {}

        if before.step_latency_ms is not None and after.step_latency_ms is not None:
            deltas["step_latency_ms"] = after.step_latency_ms - before.step_latency_ms
        if (
            before.throughput_toks_per_sec is not None
            and after.throughput_toks_per_sec is not None
        ):
            deltas["throughput_toks_per_sec"] = (
                after.throughput_toks_per_sec - before.throughput_toks_per_sec
            )
        if before.mfu_pct is not None and after.mfu_pct is not None:
            deltas["mfu_pct"] = after.mfu_pct - before.mfu_pct
        if before.nccl_time_pct is not None and after.nccl_time_pct is not None:
            deltas["nccl_time_pct"] = after.nccl_time_pct - before.nccl_time_pct

        improved = None
        if "step_latency_ms" in deltas and "throughput_toks_per_sec" in deltas:
            improved = deltas["step_latency_ms"] < 0 and deltas["throughput_toks_per_sec"] > 0

        summary = "Insufficient overlap in metrics for validation."
        if improved is True:
            summary = "Change appears beneficial: lower latency with higher throughput."
        elif improved is False:
            summary = "Change appears non-beneficial or mixed; inspect metric trade-offs."

        return ValidationReport(improved=improved, summary=summary, deltas=deltas)
