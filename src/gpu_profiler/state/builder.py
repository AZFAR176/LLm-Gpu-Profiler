from __future__ import annotations

from gpu_profiler.models import MetricDistribution, StructuredPerformanceState, TelemetryFrame
from gpu_profiler.state.features import percentile, skew_ratio, std
from gpu_profiler.state.topology import summarize_topology


KEY_MAP = {
    "gpu_util": "gpu_util_pct",
    "mfu": "mfu_pct",
    "hbm_bw": "hbm_bw_pct",
    "memory_used": "memory_used_pct",
    "nccl_time": "nccl_time_pct",
    "cpu_dataloader_time": "cpu_dataloader_time_pct",
    "gpu_idle": "gpu_idle_pct",
    "step_latency_ms": "step_latency_ms",
    "tokens_per_sec": "throughput_toks_per_sec",
    "sequence_length": "sequence_length",
    "batch_size": "batch_size",
    "world_size": "world_size",
    "kernel_launch_overhead": "kernel_launch_overhead_pct",
}


class StateBuilder:
    """Builds rank-aware structured state from telemetry frames."""

    def build(self, frame: TelemetryFrame) -> StructuredPerformanceState:
        state = StructuredPerformanceState(run_id=frame.context.run_id, context=frame.context)

        extras = {}
        for key, value in frame.metrics.items():
            mapped = KEY_MAP.get(key)
            if mapped is None:
                if isinstance(value, (int, float)):
                    extras[key] = float(value)
                continue
            setattr(state, mapped, value)
        state.extra_metrics = extras

        if frame.events:
            self._populate_from_events(state, frame)
        self._populate_topology_notes(state)
        state.notes.append(f"Loaded metrics={len(frame.metrics)}, events={len(frame.events)}")
        return state

    def _populate_from_events(self, state: StructuredPerformanceState, frame: TelemetryFrame) -> None:
        metric_values: dict[str, list[float]] = {}
        metric_by_rank: dict[str, dict[int, list[float]]] = {}
        phase_counts: dict[str, int] = {}

        for event in frame.events:
            metric_values.setdefault(event.metric_name, []).append(event.metric_value)
            if event.rank is not None:
                metric_by_rank.setdefault(event.metric_name, {}).setdefault(event.rank, []).append(
                    event.metric_value
                )
            if event.phase:
                phase_counts[event.phase] = phase_counts.get(event.phase, 0) + 1

        for metric_name, values in metric_values.items():
            vals = sorted(values)
            mean = sum(values) / len(values)
            by_rank_avg = {
                str(rank): sum(rank_values) / len(rank_values)
                for rank, rank_values in metric_by_rank.get(metric_name, {}).items()
            }
            state.metric_distributions[metric_name] = MetricDistribution(
                mean=mean,
                p50=percentile(vals, 0.50),
                p95=percentile(vals, 0.95),
                p99=percentile(vals, 0.99),
                std=std(values, mean),
                min=min(values),
                max=max(values),
                by_rank=by_rank_avg,
            )

        self._set_if_present(state, "gpu_util", "gpu_util_pct")
        self._set_if_present(state, "mfu", "mfu_pct")
        self._set_if_present(state, "hbm_bw", "hbm_bw_pct")
        self._set_if_present(state, "memory_used", "memory_used_pct")
        self._set_if_present(state, "nccl_time", "nccl_time_pct")
        self._set_if_present(state, "cpu_dataloader_time", "cpu_dataloader_time_pct")
        self._set_if_present(state, "gpu_idle", "gpu_idle_pct")
        self._set_if_present(state, "step_latency_ms", "step_latency_ms")
        self._set_if_present(state, "tokens_per_sec", "throughput_toks_per_sec")
        self._set_if_present(state, "kernel_launch_overhead", "kernel_launch_overhead_pct")

        step_latency_dist = state.metric_distributions.get("step_latency_ms")
        if step_latency_dist:
            state.step_latency_p50_ms = step_latency_dist.p50
            state.step_latency_p95_ms = step_latency_dist.p95
            state.step_latency_p99_ms = step_latency_dist.p99

        rank_gpu_util = state.metric_distributions.get("gpu_util")
        if rank_gpu_util:
            state.rank_utilization_skew = skew_ratio(rank_gpu_util.by_rank)

        rank_step = state.metric_distributions.get("step_latency_ms")
        if rank_step:
            state.rank_step_time_skew = skew_ratio(rank_step.by_rank)

        rank_nccl = state.metric_distributions.get("nccl_time")
        if rank_nccl:
            state.nccl_fraction_by_rank = rank_nccl.by_rank

        rank_idle = state.metric_distributions.get("gpu_idle")
        if rank_idle:
            state.idle_fraction_by_rank = rank_idle.by_rank

        total = sum(phase_counts.values())
        if total > 0:
            state.phase_breakdown_pct = {
                phase: (count / total) * 100.0 for phase, count in phase_counts.items()
            }

    def _populate_topology_notes(self, state: StructuredPerformanceState) -> None:
        topo = summarize_topology(state.context)
        state.topology = topo.model_dump(mode="json")
        if topo.notes:
            state.notes.extend(topo.notes)

    @staticmethod
    def _set_if_present(state: StructuredPerformanceState, metric_name: str, attr: str) -> None:
        dist = state.metric_distributions.get(metric_name)
        if dist is not None:
            setattr(state, attr, dist.mean)
