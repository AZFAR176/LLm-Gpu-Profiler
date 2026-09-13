from __future__ import annotations

from pydantic import BaseModel, Field

from gpu_profiler.detectors.engine import DetectorEngine
from gpu_profiler.diagnosis.baseline import BaselineClassifier
from gpu_profiler.models import StructuredPerformanceState


class BenchmarkScenario(BaseModel):
    name: str
    expected_label: str
    state: StructuredPerformanceState


class BenchmarkResult(BaseModel):
    scenario: str
    expected: str
    predicted: str
    confidence: float
    top1_correct: bool


class BenchmarkSummary(BaseModel):
    total: int
    top1_accuracy: float
    results: list[BenchmarkResult] = Field(default_factory=list)


def _scenarios() -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name="cpu_dataloader_starvation",
            expected_label="cpu_data_pipeline_bound",
            state=StructuredPerformanceState(
                cpu_dataloader_time_pct=68,
                mfu_pct=22,
                hbm_bw_pct=30,
                nccl_time_pct=14,
                gpu_idle_pct=21,
                step_latency_ms=810,
                throughput_toks_per_sec=1200,
            ),
        ),
        BenchmarkScenario(
            name="memory_bandwidth_saturation",
            expected_label="memory_bandwidth_bound",
            state=StructuredPerformanceState(
                mfu_pct=29,
                hbm_bw_pct=91,
                nccl_time_pct=12,
                gpu_idle_pct=3,
                step_latency_ms=620,
                throughput_toks_per_sec=1750,
            ),
        ),
        BenchmarkScenario(
            name="nccl_bottleneck",
            expected_label="communication_bound",
            state=StructuredPerformanceState(
                mfu_pct=31,
                hbm_bw_pct=52,
                nccl_time_pct=67,
                gpu_idle_pct=10,
                step_latency_ms=940,
                throughput_toks_per_sec=980,
            ),
        ),
        BenchmarkScenario(
            name="underfed_gpu",
            expected_label="gpu_underfeeding",
            state=StructuredPerformanceState(
                mfu_pct=25,
                hbm_bw_pct=40,
                nccl_time_pct=11,
                gpu_idle_pct=19,
                step_latency_ms=760,
                throughput_toks_per_sec=1300,
            ),
        ),
        BenchmarkScenario(
            name="rank_imbalance_comm",
            expected_label="rank_communication_imbalance",
            state=StructuredPerformanceState(
                mfu_pct=34,
                hbm_bw_pct=57,
                nccl_time_pct=44,
                gpu_idle_pct=7,
                step_latency_ms=700,
                throughput_toks_per_sec=1600,
                nccl_fraction_by_rank={"0": 12, "1": 13, "2": 65, "3": 14},
                metric_distributions={},
            ),
        ),
    ]


def run_benchmark_suite() -> BenchmarkSummary:
    detector = DetectorEngine()
    baseline = BaselineClassifier()
    results: list[BenchmarkResult] = []

    for scenario in _scenarios():
        # Inject synthetic by-rank NCCL distribution through metric_distributions for imbalance case.
        if scenario.state.nccl_fraction_by_rank and "nccl_time" not in scenario.state.metric_distributions:
            from gpu_profiler.models import MetricDistribution

            values = list(scenario.state.nccl_fraction_by_rank.values())
            scenario.state.metric_distributions["nccl_time"] = MetricDistribution(
                mean=sum(values) / len(values),
                p50=values[len(values) // 2],
                p95=max(values),
                p99=max(values),
                std=0.0,
                min=min(values),
                max=max(values),
                by_rank=scenario.state.nccl_fraction_by_rank,
            )

        findings = detector.detect(scenario.state)
        diagnosis = baseline.reason(scenario.state, findings)
        ok = diagnosis.primary_bottleneck == scenario.expected_label
        results.append(
            BenchmarkResult(
                scenario=scenario.name,
                expected=scenario.expected_label,
                predicted=diagnosis.primary_bottleneck,
                confidence=diagnosis.confidence,
                top1_correct=ok,
            )
        )

    total = len(results)
    correct = sum(1 for r in results if r.top1_correct)
    return BenchmarkSummary(
        total=total,
        top1_accuracy=(correct / total) if total else 0.0,
        results=results,
    )
