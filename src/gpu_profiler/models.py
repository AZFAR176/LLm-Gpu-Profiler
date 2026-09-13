from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class WorkloadContext(BaseModel):
    run_id: str | None = None
    framework: str | None = None
    model_name: str | None = None
    parameter_count_b: float | None = None
    dtype: str | None = None
    parallelism: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    interconnect: str | None = None


class TelemetryEvent(BaseModel):
    run_id: str | None = None
    host_id: str | None = None
    rank: int | None = None
    local_rank: int | None = None
    gpu_uuid: str | None = None
    timestamp_ms: int
    step_id: int | None = None
    phase: str | None = None
    metric_name: str
    metric_value: float
    unit: str | None = None
    source: str | None = None


class TelemetryFrame(BaseModel):
    """Raw telemetry bundle loaded from disk or stream."""

    metrics: Dict[str, float] = Field(default_factory=dict)
    events: List[TelemetryEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: WorkloadContext = Field(default_factory=WorkloadContext)


class MetricDistribution(BaseModel):
    mean: float
    p50: float
    p95: float
    p99: float
    std: float
    min: float
    max: float
    by_rank: Dict[str, float] = Field(default_factory=dict)


class StructuredPerformanceState(BaseModel):
    """Normalized state fed to deterministic detectors and the LLM."""

    run_id: str | None = None
    gpu_util_pct: float | None = None
    mfu_pct: float | None = None
    hbm_bw_pct: float | None = None
    memory_used_pct: float | None = None
    nccl_time_pct: float | None = None
    cpu_dataloader_time_pct: float | None = None
    gpu_idle_pct: float | None = None
    step_latency_ms: float | None = None
    step_latency_p50_ms: float | None = None
    step_latency_p95_ms: float | None = None
    step_latency_p99_ms: float | None = None
    throughput_toks_per_sec: float | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    world_size: int | None = None
    kernel_launch_overhead_pct: float | None = None

    metric_distributions: Dict[str, MetricDistribution] = Field(default_factory=dict)
    phase_breakdown_pct: Dict[str, float] = Field(default_factory=dict)
    rank_utilization_skew: float | None = None
    rank_step_time_skew: float | None = None
    nccl_fraction_by_rank: Dict[str, float] = Field(default_factory=dict)
    idle_fraction_by_rank: Dict[str, float] = Field(default_factory=dict)
    topology: Dict[str, Any] = Field(default_factory=dict)
    extra_metrics: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    context: WorkloadContext = Field(default_factory=WorkloadContext)


class FindingEvidence(BaseModel):
    metric: str
    observed: float
    relation: str
    threshold: float | None = None
    window: str | None = None
    ranks: List[int] = Field(default_factory=list)
    reference: str


class DetectorFinding(BaseModel):
    finding_id: str
    finding: str
    score: float
    evidence: List[FindingEvidence] = Field(default_factory=list)
    affected_ranks: List[int] = Field(default_factory=list)
    time_window: str | None = None


class ExperimentProposal(BaseModel):
    hypothesis: str
    action: str
    measure: List[str] = Field(default_factory=list)
    stop_condition: str


class ProfilingActionRequest(BaseModel):
    action_id: str
    reason: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ProfilingActionResult(BaseModel):
    action_id: str
    status: str
    details: str


class DiagnosticOutput(BaseModel):
    primary_bottleneck: str
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    likely_root_causes: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    next_experiment: ExperimentProposal | None = None
    ask_for_more_profiling: bool = False
    profiling_request: str | None = None
    requested_profiling_actions: List[ProfilingActionRequest] = Field(default_factory=list)


class ValidationReport(BaseModel):
    improved: bool | None = None
    summary: str
    deltas: Dict[str, float] = Field(default_factory=dict)


class LLMCallTrace(BaseModel):
    provider: str
    model: str
    request_id: str | None = None
    latency_ms: int | None = None
    attempt_count: int = 1
    success: bool = False
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


class ProfilerOutput(BaseModel):
    state: StructuredPerformanceState
    findings: List[DetectorFinding] = Field(default_factory=list)
    diagnosis: DiagnosticOutput
    diagnosis_source: str = "baseline"
    baseline_diagnosis: DiagnosticOutput | None = None
    llm_diagnosis: DiagnosticOutput | None = None
    llm_calls: List[LLMCallTrace] = Field(default_factory=list)
    executed_profiling_actions: List[ProfilingActionResult] = Field(default_factory=list)
    validation: ValidationReport | None = None
