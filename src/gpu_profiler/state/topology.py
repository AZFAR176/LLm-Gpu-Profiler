from __future__ import annotations

from pydantic import BaseModel, Field

from gpu_profiler.models import WorkloadContext


class TopologySummary(BaseModel):
    gpu_model: str | None = None
    gpu_count: int | None = None
    interconnect: str | None = None
    notes: list[str] = Field(default_factory=list)


def summarize_topology(context: WorkloadContext) -> TopologySummary:
    notes: list[str] = []
    if context.interconnect is None:
        notes.append("Interconnect not specified (NVLink/PCIe/IB unknown).")
    if context.gpu_count is None:
        notes.append("GPU count missing from workload context.")
    return TopologySummary(
        gpu_model=context.gpu_model,
        gpu_count=context.gpu_count,
        interconnect=context.interconnect,
        notes=notes,
    )
