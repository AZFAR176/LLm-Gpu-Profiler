from __future__ import annotations

from gpu_profiler.models import StructuredPerformanceState


def norm(value: float | None) -> float:
    if value is None:
        return 0.0
    return value / 100.0 if value > 1.0 else value


def run_ref(state: StructuredPerformanceState, suffix: str) -> str:
    run = state.run_id or "unknown-run"
    return f"trace://{run}/{suffix}"
