from __future__ import annotations

import json
from pathlib import Path

from gpu_profiler.models import TelemetryEvent, TelemetryFrame, WorkloadContext


class PyTorchProfilerProvider:
    """
    Adapter for PyTorch profiler exports in JSON.

    Expected input files:
    - pytorch_metrics.json (aggregate metrics)
    - pytorch_events.json (events, including phase/step/rank)
    - workload_context.json (optional context)
    """

    provider_name = "pytorch_profiler"

    def collect(self, input_path: Path | None = None) -> TelemetryFrame:
        if input_path is None:
            raise ValueError("PyTorch profiler provider requires input_path.")
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        metrics_file = input_path / "pytorch_metrics.json"
        events_file = input_path / "pytorch_events.json"
        context_file = input_path / "workload_context.json"

        metrics: dict[str, float] = {}
        events: list[TelemetryEvent] = []
        metadata: dict[str, str] = {"provider": self.provider_name, "source_path": str(input_path)}
        context = WorkloadContext()

        if metrics_file.exists():
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", payload)
            metadata.update(payload.get("metadata", {}))
        else:
            metadata["warning_metrics"] = "pytorch_metrics.json not found"

        if events_file.exists():
            payload = json.loads(events_file.read_text(encoding="utf-8"))
            raw_events = payload.get("events", payload)
            events = [TelemetryEvent.model_validate(evt) for evt in raw_events]
        else:
            metadata["warning_events"] = "pytorch_events.json not found"

        if context_file.exists():
            context = WorkloadContext.model_validate_json(context_file.read_text(encoding="utf-8"))

        return TelemetryFrame(metrics=metrics, events=events, metadata=metadata, context=context)
