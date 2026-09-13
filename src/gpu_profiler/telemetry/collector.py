from __future__ import annotations

import json
from pathlib import Path

from gpu_profiler.models import TelemetryEvent, TelemetryFrame, WorkloadContext
from gpu_profiler.telemetry.dcgm import DCGMProvider
from gpu_profiler.telemetry.nvml import NVMLProvider
from gpu_profiler.telemetry.pytorch import PyTorchProfilerProvider


class FileTelemetryProvider:
    provider_name = "file"

    def collect(self, input_path: Path | None = None) -> TelemetryFrame:
        if input_path is None:
            raise ValueError("File provider requires input_path.")
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        file_path = input_path / "metrics.json"
        events_file = input_path / "telemetry_events.json"
        context_file = input_path / "workload_context.json"

        metrics: dict[str, float] = {}
        metadata: dict[str, str] = {"source_path": str(input_path), "provider": self.provider_name}
        events: list[TelemetryEvent] = []
        context = WorkloadContext()

        if file_path.exists():
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", payload)
            metadata.update(payload.get("metadata", {}))
        else:
            metadata["warning_metrics"] = "metrics.json not found"

        if events_file.exists():
            events_payload = json.loads(events_file.read_text(encoding="utf-8"))
            raw_events = events_payload.get("events", events_payload)
            events = [TelemetryEvent.model_validate(evt) for evt in raw_events]
        else:
            metadata["warning_events"] = "telemetry_events.json not found"

        if context_file.exists():
            context = WorkloadContext.model_validate_json(context_file.read_text(encoding="utf-8"))

        return TelemetryFrame(metrics=metrics, events=events, metadata=metadata, context=context)


class TelemetryCollector:
    """Collects telemetry through pluggable providers into one common schema."""

    def __init__(self) -> None:
        self.providers = {
            "file": FileTelemetryProvider(),
            "nvml": NVMLProvider(),
            "dcgm": DCGMProvider(),
            "pytorch_profiler": PyTorchProfilerProvider(),
        }

    def collect(self, input_path: Path | None = None, provider: str = "file") -> TelemetryFrame:
        provider_impl = self.providers.get(provider)
        if provider_impl is None:
            raise ValueError(f"Unknown telemetry provider '{provider}'.")
        return provider_impl.collect(input_path)
