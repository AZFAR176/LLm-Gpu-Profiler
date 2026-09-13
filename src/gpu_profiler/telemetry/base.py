from __future__ import annotations

from pathlib import Path
from typing import Protocol

from gpu_profiler.models import TelemetryFrame


class TelemetryProvider(Protocol):
    provider_name: str

    def collect(self, input_path: Path | None = None) -> TelemetryFrame:
        """Collect telemetry and return a normalized frame."""
