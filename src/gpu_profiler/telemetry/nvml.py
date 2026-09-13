from __future__ import annotations

import socket
import time
from pathlib import Path

from gpu_profiler.models import TelemetryEvent, TelemetryFrame


class NVMLProvider:
    """Collects point-in-time GPU telemetry using pynvml when available."""

    provider_name = "nvml"

    def collect(self, input_path: Path | None = None) -> TelemetryFrame:
        del input_path
        now_ms = int(time.time() * 1000)
        host = socket.gethostname()

        try:
            import pynvml  # type: ignore
        except Exception:
            return TelemetryFrame(
                metadata={
                    "provider": self.provider_name,
                    "warning": "pynvml not installed; returning empty NVML frame.",
                }
            )

        pynvml.nvmlInit()
        events: list[TelemetryEvent] = []
        metrics: dict[str, float] = {}

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")

                gpu_util = float(util.gpu)
                mem_util = float((mem.used / mem.total) * 100.0) if mem.total > 0 else 0.0

                events.extend(
                    [
                        TelemetryEvent(
                            host_id=host,
                            rank=idx,
                            local_rank=idx,
                            gpu_uuid=uuid,
                            timestamp_ms=now_ms,
                            phase="snapshot",
                            metric_name="gpu_util",
                            metric_value=gpu_util,
                            unit="pct",
                            source=f"nvml:{name}",
                        ),
                        TelemetryEvent(
                            host_id=host,
                            rank=idx,
                            local_rank=idx,
                            gpu_uuid=uuid,
                            timestamp_ms=now_ms,
                            phase="snapshot",
                            metric_name="memory_used",
                            metric_value=mem_util,
                            unit="pct",
                            source=f"nvml:{name}",
                        ),
                    ]
                )
                metrics[f"gpu_{idx}_util"] = gpu_util
                metrics[f"gpu_{idx}_memory_used"] = mem_util

            if device_count > 0:
                gpu_utils = [metrics[f"gpu_{i}_util"] for i in range(device_count)]
                mem_utils = [metrics[f"gpu_{i}_memory_used"] for i in range(device_count)]
                metrics["gpu_util"] = sum(gpu_utils) / device_count
                metrics["memory_used"] = sum(mem_utils) / device_count
                metrics["world_size"] = float(device_count)
        finally:
            pynvml.nvmlShutdown()

        return TelemetryFrame(
            metrics=metrics,
            events=events,
            metadata={"provider": self.provider_name, "host": host},
        )
