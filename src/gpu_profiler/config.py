from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    input_path: Path = Field(description="Path to baseline telemetry directory")
    telemetry_provider: str = Field(
        default="file",
        description="Telemetry provider: file | nvml | dcgm | pytorch_profiler",
    )
    mode: str = Field(default="train", description="train or infer")
    followup_path: Path | None = Field(
        default=None,
        description="Optional path with additional profiling telemetry",
    )
    validate_path: Path | None = Field(
        default=None,
        description="Optional post-change telemetry path for validation",
    )
    max_reasoning_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum observe->reason rounds before returning",
    )
    llm_response_path: Path | None = Field(
        default=None,
        description="Optional JSON file containing LLM diagnostic output",
    )
    llm_provider: str = Field(
        default="disabled",
        description="LLM provider: disabled | mock | openai | local_openai",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Model name for the LLM provider")
    llm_api_base: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible providers",
    )
    llm_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable containing provider API key",
    )
    llm_timeout_s: float = Field(default=20.0, ge=1.0, le=120.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_backoff_s: float = Field(default=1.5, ge=0.1, le=30.0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    llm_max_output_tokens: int = Field(default=1200, ge=64, le=8192)
    llm_input_cost_per_mtok_usd: float | None = Field(
        default=None,
        description="Optional cost model: USD per 1M input tokens",
    )
    llm_output_cost_per_mtok_usd: float | None = Field(
        default=None,
        description="Optional cost model: USD per 1M output tokens",
    )
    dashboard_output_dir: Path | None = Field(
        default=None,
        description="Optional directory to emit latest pipeline JSON for dashboard polling",
    )
