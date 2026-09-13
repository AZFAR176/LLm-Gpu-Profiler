from __future__ import annotations

import time
import json
from pathlib import Path

from gpu_profiler.actions.executor import ProfilingActionExecutor
from gpu_profiler.config import RunConfig
from gpu_profiler.detectors.engine import DetectorEngine
from gpu_profiler.diagnosis.baseline import BaselineClassifier
from gpu_profiler.diagnosis.llm_agent import LLMDiagnosticAgent
from gpu_profiler.llm.factory import build_llm_client
from gpu_profiler.models import (
    DetectorFinding,
    DiagnosticOutput,
    LLMCallTrace,
    ProfilerOutput,
    ProfilingActionResult,
    TelemetryFrame,
    ValidationReport,
)
from gpu_profiler.state.builder import StateBuilder
from gpu_profiler.telemetry.collector import TelemetryCollector
from gpu_profiler.validation.evaluator import ValidationEvaluator


def merge_frames(base: TelemetryFrame, followup: TelemetryFrame) -> TelemetryFrame:
    merged_metrics = dict(base.metrics)
    merged_metrics.update(followup.metrics)
    merged_events = [*base.events, *followup.events]
    merged_meta = dict(base.metadata)
    merged_meta["followup_merged"] = True
    context = followup.context if followup.context.run_id else base.context
    return TelemetryFrame(
        metrics=merged_metrics,
        events=merged_events,
        metadata=merged_meta,
        context=context,
    )


class ProfilerPipeline:
    def __init__(self) -> None:
        self.collector = TelemetryCollector()
        self.builder = StateBuilder()
        self.detectors = DetectorEngine()
        self.baseline = BaselineClassifier()
        self.llm_agent = LLMDiagnosticAgent()
        self.action_executor = ProfilingActionExecutor()
        self.validator = ValidationEvaluator()

    def run(self, cfg: RunConfig) -> ProfilerOutput:
        base_frame = self.collector.collect(cfg.input_path, provider=cfg.telemetry_provider)
        current_frame = base_frame
        diagnosis: DiagnosticOutput | None = None
        baseline_diagnosis: DiagnosticOutput | None = None
        llm_diagnosis: DiagnosticOutput | None = None
        findings: list[DetectorFinding] = []
        diagnosis_source = "baseline"
        action_results: list[ProfilingActionResult] = []
        llm_calls: list[LLMCallTrace] = []
        llm_raw_response = (
            cfg.llm_response_path.read_text(encoding="utf-8")
            if cfg.llm_response_path is not None
            else None
        )
        llm_client = build_llm_client(cfg)

        for round_idx in range(cfg.max_reasoning_rounds):
            state = self.builder.build(current_frame)
            findings = self.detectors.detect(state)
            baseline_diagnosis = self.baseline.reason(state, findings)

            if llm_raw_response is not None and round_idx == 0:
                llm_diagnosis = self.llm_agent.parse_llm_json(llm_raw_response)
                diagnosis = llm_diagnosis
                diagnosis_source = "llm"
            elif llm_client is not None:
                llm_diagnosis, trace = self._call_llm_with_retry(
                    cfg=cfg,
                    llm_client=llm_client,
                    state=state,
                    findings=findings,
                )
                llm_calls.append(trace)
                if llm_diagnosis is not None:
                    diagnosis = llm_diagnosis
                    diagnosis_source = "llm"
                else:
                    diagnosis = baseline_diagnosis
                    diagnosis_source = "baseline_fallback"
            else:
                llm_diagnosis = None
                diagnosis = baseline_diagnosis
                diagnosis_source = "baseline"
            if not diagnosis.ask_for_more_profiling:
                break

            # Prefer explicit action execution over generic follow-up merges.
            action_frames = self._execute_requested_actions(
                diagnosis=diagnosis,
                followup_path=cfg.followup_path,
                action_results=action_results,
            )
            for frame in action_frames:
                current_frame = merge_frames(current_frame, frame)

            # Backward-compatible path: optionally merge a single generic followup metrics.json.
            if cfg.followup_path is not None and not action_frames:
                followup_frame = self.collector.collect(cfg.followup_path, provider="file")
                current_frame = merge_frames(current_frame, followup_frame)

            if round_idx == cfg.max_reasoning_rounds - 1:
                break

        assert diagnosis is not None
        state = self.builder.build(current_frame)

        validation: ValidationReport | None = None
        if cfg.validate_path is not None:
            after_state = self.builder.build(self.collector.collect(cfg.validate_path, provider="file"))
            validation = self.validator.compare(before=state, after=after_state)

        output = ProfilerOutput(
            state=state,
            findings=findings,
            diagnosis=diagnosis,
            diagnosis_source=diagnosis_source,
            baseline_diagnosis=baseline_diagnosis,
            llm_diagnosis=llm_diagnosis,
            llm_calls=llm_calls,
            executed_profiling_actions=action_results,
            validation=validation,
        )
        if cfg.dashboard_output_dir is not None:
            self._emit_dashboard_snapshot(output, cfg.dashboard_output_dir)
        return output

    def build_prompt(self, cfg: RunConfig) -> str:
        state = self.builder.build(self.collector.collect(cfg.input_path, provider=cfg.telemetry_provider))
        findings = self.detectors.detect(state)
        return self.llm_agent.build_prompt(state=state, findings=findings, mode=cfg.mode)

    def _execute_requested_actions(
        self,
        diagnosis,
        followup_path: Path | None,
        action_results: list[ProfilingActionResult],
    ) -> list[TelemetryFrame]:
        frames: list[TelemetryFrame] = []
        for request in diagnosis.requested_profiling_actions:
            result, frame = self.action_executor.execute(request, followup_path)
            action_results.append(result)
            if frame is not None:
                frames.append(frame)
        return frames

    def _call_llm_with_retry(
        self,
        cfg: RunConfig,
        llm_client,
        state,
        findings,
    ) -> tuple[DiagnosticOutput | None, LLMCallTrace]:
        prompt = self.llm_agent.build_prompt(state=state, findings=findings, mode=cfg.mode)
        start = time.perf_counter()
        last_err: str | None = None

        for attempt in range(1, cfg.llm_max_retries + 2):
            try:
                result = llm_client.diagnose(prompt)
                diagnosis = self.llm_agent.parse_llm_json(result.raw_json_text)
                elapsed = int((time.perf_counter() - start) * 1000)
                cost = self._estimate_cost(cfg, result.prompt_tokens, result.completion_tokens)
                return diagnosis, LLMCallTrace(
                    provider=llm_client.provider_name,
                    model=result.model,
                    request_id=result.request_id,
                    latency_ms=elapsed,
                    attempt_count=attempt,
                    success=True,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    estimated_cost_usd=cost,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                if attempt <= cfg.llm_max_retries:
                    time.sleep(cfg.llm_retry_backoff_s * attempt)
                    continue

        elapsed = int((time.perf_counter() - start) * 1000)
        return None, LLMCallTrace(
            provider=llm_client.provider_name,
            model=cfg.llm_model,
            latency_ms=elapsed,
            attempt_count=cfg.llm_max_retries + 1,
            success=False,
            error=last_err or "unknown llm error",
        )

    @staticmethod
    def _estimate_cost(
        cfg: RunConfig, prompt_tokens: int | None, completion_tokens: int | None
    ) -> float | None:
        if (
            cfg.llm_input_cost_per_mtok_usd is None
            or cfg.llm_output_cost_per_mtok_usd is None
            or prompt_tokens is None
            or completion_tokens is None
        ):
            return None
        input_cost = (prompt_tokens / 1_000_000) * cfg.llm_input_cost_per_mtok_usd
        output_cost = (completion_tokens / 1_000_000) * cfg.llm_output_cost_per_mtok_usd
        return round(input_cost + output_cost, 8)

    @staticmethod
    def _emit_dashboard_snapshot(output: ProfilerOutput, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = output.model_dump(mode="json")
        (out_dir / "latest_output.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "latest_state.json").write_text(
            json.dumps(payload.get("state", {}), indent=2), encoding="utf-8"
        )
        (out_dir / "latest_diagnosis.json").write_text(
            json.dumps(payload.get("diagnosis", {}), indent=2), encoding="utf-8"
        )
        (out_dir / "latest_validation.json").write_text(
            json.dumps(payload.get("validation", {}), indent=2), encoding="utf-8"
        )
