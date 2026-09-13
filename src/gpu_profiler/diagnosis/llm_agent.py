from __future__ import annotations

import json
from textwrap import dedent

from pydantic import ValidationError

from gpu_profiler.models import DetectorFinding, DiagnosticOutput, StructuredPerformanceState


class LLMDiagnosticAgent:
    """Prompt + schema contract for LLM-based bottleneck reasoning."""

    def build_prompt(
        self, state: StructuredPerformanceState, findings: list[DetectorFinding], mode: str
    ) -> str:
        state_json = json.dumps(state.model_dump(mode="json"), indent=2)
        findings_json = json.dumps([f.model_dump(mode="json") for f in findings], indent=2)
        schema_json = json.dumps(
            DiagnosticOutput.model_json_schema(mode="validation"),
            indent=2,
        )
        return dedent(
            f"""
            You are an expert GPU performance debugging agent for {mode} workloads.
            Reason ONLY over provided telemetry facts. Do not invent measurements.

            TASK:
            1) Diagnose the primary bottleneck.
            2) Provide confidence and evidence with exact metric references.
            3) If evidence is insufficient, set ask_for_more_profiling=true and provide:
               - missing_evidence
               - profiling_request
               - requested_profiling_actions
            4) Recommend concrete optimization actions and one next experiment.
            5) Cross-check deterministic detector findings and resolve contradictions.

            REQUIRED OUTPUT:
            - Return STRICT JSON, no markdown, no extra text.
            - JSON must validate against this schema:
            {schema_json}

            STRUCTURED PERFORMANCE STATE:
            {state_json}

            DETERMINISTIC DETECTOR FINDINGS:
            {findings_json}
            """
        ).strip()

    def parse_llm_json(self, raw_response: str) -> DiagnosticOutput:
        try:
            return DiagnosticOutput.model_validate_json(raw_response)
        except ValidationError as exc:
            raise ValueError(
                "LLM output failed schema validation. Ensure strict JSON with required fields."
            ) from exc
