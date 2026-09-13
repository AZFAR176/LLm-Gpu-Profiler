from __future__ import annotations

import json

from gpu_profiler.llm.base import LLMResult


class MockLLMClient:
    provider_name = "mock"

    def __init__(self, model: str) -> None:
        self.model = model

    def diagnose(self, prompt: str) -> LLMResult:
        del prompt
        payload = {
            "primary_bottleneck": "insufficient_evidence",
            "confidence": 0.3,
            "evidence": ["mock_llm: no live provider configured"],
            "evidence_ids": [],
            "likely_root_causes": ["No remote LLM call was configured."],
            "missing_evidence": ["Provide real telemetry and configure a live LLM provider."],
            "recommended_actions": ["Use --llm-provider openai/local_openai with API settings."],
            "next_experiment": {
                "hypothesis": "Provider wiring is the current bottleneck.",
                "action": "Enable live provider and rerun diagnosis.",
                "measure": ["diagnosis_source", "llm_calls.success"],
                "stop_condition": "LLM call succeeds and returns valid schema output.",
            },
            "ask_for_more_profiling": False,
            "profiling_request": None,
            "requested_profiling_actions": [],
        }
        return LLMResult(
            raw_json_text=json.dumps(payload),
            request_id="mock-request",
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
