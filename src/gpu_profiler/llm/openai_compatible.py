from __future__ import annotations

import json
import os
from urllib import request

from gpu_profiler.llm.base import LLMResult


class OpenAICompatibleClient:
    provider_name = "openai_compatible"

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key_env: str,
        require_api_key: bool,
        timeout_s: float,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.require_api_key = require_api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def diagnose(self, prompt: str) -> LLMResult:
        key = os.getenv(self.api_key_env)
        if self.require_api_key and not key:
            raise RuntimeError(f"Missing API key in environment variable: {self.api_key_env}")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.api_base}/v1/chat/completions",
            data=body,
            method="POST",
            headers=self._headers(key),
        )

        with request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResult(
            raw_json_text=content,
            request_id=data.get("id"),
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    @staticmethod
    def _headers(key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers
