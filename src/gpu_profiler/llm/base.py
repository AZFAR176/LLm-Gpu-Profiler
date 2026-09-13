from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    raw_json_text: str
    request_id: str | None
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient(Protocol):
    provider_name: str

    def diagnose(self, prompt: str) -> LLMResult:
        """Return raw JSON text response for DiagnosticOutput schema."""
