from __future__ import annotations

from gpu_profiler.config import RunConfig
from gpu_profiler.llm.base import LLMClient
from gpu_profiler.llm.mock import MockLLMClient
from gpu_profiler.llm.openai_compatible import OpenAICompatibleClient


def build_llm_client(cfg: RunConfig) -> LLMClient | None:
    provider = cfg.llm_provider.lower()
    if provider == "disabled":
        return None
    if provider == "mock":
        return MockLLMClient(model=cfg.llm_model)
    if provider == "openai":
        return OpenAICompatibleClient(
            model=cfg.llm_model,
            api_base=cfg.llm_api_base or "https://api.openai.com",
            api_key_env=cfg.llm_api_key_env,
            require_api_key=True,
            timeout_s=cfg.llm_timeout_s,
            temperature=cfg.llm_temperature,
            max_output_tokens=cfg.llm_max_output_tokens,
        )
    if provider == "local_openai":
        # OpenAI-compatible local endpoints (for example local vLLM servers).
        return OpenAICompatibleClient(
            model=cfg.llm_model,
            api_base=cfg.llm_api_base or "http://localhost:8000",
            api_key_env=cfg.llm_api_key_env,
            require_api_key=False,
            timeout_s=cfg.llm_timeout_s,
            temperature=cfg.llm_temperature,
            max_output_tokens=cfg.llm_max_output_tokens,
        )
    raise ValueError(f"Unsupported llm_provider: {cfg.llm_provider}")
