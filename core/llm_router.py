# -*- coding: utf-8 -*-
# core/llm_router.py
from typing import Optional, Dict
from .llm_openai import OpenAILLM
from .llm_ax import AXLLM  # 이미 있으니 사용
from .llm_midm import MidmLLM  # 이미 있으니 사용
from .llm_ollama import OllamaLLM  # 이미 있으니 사용
from .llm_placeholder import HttpLLM


class LLMRouter:
    """
    페르소나/설정에 따라 백엔드 모델을 선택.
    예) 조진경 → openai:gpt-4o-mini, 또는 ollama:llama3 등
    """

    def __init__(self, default_backend: str = "openai:gpt-4o-mini"):
        self.default_backend = default_backend

    def get_model(self, backend: Optional[str]) -> object:
        name = (backend or self.default_backend).lower()
        if name.startswith("openai:"):
            return OpenAILLM(name.split(":", 1)[1])
        if name.startswith("ollama:"):
            return OllamaLLM(name.split(":", 1)[1])
        if name.startswith("ax:"):
            return AXLLM(name.split(":", 1)[1])
        if name.startswith("midm:"):
            return MidmLLM(name.split(":", 1)[1])
        if name.startswith("http:"):
            # 임의 HTTP 엔진
            return HttpLLM(name, endpoint="http://localhost:8000/chat")
        # fallback
        return OpenAILLM("gpt-4o-mini")

    def complete(
        self, backend: Optional[str], prompt: str, temperature: float = 0.2
    ) -> str:
        m = self.get_model(backend)
        return m.complete(prompt, temperature=temperature)
