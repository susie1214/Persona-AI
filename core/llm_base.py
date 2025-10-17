# -*- coding: utf-8 -*-
class LLM:
    """모든 LLM 백엔드의 최소 공통 인터페이스."""
    name: str = "llm:base"

    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError

