# -*- coding: utf-8 -*-
# core/llm/__init__.py
"""
LLM Backend Package

다양한 LLM 백엔드 통합 및 라우팅 (OpenAI, Kanana, Ollama, etc.)
"""

from ..llm_router import LLMRouter

__all__ = [
    'LLMRouter',
]
