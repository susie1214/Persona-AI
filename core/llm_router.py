# # -*- coding: utf-8 -*-
# # core/llm_router.py
# from typing import Optional, Dict
# from .llm_openai import OpenAILLM
# from .llm_ax import AXLLM  # 이미 있으니 사용
# from .llm_midm import MidmLLM  # 이미 있으니 사용
# from .llm_ollama import OllamaLLM  # 이미 있으니 사용
# from .llm_placeholder import HttpLLM


# class LLMRouter:
#     """
#     페르소나/설정에 따라 백엔드 모델을 선택.
#     예) 조진경 → openai:gpt-4o-mini, 또는 ollama:llama3 등
#     """

#     def __init__(self, default_backend: str = "openai:gpt-4o-mini"):
#         self.default_backend = default_backend

#     def get_model(self, backend: Optional[str]) -> object:
#         name = (backend or self.default_backend).lower()
#         if name.startswith("openai:"):
#             return OpenAILLM(name.split(":", 1)[1])
#         if name.startswith("ollama:"):
#             return OllamaLLM(name.split(":", 1)[1])
#         if name.startswith("ax:"):
#             return AXLLM(name.split(":", 1)[1])
#         if name.startswith("midm:"):
#             return MidmLLM(name.split(":", 1)[1])
#         if name.startswith("http:"):
#             # 임의 HTTP 엔진
#             return HttpLLM(name, endpoint="http://localhost:8000/chat")
#         # fallback
#         return OpenAILLM("gpt-4o-mini")

#     def complete(
#         self, backend: Optional[str], prompt: str, temperature: float = 0.2
#     ) -> str:
#         m = self.get_model(backend)
#         return m.complete(prompt, temperature=temperature)
# -*- coding: utf-8 -*-
from typing import Optional
from .llm_openai import OpenAILLM
from .llm_ax import AXLLM
from .llm_midm import MidmLLM
from .llm_ollama import OllamaLLM
from .llm_placeholder import HttpLLM
from .llm_kanana import KananaLLM


# 사용자 친화적 별칭 → 정식 모델 ID로 보정
ALIAS = {
    # Midm 별칭
    "midm-2.0-mini-instruct": "K-Intelligence/Midm-2.0-Mini-Instruct",
    "midm-mini": "K-Intelligence/Midm-2.0-Mini-Instruct",
    # Ollama 예시
    "llama3": "llama3",
    "kanana": "kakaocorp/kanana-1.5-v-3b-instruct",
    "kanana-1.5": "kakaocorp/kanana-1.5-v-3b-instruct",
}

def _normalize(name: str) -> str:
    name = name.strip()
    if ":" not in name:
        return name
    prov, model = name.split(":", 1)
    model_key = model.strip().lower()
    canonical = ALIAS.get(model_key, model.strip())
    return f"{prov.lower()}:{canonical}"

class LLMRouter:
    """
    페르소나/설정에 따라 백엔드 모델을 선택.
    예) openai:gpt-4o-mini, ollama:llama3, midm:K-Intelligence/...
    """
    def __init__(self, default_backend: str = "openai:gpt-4o-mini"):
        self.default_backend = default_backend

    def get_model(self, backend: Optional[str]) -> object:
        name = _normalize((backend or self.default_backend))
        prov, model = (name.split(":", 1) + [""])[:2]

        if prov == "openai":
            return OpenAILLM(model)
        if prov == "ollama":
            return OllamaLLM(model)
        if prov == "ax":
            return AXLLM(model)
        if prov == "midm":
            return MidmLLM(model)
        if prov == "http":
            return HttpLLM(name, endpoint="http://localhost:8000/chat")
        if prov == "kanana":
            # model: 로컬 경로 또는 허깅페이스 모델 ID
            model_path = model or "C:/models/kanana-1.5-v-3b-instruct"
            return KananaLLM(model_id_or_path=model_path)
        # fallback
        return OpenAILLM("gpt-4o-mini")

    def complete(self, backend: Optional[str], prompt: str, temperature: float = 0.2) -> str:
        m = self.get_model(backend)
        return m.complete(prompt, temperature=temperature)
