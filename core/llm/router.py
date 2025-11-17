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
from .backends.openai import OpenAILLM
from .backends.ax import AXLLM
from .backends.midm import MidmLLM
from .backends.ollama import OllamaLLM
from .backends.kanana import KananaLLM
from .backends.placeholder import HttpLLM
from core.persona import AdapterManager

# 사용자 친화적 별칭 → 정식 모델 ID로 보정
ALIAS = {
    # Midm 별칭
    "midm-2.0-mini-instruct": "K-Intelligence/Midm-2.0-Mini-Instruct",
    "midm-mini": "K-Intelligence/Midm-2.0-Mini-Instruct",
    # Kanana 별칭
    "kanana-1.5-v-3b-instruct": "kakaocorp/kanana-1.5-v-3b-instruct",
    "kanana": "kakaocorp/kanana-1.5-v-3b-instruct",
    # Ollama 예시
    "llama3": "llama3",
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
    QLoRA 어댑터 지원: Kanana 모델의 경우 페르소나별 어댑터 사용
    """
    def __init__(self, default_backend: str = "openai:gpt-4o-mini"):
        self.default_backend = default_backend
        self.adapter_manager: Optional[AdapterManager] = None
        self.active_speaker_id: Optional[str] = None

    def init_adapter_manager(self, use_4bit: bool = True) -> bool:
        """
        어댑터 매니저 초기화 (Kanana 모델용)

        Args:
            use_4bit: 4-bit 양자화 사용 여부

        Returns:
            성공 여부
        """
        try:
            self.adapter_manager = AdapterManager(use_4bit=use_4bit)
            if self.adapter_manager.load_base("models/kanana-1.5-2.1b-instruct"):
                # 저장된 어댑터들 자동 로드
                self.adapter_manager.load_all_adapters("adapters")
                print(f"[INFO] AdapterManager initialized with {len(self.adapter_manager.get_loaded_adapters())} adapters")
                return True
            return False
        except Exception as e:
            print(f"[WARN] AdapterManager initialization failed: {e}")
            self.adapter_manager = None
            return False

    def set_active_speaker(self, speaker_id: Optional[str]):
        """활성 화자 (어댑터) 설정"""
        self.active_speaker_id = speaker_id
        if self.adapter_manager:
            self.adapter_manager.set_active(speaker_id)

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
        if prov == "kanana":
            return KananaLLM(model)
        if prov == "http":
            return HttpLLM(name, endpoint="http://localhost:8000/chat")
        # fallback
        return OpenAILLM("gpt-4o-mini")

    def complete(self, backend: Optional[str], prompt: str, temperature: float = 0.2, max_new_tokens: Optional[int] = None) -> str:
        """
        LLM 완료 (완성 텍스트 생성)

        Kanana + 어댑터: 개인화된 응답
        다른 모델: 기본 응답
        """
        name = _normalize((backend or self.default_backend))
        prov, _ = (name.split(":", 1) + [""])[:2]

        # Kanana 모델이고 어댑터가 활성화되어 있으면 어댑터 사용
        if prov == "kanana" and self.adapter_manager and self.active_speaker_id:
            try:
                return self.adapter_manager.generate(
                    prompt,
                    max_new_tokens=max_new_tokens or 256,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[WARN] Adapter generation failed: {e}, falling back to base model")

        # 기본: LLMRouter 모델 사용
        m = self.get_model(backend)
        try:
            if max_new_tokens is not None:
                return m.complete(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            else:
                return m.complete(prompt, temperature=temperature)
        except TypeError:
            # max_new_tokens을 지원하지 않는 모델의 경우
            return m.complete(prompt, temperature=temperature)
