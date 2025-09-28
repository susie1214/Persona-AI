# core/llm_placeholder.py
# (에이닷/믿음 등 외부 HTTP 모델을 여기에 붙이면 됩니다)
import requests
from typing import Dict, Optional
from .llm_base import LLM

class HttpLLM(LLM):
    """
    예시용 Placeholder.
    실제 제공사의 API 스펙에 맞게 payload/headers/응답 파싱을 구현하세요.
    """
    def __init__(self, name: str, endpoint: str, headers: Optional[Dict[str, str]] = None):
        self.name = name
        self.endpoint = endpoint
        self.headers = headers or {}

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        # 예시: {"prompt": "...", "temperature": 0.0}
        # 실제 스펙에 맞게 수정하세요.
        payload = {"prompt": prompt, "temperature": temperature}
        try:
            r = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            # 예시: data["text"] 나 data["choices"][0]["message"]["content"] 등으로 바꿔주세요.
            return (data.get("text") or data.get("response") or "").strip()
        except Exception as e:
            return f"[HTTP LLM ERROR] {e}"

