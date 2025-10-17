# # core/llm_ollama.py
# import requests
# from .llm_base import LLM

# class OllamaLLM(LLM):
#     def __init__(self, model: str = "llama3", endpoint: str = "http://localhost:11434"):
#         self.name = f"ollama:{model}"
#         self.model = model
#         self.endpoint = endpoint.rstrip("/")

#     def complete(self, prompt: str, temperature: float = 0.0) -> str:
#         url = f"{self.endpoint}/api/generate"
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "options": {"temperature": temperature},
#         }
#         r = requests.post(url, json=payload, timeout=120)
#         r.raise_for_status()
#         data = r.json()
#         return (data.get("response") or "").strip()

# -*- coding: utf-8 -*-
import json
import requests
from .llm_base import LLM

class OllamaLLM(LLM):
    """
    Ollama REST API. 기본은 /api/generate 사용, 404면 /api/chat으로 폴백.
    """
    def __init__(self, model: str = "llama3", endpoint: str = "http://localhost:11434", stream: bool = False):
        self.name = f"ollama:{model}"
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.stream = stream

    def _post_json(self, path: str, payload: dict):
        url = f"{self.endpoint}{path}"
        r = requests.post(url, json=payload, timeout=120, stream=self.stream)
        if r.status_code == 404 and path == "/api/generate":
            # 일부 배포는 /api/chat만 활성화되는 경우가 있음 → 폴백
            chat_payload = {
                "model": payload["model"],
                "messages": [{"role": "user", "content": payload.get("prompt", "")}],
                "options": payload.get("options", {}),
                "stream": self.stream,
            }
            r = requests.post(f"{self.endpoint}/api/chat", json=chat_payload, timeout=120, stream=self.stream)
        r.raise_for_status()
        return r

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": self.stream,
        }
        r = self._post_json("/api/generate", payload)

        # 스트리밍이면 라인별 JSON을 모아 합침
        if self.stream:
            chunks = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(obj.get("response") or obj.get("message", {}).get("content", ""))
                except Exception:
                    pass
            return "".join(chunks).strip()

        data = r.json()
        return (data.get("response") or data.get("message", {}).get("content", "")).strip()

