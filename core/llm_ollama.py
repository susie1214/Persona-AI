# core/llm_ollama.py
import requests
from .llm_base import LLM

class OllamaLLM(LLM):
    def __init__(self, model: str = "llama3", endpoint: str = "http://localhost:11434"):
        self.name = f"ollama:{model}"
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
