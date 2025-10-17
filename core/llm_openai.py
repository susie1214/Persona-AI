import os
from openai import OpenAI
from .llm_base import LLM
from dotenv import load_dotenv
import requests

load_dotenv()

# class OpenAILLM(LLM):
#     def __init__(self, model="gpt-4o-mini"):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise RuntimeError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
#         self.name = f"openai:{model}"
#         self.client = OpenAI(api_key=api_key)
#         self.model = model

#     def complete(self, prompt, temperature=0.0):
#         r = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature
#         )
#         return r.choices[0].message.content.strip()
# -*- coding: utf-8 -*-
class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o-mini", endpoint: str = "https://api.openai.com/v1/chat/completions"):
        self.name = f"openai:{model}"
        self.model = model
        self.endpoint = endpoint

    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
