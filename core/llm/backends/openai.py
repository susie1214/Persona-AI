import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()


class OpenAILLM():
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        self.name = f"openai:{model}"
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt, temperature=0.0):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return r.choices[0].message.content.strip()
