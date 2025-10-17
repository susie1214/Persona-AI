# core/llm_ollama.py
import ollama

class OllamaLLM():
    def __init__(self, model: str = "llama3"):
        self.name = f"ollama:{model}"
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Use ollama Python library to generate completions with llama3
        """
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
            }
        )
        return (response.get("response") or "").strip()
