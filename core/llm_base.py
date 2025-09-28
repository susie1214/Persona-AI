from abc import ABC, abstractmethod


class LLM(ABC):
    name: str

    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> str: ...
