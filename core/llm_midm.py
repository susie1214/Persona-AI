from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .llm_base import LLM

class MidmLLM(LLM):
    def __init__(self, model="K-intelligence/Midm-2.0-Mini-Instruct"):
        self.name = f"midm:{model}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def complete(self, prompt, temperature=0.7, max_length=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_length=max_length
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
