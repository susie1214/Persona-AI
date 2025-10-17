from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class AXLLM():
    """
    (예시) HF에 공개된 모델 아이디가 실제로 있어야 동작합니다.
    사용 전 model ID 확인: 예) "skt/A.X-4.0" (실제 공개 여부는 별도 확인 필요)
    """
    def __init__(self, model: str = "skt/A.X-4.0"):
        self.name = f"ax:{model}"
        
        # Load from local models directory
        local_model_path = os.path.join("models", "A.X-4.0-Light")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def complete(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        print(f"[DEBUG] device : {self.model.device}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
