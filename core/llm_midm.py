# core/llm_midm.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MidmLLM():
    def __init__(self, model="K-intelligence/Midm-2.0-Mini-Instruct"):
        self.name = f"midm:{model}"
        
        print(f"[DEBUG] using CUDA : {torch.cuda.is_available()}")
        
        # Load from local models directory
        local_model_path = os.path.join("models", "K-intelligence_Midm-2.0-Mini-Instruct")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def complete(self, prompt, temperature=0.7, max_length=512):
        # 모델의 실제 device 자동 감지 (device_map="auto" 사용 시 필수)
        device = next(self.model.parameters()).device
        print(f"[DEBUG] Midm model running on device: {device}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)  # None 추가로 키가 없어도 오류 방지
        
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_length=max_length
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(torch.cuda.is_available())
