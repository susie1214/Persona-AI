from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class AXLLM():
    def __init__(self, model="skt/A.X-4.0"):
        
        print(f"[DEBUG] using CUDA : {torch.cuda.is_available()}")
        
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
            dtype=torch.float16,
            device_map="auto"
        )
        
    def complete(self, prompt, temperature=0.7, max_length=512):
        print(f"[DEBUG] device : {self.model.device}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_length=max_length
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
