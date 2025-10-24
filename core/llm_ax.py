from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

class AXLLM():
    """
    A.X-4.0 모델 with 4-bit 양자화 지원
    bitsandbytes를 사용한 메모리 효율적 추론
    """
    def __init__(self, model: str = "skt/A.X-4.0", use_4bit: bool = True):
        self.name = f"ax:{model}"
        self.use_4bit = use_4bit

        # Load from local models directory
        local_model_path = os.path.join("models", "A.X-4.0-Light")

        print(f"[INFO] Loading A.X-4.0 model (4-bit: {use_4bit})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True
        )

        # 4-bit 양자화 설정
        quantization_config = None
        if use_4bit and torch.cuda.is_available():
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 (권장)
                    bnb_4bit_compute_dtype=torch.float16,  # 계산 시 float16 사용
                    bnb_4bit_use_double_quant=True,  # 더블 양자화로 메모리 추가 절약
                )
                print("[INFO] 4-bit 양자화 활성화 (메모리 ~75% 절약)")
            except Exception as e:
                print(f"[WARN] 4-bit 양자화 실패, FP16으로 폴백: {e}")
                quantization_config = None

        # 모델 로드
        load_kwargs = {
            "local_files_only": True,
            "device_map": "auto",
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # 양자화 없이 로드 시 dtype 설정
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            **load_kwargs
        )

        print(f"[INFO] A.X-4.0 로드 완료 (디바이스: {next(self.model.parameters()).device})")

    def complete(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        # 모델의 실제 device 자동 감지 (device_map="auto" 사용 시 필수)
        device = next(self.model.parameters()).device
        print(f"[DEBUG] A.X model running on device: {device}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
