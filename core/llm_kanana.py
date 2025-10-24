# core/llm_kanana.py
import os
from threading import local
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class KananaLLM():
    """
    Kakao Kanana-1.5-v-3b-instruct 모델 with 4-bit 양자화 지원
    bitsandbytes를 사용한 메모리 효율적 추론
    """
    def __init__(self, model="kakaocorp/kanana-1.5-2.1b-instruct", use_4bit: bool = True):
        self.name = f"kanana:{model}"
        self.use_4bit = use_4bit

        print(f"[INFO] Loading Kanana-1.5-2.1b model (4-bit: {use_4bit}, CUDA: {torch.cuda.is_available()})")

        # Load from local models directory
        local_model_path = os.path.join("models", "kanana-1.5-2.1b-instruct")
        
        print(f"[DEBUG] llm_kanana path : {local_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True,
            # trust_remote_code=True  # Kakao 모델은 custom code 필요할 수 있음
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
            # "trust_remote_code": True,  # Kakao 모델용
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

        print(f"[INFO] Kanana-1.5-v-3b 로드 완료 (디바이스: {next(self.model.parameters()).device})")

    def complete(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 1024) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            temperature: 샘플링 온도 (0.0-1.0)
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            생성된 텍스트
        """
        # 모델의 실제 device 자동 감지 (device_map="auto" 사용 시 필수)
        device = next(self.model.parameters()).device
        print(f"[DEBUG] Kanana model running on device: {device}")

        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # token_type_ids 제거 (일부 모델에서 불필요)
        inputs.pop("token_type_ids", None)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text


if __name__ == "__main__":
    """테스트 코드"""
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    try:
        # 양자화 버전 테스트
        llm = KananaLLM(use_4bit=True)
        prompt = "안녕하세요, 오늘 날씨가 좋네요."
        response = llm.complete(prompt, max_new_tokens=50)
        print(f"\n입력: {prompt}")
        print(f"출력: {response}")
    except Exception as e:
        print(f"테스트 실패: {e}")
