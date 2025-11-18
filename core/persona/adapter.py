# -*- coding: utf-8 -*-
# core/persona/adapter.py
"""
QLoRA 어댑터 관리 및 추론 모듈
- 베이스 모델 + Speaker별 LoRA 어댑터 동적 로딩
- 디지털 페르소나 시스템에서 공통으로 사용하는 LLM 래퍼
"""

import os
import json
from typing import Optional, Dict

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class AdapterManager:
    """
    QLoRA 어댑터 관리자
    - 베이스 모델 로드
    - Speaker별 어댑터 동적 로딩/언로드
    - 간단한 Chat/Generation 인터페이스 제공
    """

    def __init__(self, use_4bit: bool = True):
        self.available = PEFT_AVAILABLE
        self.use_4bit = use_4bit

        self.base_model_id: Optional[str] = None
        self.base_model = None
        self.tokenizer = None

        # speaker_id -> PeftModel
        self.loaded_adapters: Dict[str, PeftModel] = {}
        # speaker_id -> metadata(dict)
        self.adapter_metadata: Dict[str, Dict] = {}
        # 현재 활성 speaker_id (None이면 베이스 모델만 사용)
        self.active_adapter: Optional[str] = None

        print(f"[INFO] AdapterManager initialized (PEFT available: {self.available})")

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    def load_base(self, base_model_id: str) -> bool:
        """
        베이스 모델 로드

        Args:
            base_model_id: HuggingFace 모델 ID 또는 로컬 경로

        Returns:
            bool: 성공 여부
        """
        if not self.available:
            print("[WARN] PEFT not available (transformers/peft import 실패)")
            return False

        try:
            print(f"[INFO] Loading base model: {base_model_id}")

            # 4-bit 양자화 설정
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None

            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 베이스 모델
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config if self.use_4bit else None,
                device_map="auto",
                trust_remote_code=True,
            )

            self.base_model_id = base_model_id
            print("[INFO] Base model loaded successfully")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load base model: {e}")
            return False

    def load_adapter(self, speaker_id: str, adapter_path: str) -> bool:
        """
        특정 speaker용 QLoRA 어댑터 로드

        Args:
            speaker_id: 화자 ID (예: "speaker_01")
            adapter_path: 어댑터 디렉터리 경로

        Returns:
            bool: 성공 여부
        """
        if not self.available or self.base_model is None:
            print("[WARN] Base model not loaded or PEFT unavailable")
            return False

        try:
            print(f"[INFO] Loading adapter for {speaker_id} from {adapter_path}")

            model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.loaded_adapters[speaker_id] = model

            # metadata.json optional
            metadata_path = os.path.join(adapter_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.adapter_metadata[speaker_id] = json.load(f)
            else:
                self.adapter_metadata[speaker_id] = {"speaker_id": speaker_id}

            print(f"[INFO] Adapter loaded for {speaker_id}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load adapter for {speaker_id}: {e}")
            return False

    def load_all_adapters(self, adapters_root: str = "adapters") -> int:
        """
        adapters_root 안의 speaker 디렉터리를 훑어서,
        metadata만 미리 인덱싱하는 함수 (실제 LoRA 로드는 set_active 시점)

        디렉터리 구조 예:
            adapters/
              speaker_01/final/adapter_model.bin
              speaker_02/final/...

        Returns:
            int: 발견한 어댑터 수
        """
        if not os.path.exists(adapters_root):
            print(f"[WARN] Adapters directory not found: {adapters_root}")
            return 0

        count = 0
        for speaker_id in os.listdir(adapters_root):
            adapter_path = os.path.join(adapters_root, speaker_id, "final")
            if not os.path.isdir(adapter_path):
                continue

            metadata_path = os.path.join(adapter_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self.adapter_metadata[speaker_id] = json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load metadata for {speaker_id}: {e}")
                    self.adapter_metadata[speaker_id] = {"speaker_id": speaker_id}
            else:
                self.adapter_metadata[speaker_id] = {"speaker_id": speaker_id}

            count += 1

        print(f"[INFO] Found {count} available adapters (lazy loading)")
        return count

    # ------------------------------------------------------------------
    # 어댑터 활성/언로드
    # ------------------------------------------------------------------
    def unload_adapter(self, speaker_id: str):
        """
        특정 어댑터 언로드 (GPU 메모리 해제)

        Args:
            speaker_id: 언로드할 화자 ID
        """
        if speaker_id not in self.loaded_adapters:
            return

        try:
            del self.loaded_adapters[speaker_id]
            torch.cuda.empty_cache()
            print(f"[INFO] Unloaded adapter: {speaker_id}")
        except Exception as e:
            print(f"[WARN] Failed to unload adapter {speaker_id}: {e}")

    def set_active(self, speaker_id: Optional[str]):
        """
        활성 어댑터 설정

        Args:
            speaker_id: 화자 ID (None이면 베이스 모델만 사용)
        """
        # 다른 speaker가 이미 활성화되어 있으면 언로드
        if self.active_adapter and self.active_adapter != speaker_id:
            self.unload_adapter(self.active_adapter)

        if speaker_id is None:
            self.active_adapter = None
            print("[INFO] Active adapter set to: base model")
            return

        # 메모리에 없으면 디스크에서 로드
        if speaker_id not in self.loaded_adapters:
            adapter_path = os.path.join("adapters", speaker_id, "final")
            if not os.path.exists(adapter_path):
                print(f"[WARN] Adapter path not found: {adapter_path}")
                self.active_adapter = None
                return

            if not self.load_adapter(speaker_id, adapter_path):
                print(f"[WARN] Failed to load adapter for {speaker_id}")
                self.active_adapter = None
                return

        self.active_adapter = speaker_id
        print(f"[INFO] Active adapter set to: {speaker_id}")

    # ------------------------------------------------------------------
    # 생성 API
    # ------------------------------------------------------------------
    def _get_current_model(self):
        """현재 활성 모델(어댑터 또는 베이스)을 반환"""
        if (
            self.active_adapter
            and self.active_adapter in self.loaded_adapters
        ):
            return self.loaded_adapters[self.active_adapter]
        return self.base_model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        단일 prompt에 대한 텍스트 생성

        Returns:
            str: 생성된 텍스트
        """
        if not self.available or self.base_model is None or self.tokenizer is None:
            return "[ERROR] Model or tokenizer not loaded"

        model = self._get_current_model()
        tokenizer = self.tokenizer

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in text:
                text = text.replace(prompt, "").strip()
            return text

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return f"[ERROR] {e}"

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        system_prompt + user_message 조합으로 한 번의 답변 생성
        (디지털 페르소나와 연결할 때 주로 사용)

        system_prompt: DigitalPersonaManager.build_combined_system_prompt(...) 결과
        """
        prompt = (
            f"{system_prompt}\n\n"
            f"[사용자]\n{user_message}\n\n"
            f"[어시스턴트]\n"
        )
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # 레거시 호환
    def respond(self, prompt: str) -> str:
        return self.generate(prompt)


# ----------------------------------------------------------------------
# 독립 실행 테스트용
# ----------------------------------------------------------------------
def test_adapter():
    manager = AdapterManager(use_4bit=True)

    if not manager.load_base("Qwen/Qwen2.5-3B-Instruct"):
        print("베이스 모델 로드 실패")
        return

    manager.load_all_adapters("adapters")

    query = "데이터베이스 성능 최적화 방법을 간단히 설명해줘."

    print("\n[Base Model]")
    manager.set_active(None)
    print(manager.generate(query, max_new_tokens=100))

    for sid in manager.adapter_metadata.keys():
        print(f"\n[{sid}]")
        manager.set_active(sid)
        print(manager.generate(query, max_new_tokens=100))


if __name__ == "__main__":
    test_adapter()
