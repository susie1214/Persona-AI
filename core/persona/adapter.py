# core/adapter.py
"""
QLoRA 어댑터 관리 및 추론 모듈
Speaker별 디지털 페르소나를 로드하고 말투를 재현
"""

import os
import json
import torch
from typing import Optional, Dict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class AdapterManager:
    """
    QLoRA 어댑터 관리자
    - 베이스 모델 로드
    - Speaker별 어댑터 동적 로딩
    - RAG 컨텍스트 통합 추론
    """

    def __init__(self, use_4bit: bool = True):
        self.available = PEFT_AVAILABLE
        self.use_4bit = use_4bit

        self.base_model_id = None
        self.base_model = None
        self.tokenizer = None

        self.loaded_adapters: Dict[str, PeftModel] = {}
        self.adapter_metadata: Dict[str, Dict] = {}
        self.active_adapter = None

        print(f"[INFO] AdapterManager initialized (PEFT available: {self.available})")

    def load_base(self, base_model_id: str = "models/kanana-1.5-2.1b-instruct") -> bool:
        """
        베이스 모델 로드

        Args:
            base_model_id: HuggingFace 모델 ID

        Returns:
            성공 여부
        """
        if not self.available:
            print("[WARN] PEFT not available")
            return False

        try:
            print(f"[INFO] Loading base model: {base_model_id}")

            # 4-bit 양자화 설정 (메모리 절약)
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 모델 로드
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

    def load_adapter(
        self,
        speaker_id: str,
        adapter_path: str
    ) -> bool:
        """
        Speaker 어댑터 로드

        Args:
            speaker_id: 화자 ID
            adapter_path: 어댑터 디렉터리 경로

        Returns:
            성공 여부
        """
        if not self.available or self.base_model is None:
            print("[WARN] Base model not loaded")
            return False

        try:
            print(f"[INFO] Loading adapter for {speaker_id} from {adapter_path}")

            # 어댑터 로드
            model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.loaded_adapters[speaker_id] = model

            # 메타데이터 로드
            metadata_path = os.path.join(adapter_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.adapter_metadata[speaker_id] = json.load(f)
            else:
                self.adapter_metadata[speaker_id] = {"speaker_id": speaker_id}

            print(f"[INFO] Adapter loaded for {speaker_id}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load adapter: {e}")
            return False

    def load_all_adapters(self, adapters_dir: str = "adapters") -> int:
        """
        사용 가능한 어댑터 목록만 인덱싱 (실제 로드는 필요할 때만)

        Args:
            adapters_dir: 어댑터 루트 디렉터리

        Returns:
            발견된 어댑터 수
        """
        if not os.path.exists(adapters_dir):
            print(f"[WARN] Adapters directory not found: {adapters_dir}")
            return 0

        count = 0
        available_adapters = []

        for speaker_id in os.listdir(adapters_dir):
            adapter_path = os.path.join(adapters_dir, speaker_id, "final")
            if os.path.isdir(adapter_path):
                # 메타데이터 로드 (가벼움)
                metadata_path = os.path.join(adapter_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            self.adapter_metadata[speaker_id] = json.load(f)
                        available_adapters.append(speaker_id)
                        count += 1
                    except Exception as e:
                        print(f"[WARN] Failed to load metadata for {speaker_id}: {e}")
                else:
                    # 메타데이터 없어도 등록
                    self.adapter_metadata[speaker_id] = {"speaker_id": speaker_id}
                    available_adapters.append(speaker_id)
                    count += 1

        print(f"[INFO] Found {count} available adapters (lazy loading enabled)")
        return count

    def set_active(self, speaker_id: Optional[str]):
        """
        활성 어댑터 설정 (메모리 최적화: 필요할 때만 로드)

        Args:
            speaker_id: 화자 ID (None이면 베이스 모델 사용)
        """
        # 기존 활성 어댑터 언로드 (메모리 절약)
        if self.active_adapter and self.active_adapter != speaker_id:
            self.unload_adapter(self.active_adapter)

        if speaker_id:
            # 요청한 어댑터가 메모리에 없으면 로드
            if speaker_id not in self.loaded_adapters:
                adapter_path = os.path.join("adapters", speaker_id, "final")
                if os.path.exists(adapter_path):
                    print(f"[INFO] Loading adapter for {speaker_id} on demand...")
                    if not self.load_adapter(speaker_id, adapter_path):
                        print(f"[WARN] Failed to load adapter for {speaker_id}")
                        self.active_adapter = None
                        return
                else:
                    print(f"[WARN] Adapter not found: {adapter_path}")
                    self.active_adapter = None
                    return

            self.active_adapter = speaker_id
            print(f"[INFO] Active adapter set to: {speaker_id}")
        else:
            self.active_adapter = None
            print(f"[INFO] Active adapter set to: base model")

    def unload_adapter(self, speaker_id: str):
        """
        특정 어댑터 언로드 (메모리 절약)

        Args:
            speaker_id: 언로드할 어댑터의 화자 ID
        """
        if speaker_id in self.loaded_adapters:
            try:
                # 메모리에서 제거
                del self.loaded_adapters[speaker_id]
                if speaker_id in self.adapter_metadata:
                    del self.adapter_metadata[speaker_id]

                # GPU 메모리 정리
                import torch
                torch.cuda.empty_cache()

                print(f"[INFO] Unloaded adapter: {speaker_id}")
            except Exception as e:
                print(f"[WARN] Failed to unload adapter {speaker_id}: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링

        Returns:
            생성된 텍스트
        """
        if not self.available or self.base_model is None or self.tokenizer is None:
            return "[ERROR] Model or tokenizer not loaded"

        # Type narrowing: after the None check above, we know these are not None
        tokenizer = self.tokenizer
        base_model = self.base_model

        # 활성 모델 선택
        if self.active_adapter and self.active_adapter in self.loaded_adapters:
            model = self.loaded_adapters[self.active_adapter]
        else:
            model = base_model

        try:
            # 토크나이징
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 생성
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

            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 프롬프트 제거 후 답변만 반환
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return f"[ERROR] {e}"

    def respond(self, prompt: str) -> str:
        """
        간단한 응답 생성 (하위 호환성)

        Args:
            prompt: 입력 프롬프트

        Returns:
            생성된 응답
        """
        return self.generate(prompt)

    def respond_with_context(
        self,
        query: str,
        rag_context: list,
        speaker_id: Optional[str] = None,
    ) -> str:
        """
        RAG 컨텍스트를 포함한 답변 생성

        Args:
            query: 사용자 질문
            rag_context: RAG 검색 결과 리스트
            speaker_id: 화자 ID (페르소나)

        Returns:
            생성된 답변
        """
        # 활성 어댑터 설정
        if speaker_id:
            self.set_active(speaker_id)

        # Speaker 정보
        speaker_name = "Assistant"
        if speaker_id and speaker_id in self.adapter_metadata:
            speaker_name = self.adapter_metadata[speaker_id].get("speaker_name", speaker_id)

        # 프롬프트 구성
        prompt = f"""### Instruction:
당신은 {speaker_name}입니다. 아래 과거 대화 기록을 참고하여 질문에 답변하세요.
"""

        # RAG 컨텍스트 추가
        if rag_context:
            prompt += "\n### 과거 대화 기록:\n"
            for i, ctx in enumerate(rag_context[:3], 1):
                text = ctx.get("text", "")
                speaker = ctx.get("speaker_name", "Unknown")
                prompt += f"{i}. [{speaker}] {text}\n"

        # 질문 추가
        prompt += f"\n### 질문:\n{query}\n"
        prompt += f"\n### {speaker_name}의 답변:\n"

        # 생성
        return self.generate(prompt)

    def get_loaded_adapters(self) -> list:
        """로드된 어댑터 목록"""
        return list(self.loaded_adapters.keys())

    def get_adapter_info(self, speaker_id: str) -> Optional[Dict]:
        """어댑터 메타데이터 조회"""
        return self.adapter_metadata.get(speaker_id)


# 간단한 테스트 인터페이스
def test_adapter():
    """어댑터 테스트"""
    manager = AdapterManager(use_4bit=True)

    # 베이스 모델 로드
    if not manager.load_base("Qwen/Qwen2.5-3B-Instruct"):
        print("베이스 모델 로드 실패")
        return

    # 어댑터 로드
    manager.load_all_adapters("adapters")

    # 테스트 생성
    test_query = "데이터베이스 성능 최적화 방법을 설명해주세요"

    # 베이스 모델
    print("\n[Base Model]")
    manager.set_active(None)
    response = manager.generate(test_query, max_new_tokens=100)
    print(response)

    # 어댑터 모델
    for speaker_id in manager.get_loaded_adapters():
        print(f"\n[{speaker_id}]")
        manager.set_active(speaker_id)
        response = manager.generate(test_query, max_new_tokens=100)
        print(response)


if __name__ == "__main__":
    test_adapter()
