# -*- coding: utf-8 -*-
# core/llm_midm.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# -----------------------------------------------------------------------------
# config.py 위치 보장(+ 폴백)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트 = core 상위
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import LOCAL_MODELS_DIR  # type: ignore
except Exception:
    LOCAL_MODELS_DIR = ROOT / "models"  # 폴백 디렉터리

from .llm_base import LLM  # 프로젝트 기본 인터페이스


# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def _safe_model_id(model_id: str) -> str:
    """허브 표기 흔들림(대소문자/하이픈)에 안전하게."""
    # 정확 표기: K-Intelligence/Midm-2.0-Mini-Instruct
    low = model_id.lower()
    if "k-intelligence/midm-2.0-mini-instruct" in low or "kintelligence/midm-2.0-mini-instruct" in low:
        return "K-Intelligence/Midm-2.0-Mini-Instruct"
    return model_id


def _pick_dtype() -> "torch.dtype | None":
    """환경에 맞는 dtype 선택: GPU->fp16, CPU->bfloat16(지원 시), 아니면 auto."""
    if torch.cuda.is_available():
        return torch.float16
    if hasattr(torch, "bfloat16"):
        return torch.bfloat16
    return None


# -----------------------------------------------------------------------------
# 본체
# -----------------------------------------------------------------------------
class MidmLLM(LLM):
    """
    KT Midm 2.0 (K-Intelligence/Midm-2.0-Mini-Instruct) 래퍼
    - prompt 또는 messages(list[{role, content}]) 입력 지원
    - stop 문자열 리스트 지원(후처리로 절단)
    - 스트리밍(on_chunk 토큰 콜백) 지원
    - dtype 최신 인자 사용, token_type_ids 경고 제거, pad 토큰 보정
    """

    def __init__(
        self,
        model: str = "K-Intelligence/Midm-2.0-Mini-Instruct",
        *,
        trust_remote_code: bool = False,
    ):
        self.model_id = _safe_model_id(model)
        self.name = f"midm:{self.model_id}"

        # HF 토큰(공개 모델이라 없어도 OK)
        hf_token = (os.getenv("HF_TOKEN") or "").strip() or None

        # 로컬 캐시 디렉터리
        local_model_dir = Path(LOCAL_MODELS_DIR) / self.model_id.replace("/", "_")
        local_model_dir.parent.mkdir(parents=True, exist_ok=True)

        if (local_model_dir / "config.json").exists():
            print(f"[INFO] Loading Midm locally: {local_model_dir}")
            model_path = str(local_model_dir)
            token_kwargs = {}
            model_kwargs = {}
        else:
            print(f"[INFO] Downloading Midm from HF: {self.model_id}")
            model_path = self.model_id
            token_kwargs = {"token": hf_token} if hf_token else {}
            model_kwargs = {"token": hf_token} if hf_token else {}

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            **token_kwargs,
        )

        # model (dtype= 사용)
        dtype = _pick_dtype()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=(dtype if dtype is not None else "auto"),  # ✅ torch_dtype → dtype
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        self.model.eval()

        # pad 토큰 보정(없으면 eos로 대체)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 최초 다운로드였다면 캐시 저장
        if model_path == self.model_id and not (local_model_dir / "config.json").exists():
            print(f"[INFO] Caching Midm to: {local_model_dir}")
            self.tokenizer.save_pretrained(local_model_dir)
            self.model.save_pretrained(local_model_dir)

        print(f"[INFO] Midm model loaded: {self.name}")

        # 기본 생성 하이퍼파라미터
        self.default_temperature = 0.7
        self.default_top_p = 0.95
        self.default_repetition_penalty = 1.1
        self.default_max_new_tokens = 512

        # 기본 stop
        self.default_stop: List[str] = []
        if self.tokenizer.eos_token:
            self.default_stop.append(self.tokenizer.eos_token)
        self.default_stop.append("</s>")

    # ------------------------ 내부 유틸 ------------------------
    def _apply_chat_template(self, system: Optional[str], user: str) -> Dict[str, torch.Tensor]:
        """토크나이저 chat_template 사용(있으면), 아니면 심플 포맷."""
        if getattr(self.tokenizer, "chat_template", None):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            sys_part = f"[SYSTEM]\n{system}\n\n" if system else ""
            text = f"{sys_part}[USER]\n{user}\n\n[ASSISTANT]\n"

        inputs = self.tokenizer(text, return_tensors="pt")

        # LLaMA 계열은 token_type_ids 미사용 → 경고 제거
        inputs.pop("token_type_ids", None)

        # 디바이스 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def _messages_to_prompts(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], str]:
        """messages → (system, user_text) 변환(간단 누적)."""
        system = None
        acc: List[str] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "system" and system is None:
                system = content
            elif role in ("user", "assistant"):
                acc.append(f"[{role.upper()}]\n{content}")
        user_text = "\n".join(acc).strip() or "안내문이 비어있습니다."
        return system, user_text

    def _truncate_at_stop(self, text: str, stop: List[str]) -> str:
        """stop 문자열들 중 가장 앞에서 잘라내기."""
        cut = len(text)
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    # ------------------------ 공개 인터페이스 ------------------------
    def complete(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        on_chunk=None,
    ) -> str:
        """
        공통 인터페이스:
        - prompt 또는 messages 중 하나 사용
        - stream=True면 on_chunk(token) 콜백으로 토큰 전송
        - return: 최종 텍스트
        """
        # 입력 정규화
        system = None
        if messages is not None:
            system, prompt = self._messages_to_prompts(messages)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("complete(): 'prompt' 또는 'messages'가 필요합니다.")

        stop = stop or self.default_stop
        inputs = self._apply_chat_template(system, prompt)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            do_sample=(temperature if temperature is not None else self.default_temperature) > 0,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_p=top_p if top_p is not None else self.default_top_p,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.default_repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        try:
            if stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs["streamer"] = streamer

                with torch.no_grad():
                    self.model.generate(**inputs, **gen_kwargs)

                acc: List[str] = []
                for piece in streamer:
                    acc.append(piece)
                    if callable(on_chunk):
                        try:
                            on_chunk(piece)
                        except Exception:
                            pass
                text = "".join(acc)
            else:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            text = self._truncate_at_stop(text, stop).strip()
            return text

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """객체 소멸 시 GPU 메모리 정리."""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ------------------------ 단독 실행 테스트 ------------------------
if __name__ == "__main__":
    llm = MidmLLM()
    out = llm.complete(
        prompt="안녕하세요! Midm 2.0 통합 테스트 중입니다. 한 문단으로 정중하게 답해주세요.",
        stop=["</s>"],
        max_new_tokens=200,
    )
    print("\n=== OUTPUT ===\n", out)
