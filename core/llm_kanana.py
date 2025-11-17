# -*- coding: utf-8 -*-
# core/llm_kanana.py

from __future__ import annotations
import os
from typing import List, Dict, Optional, Any, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

class KananaLLM:
    """
    kakaocorp/kanana-1.5-v-3b-instruct (로컬/허깅페이스 경로)용 간단 LLM 어댑터

    - model_id_or_path: 로컬 디렉터리 경로 또는 HF 모델 ID
    - load_4bit: bitsandbytes 4bit 로드 (선택)
    - device: 'cuda' 또는 'cpu' (기본 자동)
    """

    def __init__(
        self,
        model_id_or_path: str,
        load_4bit: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # 로드 옵션
        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
        )

        if self.device == "cuda":
            model_kwargs["torch_dtype"] = self.dtype
            if load_4bit:
                # bitsandbytes 4bit
                try:
                    import bitsandbytes as bnb  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "bitsandbytes가 설치되지 않았습니다. `pip install bitsandbytes` 후 다시 시도하세요."
                    ) from e

                model_kwargs.update(
                    dict(
                        device_map="auto",
                        load_in_4bit=True,
                        quantization_config=dict(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        ),
                    )
                )
            else:
                model_kwargs.update(device_map="auto")
        else:
            # CPU
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = {"": "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            **model_kwargs,
        )
        self.model.eval()

    # --- 프롬프트 구성 유틸 ---
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        HF chat_template이 등록되어 있으면 사용.
        없으면 간단한 시스템/유저/어시스턴트 포맷으로 직렬화.
        """
        if hasattr(self.tokenizer, "apply_chat_template") and callable(getattr(self.tokenizer, "apply_chat_template")):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # fallback: 매우 단순한 텍스트 변환
        text = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                text.append(f"[SYSTEM]\n{content}\n")
            elif role == "assistant":
                text.append(f"[ASSISTANT]\n{content}\n")
            else:
                text.append(f"[USER]\n{content}\n")
        text.append("[ASSISTANT]\n")
        return "\n".join(text)

    # --- 단발 infer ---
    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        messages: [{"role":"system/user/assistant","content":"..."}] 형식
        """
        prompt = self._apply_chat_template(messages)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.device == "cuda":
            input_ids = input_ids.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 생성 부분만 잘라내기
        generated = text[len(prompt):] if text.startswith(prompt) else text

        if stop:
            for s in stop:
                idx = generated.find(s)
                if idx != -1:
                    generated = generated[:idx]
                    break
        return generated.strip()

    # --- 단순 프롬프트 ---
    def generate(self, prompt: str, **gen_kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **gen_kwargs)
