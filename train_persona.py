#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA 디지털 페르소나 학습 스크립트
Speaker별 말투를 학습하는 QLoRA 어댑터 생성
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
import argparse

# Transformers & PEFT
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    from datasets import Dataset
    TRAIN_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Required packages not installed: {e}")
    print("Install with: pip install transformers peft datasets accelerate bitsandbytes")
    TRAIN_AVAILABLE = False


@dataclass
class PersonaTrainingConfig:
    """QLoRA 학습 설정"""
    # 모델
    base_model: str = "models/kanana-1.5-2.1b-instruct"  # 로컬 Kanana 모델

    # QLoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # 학습 설정
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 512

    # 출력
    output_dir: str = "adapters"
    logging_steps: int = 10
    save_steps: int = 50

    # 하드웨어
    use_4bit: bool = True
    use_fp16: bool = True


class PersonaTrainer:
    """디지털 페르소나 학습 클래스"""

    def __init__(self, config: PersonaTrainingConfig):
        if not TRAIN_AVAILABLE:
            raise RuntimeError("Training dependencies not available")

        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """베이스 모델 로드 (4-bit 양자화)"""
        print(f"[INFO] Loading base model: {self.config.base_model}")

        # 4-bit 양자화 설정
        if self.config.use_4bit:
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
            self.config.base_model,
            trust_remote_code=True
        )

        # padding token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config if self.config.use_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )

        # K-bit 학습 준비
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        print("[INFO] Model loaded successfully")

    def setup_lora(self):
        """LoRA 어댑터 설정"""
        print("[INFO] Setting up LoRA adapter")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self, dataset_path: str) -> Dataset:
        """데이터셋 로드"""
        print(f"[INFO] Loading dataset: {dataset_path}")

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        print(f"[INFO] Loaded {len(data)} training examples")
        return Dataset.from_list(data)

    def preprocess_function(self, examples):
        """데이터 전처리 (instruction-following 포맷)"""
        prompts = []

        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            output = examples["output"][i]

            # Prompt 템플릿
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

            prompts.append(prompt)

        # 토크나이징
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
        )

        # Labels 설정 (입력과 동일, loss는 response 부분만 계산됨)
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def train(
        self,
        dataset_path: str,
        speaker_id: str,
        speaker_name: str = None
    ) -> str:
        """
        학습 실행

        Args:
            dataset_path: 학습 데이터셋 경로 (.jsonl)
            speaker_id: 화자 ID
            speaker_name: 화자 이름 (선택)

        Returns:
            어댑터 저장 경로
        """
        # 모델 로드
        if self.model is None:
            self.load_model()
            self.setup_lora()

        # 데이터셋 로드
        dataset = self.load_dataset(dataset_path)

        # 전처리
        print("[INFO] Preprocessing dataset")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # 학습 설정
        output_dir = os.path.join(self.config.output_dir, speaker_id)
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            fp16=self.config.use_fp16,
            optim="paged_adamw_8bit" if self.config.use_4bit else "adamw_torch",
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="none",  # wandb/tensorboard 사용 안함
        )

        # Trainer 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        # 학습 시작
        print(f"\n[INFO] Starting training for {speaker_id}")
        print(f"[INFO] Output directory: {output_dir}")
        trainer.train()

        # 어댑터 저장
        adapter_path = os.path.join(output_dir, "final")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        # 메타데이터 저장
        metadata = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name or speaker_id,
            "base_model": self.config.base_model,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "num_epochs": self.config.num_epochs,
            "dataset_size": len(dataset),
        }

        with open(os.path.join(adapter_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Training completed!")
        print(f"[INFO] Adapter saved to: {adapter_path}")

        return adapter_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Train QLoRA persona adapter")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset (.jsonl)")
    parser.add_argument("--speaker-id", type=str, required=True, help="Speaker ID")
    parser.add_argument("--speaker-name", type=str, default=None, help="Speaker name (optional)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="adapters", help="Output directory")

    args = parser.parse_args()

    if not TRAIN_AVAILABLE:
        print("[ERROR] Training dependencies not installed")
        return

    # 설정 생성
    config = PersonaTrainingConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # 학습기 생성
    trainer = PersonaTrainer(config)

    # 학습 실행
    adapter_path = trainer.train(
        dataset_path=args.dataset,
        speaker_id=args.speaker_id,
        speaker_name=args.speaker_name,
    )

    print(f"\n✅ Persona adapter created successfully!")
    print(f"   Path: {adapter_path}")


if __name__ == "__main__":
    main()
