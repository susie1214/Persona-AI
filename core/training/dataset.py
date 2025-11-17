# -*- coding: utf-8 -*-
# core/persona_training.py
"""
QLoRA 디지털 페르소나 학습을 위한 데이터셋 생성 모듈
Speaker별 발언을 instruction-following 포맷으로 변환
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import random


class PersonaDatasetGenerator:
    """
    Speaker별 발언 데이터를 QLoRA 학습용 데이터셋으로 변환
    """

    def __init__(self, output_dir: str = "data/persona_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 대화 상황 템플릿
        self.context_templates = [
            "다음 상황에서 {speaker_name}처럼 답변하세요",
            "{speaker_name}의 말투로 다음 질문에 답변하세요",
            "{speaker_name}이라면 다음 상황에서 어떻게 말할까요?",
            "아래 내용에 대해 {speaker_name} 스타일로 설명하세요",
        ]

        # 질문 생성 템플릿
        self.question_templates = [
            "{topic}에 대해 설명해주세요",
            "{topic} 관련해서 어떻게 생각하시나요?",
            "{topic}의 문제점과 해결 방안은 무엇인가요?",
            "{topic}를 개선하려면 어떻게 해야 할까요?",
        ]

    def extract_topics_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 주요 토픽 추출 (간단한 키워드 기반)
        """
        keywords = [
            "데이터베이스", "성능", "최적화", "캐시", "인덱스",
            "프론트엔드", "번들", "코드", "스플리팅", "React",
            "아키텍처", "마이크로서비스", "CI/CD", "파이프라인",
            "API", "서버", "클라이언트", "보안", "테스트",
            "배포", "모니터링", "로깅", "에러", "버그"
        ]

        topics = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                topics.append(keyword)

        return topics if topics else ["일반적인 주제"]

    def create_training_pair(
        self,
        speaker_name: str,
        utterance: str,
        context_mode: str = "direct"
    ) -> Dict:
        """
        단일 발언에서 학습 데이터 페어 생성

        Args:
            speaker_name: 화자 이름
            utterance: 발언 텍스트
            context_mode: 'direct' (직접 발언) 또는 'qa' (질문-답변)

        Returns:
            {instruction, input, output} 형식의 dict
        """
        if context_mode == "direct":
            # 직접 발언 모드: 상황 설명 + 발언
            instruction = random.choice(self.context_templates).format(speaker_name=speaker_name)
            topics = self.extract_topics_from_text(utterance)
            input_text = f"{topics[0]} 관련 의견을 말씀해주세요" if topics else "의견을 말씀해주세요"

            return {
                "instruction": instruction,
                "input": input_text,
                "output": utterance
            }

        elif context_mode == "qa":
            # 질문-답변 모드: 발언을 답변으로 사용
            instruction = f"{speaker_name}의 말투로 답변하세요"
            topics = self.extract_topics_from_text(utterance)
            topic = topics[0] if topics else "해당 주제"
            input_text = random.choice(self.question_templates).format(topic=topic)

            return {
                "instruction": instruction,
                "input": input_text,
                "output": utterance
            }

        else:
            raise ValueError(f"Unknown context_mode: {context_mode}")

    def create_conversation_pair(
        self,
        speaker_name: str,
        utterances: List[str],
        max_context: int = 3
    ) -> Dict:
        """
        연속된 발언을 대화 컨텍스트로 활용

        Args:
            speaker_name: 화자 이름
            utterances: 연속된 발언 리스트
            max_context: 최대 컨텍스트 길이

        Returns:
            {instruction, input, output} 형식의 dict
        """
        if len(utterances) < 2:
            return self.create_training_pair(speaker_name, utterances[0])

        # 마지막 발언을 답변으로, 이전 발언들을 컨텍스트로
        context = utterances[-max_context-1:-1]
        answer = utterances[-1]

        instruction = f"{speaker_name}의 말투로 이어서 답변하세요"
        input_text = "이전 대화:\n" + "\n".join([f"- {u}" for u in context])

        return {
            "instruction": instruction,
            "input": input_text,
            "output": answer
        }

    def generate_dataset_from_speaker(
        self,
        speaker_id: str,
        speaker_name: str,
        utterances: List[Dict],
        min_utterances: int = 20
    ) -> List[Dict]:
        """
        Speaker의 발언 리스트에서 학습 데이터셋 생성

        Args:
            speaker_id: 화자 ID
            speaker_name: 화자 이름
            utterances: 발언 리스트 [{text, timestamp, meeting_id}, ...]
            min_utterances: 최소 발언 수

        Returns:
            학습 데이터 리스트
        """
        if len(utterances) < min_utterances:
            print(f"[WARN] {speaker_name}의 발언이 {len(utterances)}개로 부족합니다 (최소 {min_utterances}개 필요)")
            return []

        dataset = []

        # 1. 개별 발언 기반 데이터 생성 (70%)
        for utt in utterances:
            text = utt.get("text", "")
            if not text or len(text) < 10:
                continue

            # Direct mode
            if random.random() < 0.5:
                pair = self.create_training_pair(speaker_name, text, context_mode="direct")
            else:
                # QA mode
                pair = self.create_training_pair(speaker_name, text, context_mode="qa")

            pair["speaker_id"] = speaker_id
            pair["speaker_name"] = speaker_name
            dataset.append(pair)

        # 2. 대화 흐름 기반 데이터 생성 (30%)
        texts = [u["text"] for u in utterances if u.get("text")]
        for i in range(2, len(texts)):
            if random.random() < 0.3:  # 30% 확률로 생성
                context_utterances = texts[max(0, i-3):i+1]
                pair = self.create_conversation_pair(speaker_name, context_utterances)
                pair["speaker_id"] = speaker_id
                pair["speaker_name"] = speaker_name
                dataset.append(pair)

        print(f"[INFO] {speaker_name} 데이터셋 생성: {len(dataset)}개 페어")
        return dataset

    def save_dataset(
        self,
        dataset: List[Dict],
        speaker_id: str,
        format: str = "jsonl"
    ) -> str:
        """
        데이터셋 저장

        Args:
            dataset: 학습 데이터 리스트
            speaker_id: 화자 ID
            format: 저장 포맷 ('jsonl' 또는 'json')

        Returns:
            저장된 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{speaker_id}_dataset_{timestamp}.{format}"
        filepath = os.path.join(self.output_dir, filename)

        if format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:  # json
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 데이터셋 저장: {filepath}")
        return filepath

    def generate_dataset_from_rag(
        self,
        rag_store,
        speaker_id: str,
        speaker_name: Optional[str] = None,
        min_utterances: int = 20
    ) -> Optional[str]:
        """
        RAG Store에서 직접 화자 데이터 추출 및 데이터셋 생성

        Args:
            rag_store: RagStore 인스턴스
            speaker_id: 화자 ID
            speaker_name: 화자 이름 (선택, 없으면 ID 사용)
            min_utterances: 최소 발언 수

        Returns:
            생성된 데이터셋 파일 경로
        """
        if not rag_store.ok:
            print("[ERROR] RAG store not initialized")
            return None

        # Speaker 발언 가져오기
        results = rag_store.search_by_speaker(speaker_id, query="", topk=1000)

        if len(results) < min_utterances:
            print(f"[WARN] {speaker_id}의 발언이 {len(results)}개로 부족합니다")
            return None

        # 발언을 utterance 형식으로 변환
        utterances = []
        for r in results:
            utterances.append({
                "text": r.get("text", ""),
                "timestamp": r.get("timestamp", ""),
                "meeting_id": None
            })

        # 화자 이름
        if not speaker_name and results:
            speaker_name = results[0].get("speaker_name", speaker_id)

        # 데이터셋 생성
        dataset = self.generate_dataset_from_speaker(
            speaker_id, speaker_name, utterances, min_utterances
        )

        if not dataset:
            return None

        # 저장
        filepath = self.save_dataset(dataset, speaker_id, format="jsonl")
        return filepath

    def generate_all_datasets(
        self,
        rag_store,
        min_utterances: int = 20
    ) -> Dict[str, str]:
        """
        RAG Store에 있는 모든 화자의 데이터셋 생성

        Args:
            rag_store: RagStore 인스턴스
            min_utterances: 최소 발언 수

        Returns:
            {speaker_id: filepath} 딕셔너리
        """
        if not rag_store.ok:
            print("[ERROR] RAG store not initialized")
            return {}

        speakers = rag_store.get_all_speakers()
        print(f"[INFO] 총 {len(speakers)}명의 화자 발견")

        datasets = {}
        for speaker_id in speakers:
            print(f"\n[INFO] {speaker_id} 데이터셋 생성 중...")
            filepath = self.generate_dataset_from_rag(rag_store, speaker_id, min_utterances=min_utterances)
            if filepath:
                datasets[speaker_id] = filepath

        print(f"\n[INFO] 총 {len(datasets)}개 데이터셋 생성 완료")
        return datasets
