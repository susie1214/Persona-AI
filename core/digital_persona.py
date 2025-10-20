# -*- coding: utf-8 -*-
# core/digital_persona.py
"""
통합 디지털 페르소나 시스템
- 음성 임베딩 + 발언 이력 + 말투 학습 + 메타데이터 통합
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from core.voice_store import VoiceStore
from core.rag_store import RagStore
from core.persona_store import PersonaStore


@dataclass
class DigitalPersona:
    """
    확장된 디지털 페르소나 모델
    화자의 모든 정보를 통합 관리
    """
    # 기본 식별 정보
    speaker_id: str
    display_name: str

    # 음성 특징
    voice_embedding: Optional[np.ndarray] = None
    embedding_quality: float = 0.0  # 임베딩 신뢰도 (0-1)

    # 역할 및 전문성
    role: str = ""  # 예: "팀장", "개발자", "디자이너"
    department: str = ""
    expertise: List[str] = field(default_factory=list)  # ["AI", "백엔드", "인프라"]

    # 성격 및 소통 스타일
    personality_keywords: List[str] = field(default_factory=list)  # ["꼼꼼한", "직설적", "유머러스"]
    communication_style: Dict[str, Any] = field(default_factory=dict)  # {"tone": "정중", "format": "개조식"}

    # 말투 패턴
    speech_patterns: Dict[str, Any] = field(default_factory=dict)
    # {
    #     "common_phrases": ["그래서 말씀드리자면", "제 생각에는"],
    #     "sentence_endings": ["-습니다", "-죠"],
    #     "vocab_complexity": "medium",
    #     "avg_sentence_length": 15
    # }

    # 학습된 모델
    qlora_adapter_path: Optional[str] = None  # LoRA 어댑터 경로
    llm_backend: str = "openai:gpt-4o-mini"  # 선호 LLM

    # RAG 컬렉션
    personal_collection: Optional[str] = None  # 개인 발언 전용 컬렉션
    utterance_count: int = 0  # 총 발언 수

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_interaction: Optional[str] = None
    interaction_count: int = 0

    # 사전 지식 (manual input)
    prior_knowledge: Dict[str, Any] = field(default_factory=dict)
    # {
    #     "education": "컴퓨터공학 학사",
    #     "career_years": 5,
    #     "projects": ["프로젝트A", "프로젝트B"],
    #     "skills": ["Python", "React"],
    #     "interests": ["ML/AI", "클라우드"]
    # }

    def to_dict(self) -> Dict:
        """직렬화 (numpy array 제외)"""
        d = asdict(self)
        # numpy array는 리스트로 변환
        if self.voice_embedding is not None:
            d['voice_embedding'] = self.voice_embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'DigitalPersona':
        """역직렬화"""
        # voice_embedding을 numpy array로 복원
        if 'voice_embedding' in data and data['voice_embedding'] is not None:
            data['voice_embedding'] = np.array(data['voice_embedding'])
        return cls(**data)

    def generate_system_prompt(self) -> str:
        """페르소나 기반 시스템 프롬프트 생성"""
        parts = []

        # 기본 역할
        parts.append(f"당신은 '{self.display_name}'의 디지털 페르소나입니다.")

        if self.role:
            parts.append(f"역할: {self.role}")

        if self.department:
            parts.append(f"소속: {self.department}")

        # 전문성
        if self.expertise:
            expertise_str = ", ".join(self.expertise)
            parts.append(f"전문 분야: {expertise_str}")

        # 성격
        if self.personality_keywords:
            personality_str = ", ".join(self.personality_keywords)
            parts.append(f"성격 특징: {personality_str}")

        # 말투 스타일
        if self.communication_style:
            tone = self.communication_style.get('tone', '명확')
            format_style = self.communication_style.get('format', '서술식')
            parts.append(f"말투: {tone}하고 {format_style}으로 답변합니다.")

        # 말투 패턴
        if self.speech_patterns:
            if 'common_phrases' in self.speech_patterns:
                phrases = self.speech_patterns['common_phrases'][:3]  # 상위 3개
                parts.append(f"자주 사용하는 표현: {', '.join(phrases)}")

        # 종합
        system_prompt = "\n".join(parts)

        # 기본 가이드라인 추가
        system_prompt += "\n\n답변 시 이 사람의 성격과 말투를 재현하되, 정확하고 유용한 정보를 제공하세요."

        return system_prompt

    def update_interaction(self):
        """상호작용 기록 업데이트"""
        self.interaction_count += 1
        self.last_interaction = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class DigitalPersonaManager:
    """
    디지털 페르소나 통합 관리자
    - VoiceStore, RagStore, PersonaStore 통합
    - 페르소나 CRUD
    - 자동 학습 및 업데이트
    """

    def __init__(
        self,
        voice_store: Optional[VoiceStore] = None,
        rag_store: Optional[RagStore] = None,
        persona_store: Optional[PersonaStore] = None,
        storage_path: str = "data/personas"
    ):
        self.voice_store = voice_store or VoiceStore()
        self.rag_store = rag_store or RagStore()
        self.persona_store = persona_store or PersonaStore()
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # 메모리 캐시
        self.personas: Dict[str, DigitalPersona] = {}

        # 기존 페르소나 로드
        self._load_all()

    def _load_all(self):
        """저장된 모든 페르소나 로드"""
        if not os.path.exists(self.storage_path):
            return

        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                speaker_id = filename[:-5]  # .json 제거
                self.load_persona(speaker_id)

    def create_persona(
        self,
        speaker_id: str,
        display_name: str,
        voice_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> DigitalPersona:
        """새 페르소나 생성"""
        persona = DigitalPersona(
            speaker_id=speaker_id,
            display_name=display_name,
            voice_embedding=voice_embedding,
            **kwargs
        )

        # RAG 개인 컬렉션 생성
        persona.personal_collection = f"speaker_{speaker_id}_utterances"

        self.personas[speaker_id] = persona
        self.save_persona(speaker_id)

        print(f"[INFO] Created digital persona: {speaker_id} ({display_name})")
        return persona

    def get_persona(self, speaker_id: str) -> Optional[DigitalPersona]:
        """페르소나 조회"""
        if speaker_id in self.personas:
            return self.personas[speaker_id]

        # 메모리에 없으면 로드 시도
        return self.load_persona(speaker_id)

    def update_persona(self, speaker_id: str, **updates):
        """페르소나 정보 업데이트"""
        persona = self.get_persona(speaker_id)
        if not persona:
            print(f"[WARN] Persona not found: {speaker_id}")
            return

        for key, value in updates.items():
            if hasattr(persona, key):
                setattr(persona, key, value)

        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

    def save_persona(self, speaker_id: str):
        """페르소나 저장"""
        persona = self.personas.get(speaker_id)
        if not persona:
            return

        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(persona.to_dict(), f, ensure_ascii=False, indent=2)

    def load_persona(self, speaker_id: str) -> Optional[DigitalPersona]:
        """페르소나 로드"""
        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            persona = DigitalPersona.from_dict(data)
            self.personas[speaker_id] = persona
            return persona
        except Exception as e:
            print(f"[ERROR] Failed to load persona {speaker_id}: {e}")
            return None

    def add_utterance(self, speaker_id: str, text: str, start: float, end: float):
        """발언 추가 (RAG에 저장)"""
        persona = self.get_persona(speaker_id)
        if not persona:
            print(f"[WARN] Persona not found: {speaker_id}")
            return

        # RAG에 발언 저장
        segments = [{
            'speaker_id': speaker_id,
            'speaker_name': persona.display_name,
            'text': text,
            'start': start,
            'end': end,
        }]

        self.rag_store.upsert_segments(segments)

        # 발언 수 증가
        persona.utterance_count += 1
        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

    def enrich_from_prior_knowledge(
        self,
        speaker_id: str,
        prior_knowledge: Dict[str, Any]
    ) -> bool:
        """
        사전 지식 추가 (페르소나가 없으면 자동 생성)

        Args:
            speaker_id: 화자 ID
            prior_knowledge: {
                "role": "시니어 개발자",
                "department": "AI팀",
                "expertise": ["Python", "ML", "백엔드"],
                "personality_keywords": ["분석적", "논리적", "협력적"],
                "education": "컴퓨터공학 석사",
                "career_years": 8,
                ...
            }

        Returns:
            bool: 성공 여부
        """
        persona = self.get_persona(speaker_id)

        # 페르소나가 없으면 자동 생성 (음성 임베딩 없이)
        if not persona:
            print(f"[INFO] Persona not found for {speaker_id}, creating new one...")

            # display_name 추출 (prior_knowledge에서 또는 기본값)
            display_name = prior_knowledge.get('display_name', speaker_id)

            # 음성 임베딩 없이 페르소나 생성
            persona = DigitalPersona(
                speaker_id=speaker_id,
                display_name=display_name,
                voice_embedding=None,
                embedding_quality=0.0,
                llm_backend=prior_knowledge.get('llm_backend', 'openai:gpt-4o-mini'),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self.personas[speaker_id] = persona
            print(f"[INFO] Created new persona for {speaker_id}")

        # 역할 정보
        if 'role' in prior_knowledge:
            persona.role = prior_knowledge['role']

        if 'department' in prior_knowledge:
            persona.department = prior_knowledge['department']

        # 전문성
        if 'expertise' in prior_knowledge:
            persona.expertise = prior_knowledge['expertise']

        # 성격 키워드
        if 'personality_keywords' in prior_knowledge:
            persona.personality_keywords = prior_knowledge['personality_keywords']

        # 말투 스타일
        if 'communication_style' in prior_knowledge:
            persona.communication_style.update(prior_knowledge['communication_style'])

        # LLM 백엔드
        if 'llm_backend' in prior_knowledge:
            persona.llm_backend = prior_knowledge['llm_backend']

        # 나머지는 prior_knowledge에 저장
        persona.prior_knowledge.update(prior_knowledge)

        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

        print(f"[INFO] Enriched persona {speaker_id} with prior knowledge")
        return True

    def analyze_speech_patterns(self, speaker_id: str) -> Dict:
        """
        발언 패턴 자동 분석
        RAG에서 화자의 모든 발언을 가져와 패턴 추출
        """
        # RAG에서 화자 발언 조회
        utterances = self.rag_store.search_by_speaker(speaker_id, topk=1000)

        if not utterances:
            return {}

        texts = [u['text'] for u in utterances]

        # 간단한 패턴 분석 (실제로는 더 정교한 NLP 사용 가능)
        patterns = {
            'total_utterances': len(texts),
            'avg_length': sum(len(t) for t in texts) / len(texts),
            'total_words': sum(len(t.split()) for t in texts),
            'avg_words_per_utterance': sum(len(t.split()) for t in texts) / len(texts),
        }

        # 자주 사용하는 구문 (간단한 n-gram)
        from collections import Counter

        # 2-gram 분석
        bigrams = []
        for text in texts:
            words = text.split()
            bigrams.extend([f"{words[i]} {words[i+1]}" for i in range(len(words)-1)])

        common_bigrams = Counter(bigrams).most_common(10)
        patterns['common_phrases'] = [phrase for phrase, _ in common_bigrams]

        return patterns

    def get_all_personas(self) -> List[DigitalPersona]:
        """모든 페르소나 조회"""
        return list(self.personas.values())

    def delete_persona(self, speaker_id: str):
        """페르소나 삭제"""
        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        if os.path.exists(path):
            os.remove(path)

        if speaker_id in self.personas:
            del self.personas[speaker_id]

        print(f"[INFO] Deleted persona: {speaker_id}")
