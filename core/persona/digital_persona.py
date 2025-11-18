# -*- coding: utf-8 -*-
# core/persona/digital_persona.py
"""
통합 디지털 페르소나 시스템
- 음성 임베딩 + 발언 이력 + 말투 패턴 + 사전 설문 정보
- speaker_id 하나로 PersonaStore(설문) + DigitalPersona(이력)를 연결
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np

from core.speaker import VoiceStore
from core.rag import RagStore
from .persona_store import PersonaStore  # core/persona/persona_store.py


# ----------------------------------------------------------------------
# 디지털 페르소나 데이터 모델
# ----------------------------------------------------------------------
@dataclass
class DigitalPersona:
    """
    화자 1명에 대한 통합 프로필
    - speaker_id = 설문/프로필 키 = QLoRA 어댑터 키
    """

    # 기본 식별 정보
    speaker_id: str              # 예: "speaker_01"
    display_name: str            # 화면에 보여줄 이름

    # 음성 특징
    voice_embedding: Optional[np.ndarray] = None
    embedding_quality: float = 0.0  # 0~1 신뢰도

    # 역할/전문성
    role: str = ""               # "팀장", "개발자" 등
    department: str = ""
    expertise: List[str] = field(default_factory=list)

    # 성격 및 커뮤니케이션 스타일
    personality_keywords: List[str] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)

    # 말투 패턴 (자동 분석 결과)
    speech_patterns: Dict[str, Any] = field(default_factory=dict)

    # 학습된 모델
    qlora_adapter_path: Optional[str] = None
    llm_backend: str = "kanana:kakao/kanana-1.5-2.1b-instruct"

    # RAG 컬렉션
    personal_collection: Optional[str] = None
    utterance_count: int = 0
    meeting_count: int = 0

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_interaction: Optional[str] = None
    interaction_count: int = 0

    # 수동 입력 사전 지식
    prior_knowledge: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화용 dict (numpy -> list 변환 포함)"""
        data = asdict(self)
        if self.voice_embedding is not None:
            data["voice_embedding"] = self.voice_embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DigitalPersona":
        """JSON 역직렬화"""
        ve = data.get("voice_embedding")
        if ve is not None:
            data["voice_embedding"] = np.array(ve)
        return cls(**data)

    # --------------------------------------------------------------
    def generate_system_prompt(self) -> str:
        """
        이력 기반 동적 시스템 프롬프트
        (PersonaStore의 설문 프롬프트와 합쳐져서 최종 system prompt가 됨)
        """
        parts: List[str] = []

        parts.append(f"당신은 '{self.display_name}'의 디지털 페르소나입니다.")

        if self.role:
            parts.append(f"역할: {self.role}")
        if self.department:
            parts.append(f"소속: {self.department}")
        if self.expertise:
            parts.append("전문 분야: " + ", ".join(self.expertise))
        if self.personality_keywords:
            parts.append("성격 특징: " + ", ".join(self.personality_keywords))

        if self.communication_style:
            tone = self.communication_style.get("tone", "명확")
            fmt = self.communication_style.get("format", "서술식")
            parts.append(f"기본 말투: {tone}하고 {fmt}으로 답변합니다.")

        # 말투 패턴은 회의가 어느 정도 누적되었을 때만 사용
        if self.speech_patterns and self.meeting_count >= 3:
            parts.append("\n[학습된 말투 특징]")

            phrases = self.speech_patterns.get("common_phrases", [])[:3]
            if phrases:
                parts.append("자주 사용하는 표현: " + ", ".join(phrases))

            endings = self.speech_patterns.get("sentence_endings", [])[:3]
            if endings:
                parts.append("문장 끝 표현: " + ", ".join(endings))

            avg_words = self.speech_patterns.get("avg_words_per_utterance", 0)
            if avg_words > 0:
                if avg_words < 10:
                    parts.append("답변 스타일: 간결하고 핵심만 전달")
                elif avg_words < 20:
                    parts.append("답변 스타일: 적절한 길이로 설명")
                else:
                    parts.append("답변 스타일: 상세하고 구체적으로 설명")

        system_prompt = "\n".join(parts)

        if self.meeting_count >= 3:
            system_prompt += "\n\n답변 시 위의 학습된 말투 특징을 최대한 반영하여, 이 사람처럼 자연스럽게 대화하세요."
        else:
            system_prompt += "\n\n답변 시 이 사람의 성격과 말투를 재현하되, 정확하고 유용한 정보를 제공하세요."

        return system_prompt

    def update_interaction(self):
        self.interaction_count += 1
        now = datetime.now().isoformat()
        self.last_interaction = now
        self.updated_at = now


# ----------------------------------------------------------------------
# 디지털 페르소나 매니저
# ----------------------------------------------------------------------
class DigitalPersonaManager:
    """
    VoiceStore, RagStore, PersonaStore와 연동되는 디지털 페르소나 통합 관리자

    - speaker_id 기준으로:
        * PersonaStore: 설문/선호(정적 프로필)
        * DigitalPersona: 음성 + 회의 이력(동적 프로필)
        * adapters/speaker_id: QLoRA 어댑터
    """

    def __init__(
        self,
        voice_store: Optional[VoiceStore] = None,
        rag_store: Optional[RagStore] = None,
        persona_store: Optional[PersonaStore] = None,
        storage_path: str = "data/personas",
    ):
        self.voice_store = voice_store or VoiceStore()
        self.rag_store = rag_store or RagStore()
        self.persona_store = persona_store or PersonaStore()
        self.storage_path = storage_path

        os.makedirs(self.storage_path, exist_ok=True)

        # 캐시: speaker_id -> DigitalPersona
        self.personas: Dict[str, DigitalPersona] = {}

        self._load_all()

    # --------------------------------------------------------------
    def _load_all(self):
        """storage_path 아래 *.json 로드"""
        if not os.path.exists(self.storage_path):
            return

        for fname in os.listdir(self.storage_path):
            if fname.endswith(".json"):
                sid = fname[:-5]
                self.load_persona(sid)

    def create_persona(
        self,
        speaker_id: str,
        display_name: str,
        voice_embedding: Optional[np.ndarray] = None,
        **kwargs,
    ) -> DigitalPersona:
        """
        새 디지털 페르소나 생성
        - speaker_id = 설문, 어댑터, RAG 컬렉션 키
        """
        persona = DigitalPersona(
            speaker_id=speaker_id,
            display_name=display_name,
            voice_embedding=voice_embedding,
            **kwargs,
        )
        persona.personal_collection = f"speaker_{speaker_id}_utterances"

        self.personas[speaker_id] = persona
        self.save_persona(speaker_id)

        print(f"[INFO] Created digital persona: {speaker_id} ({display_name})")
        return persona

    def get_persona(self, speaker_id: str) -> Optional[DigitalPersona]:
        if speaker_id in self.personas:
            return self.personas[speaker_id]
        return self.load_persona(speaker_id)

    def save_persona(self, speaker_id: str):
        persona = self.personas.get(speaker_id)
        if not persona:
            return
        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(persona.to_dict(), f, ensure_ascii=False, indent=2)

    def load_persona(self, speaker_id: str) -> Optional[DigitalPersona]:
        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            persona = DigitalPersona.from_dict(data)
            self.personas[speaker_id] = persona
            return persona
        except Exception as e:
            print(f"[ERROR] Failed to load persona {speaker_id}: {e}")
            return None

    # --------------------------------------------------------------
    # 기존 코드 호환용: update_persona
    # --------------------------------------------------------------
    def update_persona(self, speaker_id: str, **updates):
        """
        기존 UI/매니저에서 사용하는 업데이트용 메서드.
        예: update_persona(speaker_id, display_name="새 이름")

        존재하는 필드만 setattr 하고, updated_at을 갱신한 뒤 저장한다.
        """
        persona = self.get_persona(speaker_id)
        if not persona:
            print(f"[WARN] Persona not found: {speaker_id}")
            return

        for key, value in updates.items():
            if hasattr(persona, key):
                setattr(persona, key, value)

        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

    # --------------------------------------------------------------
    # 발언/RAG 연동
    # --------------------------------------------------------------
    def _is_valid_utterance(self, text: str, min_words: int = 3) -> bool:
        if not text or not text.strip():
            return False
        if len(text.strip().split()) < min_words:
            return False
        return True

    def add_utterance(self, speaker_id: str, text: str, start: float, end: float):
        persona = self.get_persona(speaker_id)
        if not persona:
            print(f"[WARN] Persona not found: {speaker_id}")
            return

        if not self._is_valid_utterance(text):
            print(f"[DEBUG] Skipped short utterance: '{text[:30]}...' (speaker: {speaker_id})")
            return

        segments = [
            {
                "speaker_id": speaker_id,
                "speaker_name": persona.display_name,
                "text": text,
                "start": start,
                "end": end,
            }
        ]
        self.rag_store.upsert_segments(segments)

        persona.utterance_count += 1
        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

    def on_meeting_ended(self, speaker_ids: List[str]):
        """
        회의 녹음 종료 시 호출
        - meeting_count 증가
        - 3회마다 말투 패턴 자동 업데이트
        """
        for sid in speaker_ids:
            persona = self.get_persona(sid)
            if not persona:
                continue

            persona.meeting_count += 1
            persona.updated_at = datetime.now().isoformat()
            print(f"[INFO] Meeting ended for {sid} (#{persona.meeting_count})")

            if persona.meeting_count > 0 and persona.meeting_count % 3 == 0:
                print(f"[INFO] Auto-updating speech patterns for {sid}")
                self.auto_update_speech_patterns(sid)

            self.save_persona(sid)

    # --------------------------------------------------------------
    # 설문/사전지식 연동
    # --------------------------------------------------------------
    def enrich_from_prior_knowledge(
        self,
        speaker_id: str,
        prior_knowledge: Dict[str, Any],
    ) -> bool:
        """
        prior_knowledge를 DigitalPersona에 병합
        (필요 시 새 persona 자동 생성)
        """
        persona = self.get_persona(speaker_id)
        if not persona:
            print(f"[INFO] Persona not found for {speaker_id}, creating new one...")
            display_name = prior_knowledge.get("display_name", speaker_id)
            persona = DigitalPersona(
                speaker_id=speaker_id,
                display_name=display_name,
                voice_embedding=None,
                embedding_quality=0.0,
                llm_backend=prior_knowledge.get("llm_backend", "openai:gpt-4o-mini"),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self.personas[speaker_id] = persona

        if "role" in prior_knowledge:
            persona.role = prior_knowledge["role"]
        if "department" in prior_knowledge:
            persona.department = prior_knowledge["department"]
        if "expertise" in prior_knowledge:
            persona.expertise = prior_knowledge["expertise"]
        if "personality_keywords" in prior_knowledge:
            persona.personality_keywords = prior_knowledge["personality_keywords"]
        if "communication_style" in prior_knowledge:
            persona.communication_style.update(prior_knowledge["communication_style"])
        if "llm_backend" in prior_knowledge:
            persona.llm_backend = prior_knowledge["llm_backend"]

        persona.prior_knowledge.update(prior_knowledge)
        persona.updated_at = datetime.now().isoformat()
        self.save_persona(speaker_id)

        print(f"[INFO] Enriched persona {speaker_id} with prior knowledge")
        return True

    # --------------------------------------------------------------
    # 말투 패턴 분석
    # --------------------------------------------------------------
    def analyze_speech_patterns(self, speaker_id: str) -> Dict[str, Any]:
        utterances = self.rag_store.search_by_speaker(speaker_id, topk=1000)
        if not utterances:
            return {}

        texts = [u["text"] for u in utterances]

        from collections import Counter

        patterns: Dict[str, Any] = {
            "total_utterances": len(texts),
            "avg_length": sum(len(t) for t in texts) / len(texts),
            "total_words": sum(len(t.split()) for t in texts),
        }
        patterns["avg_words_per_utterance"] = patterns["total_words"] / max(
            patterns["total_utterances"], 1
        )

        # 2-gram으로 자주 나오는 구문
        bigrams = []
        for t in texts:
            words = t.split()
            bigrams.extend(
                [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            )
        common_bigrams = Counter(bigrams).most_common(10)
        patterns["common_phrases"] = [p for p, _ in common_bigrams]

        # 마지막 3글자 기준 종결 어미
        endings = []
        for t in texts:
            sentences = [s.strip() for s in t.split(".") if s.strip()]
            for s in sentences:
                if len(s) > 3:
                    endings.append(s[-3:])
        common_endings = Counter(endings).most_common(5)
        patterns["sentence_endings"] = [e for e, _ in common_endings]

        return patterns

    def auto_update_speech_patterns(self, speaker_id: str):
        persona = self.get_persona(speaker_id)
        if not persona:
            return

        patterns = self.analyze_speech_patterns(speaker_id)
        if not patterns:
            return

        persona.speech_patterns = patterns
        persona.updated_at = datetime.now().isoformat()

        print(f"[INFO] Updated speech patterns for {speaker_id}:")
        print(f"  - Total utterances: {patterns.get('total_utterances', 0)}")
        print(f"  - Common phrases: {patterns.get('common_phrases', [])[:3]}")
        print(f"  - Sentence endings: {patterns.get('sentence_endings', [])[:3]}")

        self.save_persona(speaker_id)

    # --------------------------------------------------------------
    # System prompt 결합
    # --------------------------------------------------------------
    def build_combined_system_prompt(self, speaker_id: str) -> str:
        """
        speaker_id 기준으로
        - PersonaStore의 설문/선호 프롬프트
        - DigitalPersona의 이력 기반 프롬프트
        를 합친 최종 system prompt 생성
        """
        base_sys = self.persona_store.build_system_prompt(speaker_id)
        persona = self.get_persona(speaker_id)

        if not persona:
            # 설문만 있는 경우
            return base_sys or "You are a helpful assistant."

        dp_sys = persona.generate_system_prompt()

        if base_sys:
            return base_sys + "\n\n" + dp_sys
        return dp_sys

    # --------------------------------------------------------------
    def get_all_personas(self) -> List[DigitalPersona]:
        return list(self.personas.values())

    def delete_persona(self, speaker_id: str):
        import shutil

        path = os.path.join(self.storage_path, f"{speaker_id}.json")
        if os.path.exists(path):
            os.remove(path)

        if speaker_id in self.personas:
            del self.personas[speaker_id]

        adapter_dir = os.path.join("adapters", speaker_id)
        if os.path.exists(adapter_dir):
            try:
                shutil.rmtree(adapter_dir)
                print(f"[INFO] Deleted adapter directory: {adapter_dir}")
            except Exception as e:
                print(f"[WARN] Failed to delete adapter directory: {e}")

        print(f"[INFO] Deleted persona: {speaker_id}")
