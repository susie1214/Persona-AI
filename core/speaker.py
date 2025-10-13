# core/speaker.py
import pickle
import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

SPEAKER_PROFILES_PATH = "speaker_profiles.pkl"
SPEAKER_MAPPING_PATH = "speaker_mapping.json"

@dataclass
class Speaker:
    speaker_id: str  # speaker_01, speaker_02, etc.
    display_name: str  # 사용자가 설정한 이름 (예: "김태진")
    embeddings: List[np.ndarray] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_scores: List[float] = field(default_factory=list)

    # 대화 히스토리 (Phase 1 추가)
    utterances: List[Dict] = field(default_factory=list)  # {text, timestamp, meeting_id}
    conversation_stats: Dict = field(default_factory=dict)  # 통계 정보

    def get_average_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)

    def get_confidence_score(self) -> float:
        """평균 신뢰도 점수 반환"""
        if not self.confidence_scores:
            return 0.0
        return np.mean(self.confidence_scores)

    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """임베딩과 신뢰도 점수 추가"""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)

    def add_utterance(self, text: str, timestamp: str = None, meeting_id: str = None):
        """발언 추가"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        self.utterances.append({
            "text": text,
            "timestamp": timestamp,
            "meeting_id": meeting_id
        })

        # 통계 업데이트
        self._update_stats()

    def _update_stats(self):
        """대화 통계 업데이트"""
        if not self.utterances:
            self.conversation_stats = {}
            return

        texts = [u["text"] for u in self.utterances]

        # 기본 통계
        self.conversation_stats = {
            "total_utterances": len(texts),
            "avg_length": sum(len(t) for t in texts) / len(texts),
            "total_chars": sum(len(t) for t in texts),
            "last_updated": datetime.now().isoformat()
        }

    def get_recent_utterances(self, limit: int = 10) -> List[Dict]:
        """최근 발언 가져오기"""
        return self.utterances[-limit:] if self.utterances else []

class SpeakerManager:
    def __init__(self):
        self.speakers: List[Speaker] = []
        self.speaker_mapping: Dict[str, str] = {}  # speaker_id -> display_name
        self.next_speaker_id = 1
        self.load_speakers()
        self.load_speaker_mapping()

        # 안전성 체크: speakers가 list인지 확인
        if not isinstance(self.speakers, list):
            print(f"ERROR: speakers is not a list, type={type(self.speakers)}, converting to list")
            if isinstance(self.speakers, dict):
                self.speakers = list(self.speakers.values())
            else:
                self.speakers = []

    def load_speakers(self):
        try:
            with open(SPEAKER_PROFILES_PATH, "rb") as f:
                loaded_data = pickle.load(f)

                # dict로 로드된 경우 list로 변환
                if isinstance(loaded_data, dict):
                    print("Warning: speakers loaded as dict, converting to list")
                    self.speakers = list(loaded_data.values()) if loaded_data else []
                elif isinstance(loaded_data, list):
                    self.speakers = loaded_data
                else:
                    print(f"Warning: unexpected speakers type: {type(loaded_data)}")
                    self.speakers = []

            print(f"Loaded {len(self.speakers)} speaker profiles.")
        except FileNotFoundError:
            self.speakers = []
            print("Speaker profiles file not found. Starting fresh.")
        except Exception as e:
            self.speakers = []
            print(f"Error loading speaker profiles: {e}")

    def save_speakers(self):
        try:
            with open(SPEAKER_PROFILES_PATH, "wb") as f:
                pickle.dump(self.speakers, f)
            print("Speaker profiles saved.")
        except Exception as e:
            print(f"Error saving speaker profiles: {e}")

    def load_speaker_mapping(self):
        """화자 ID와 표시 이름 매핑 로드"""
        try:
            if os.path.exists(SPEAKER_MAPPING_PATH):
                with open(SPEAKER_MAPPING_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.speaker_mapping = data.get("mapping", {})
                    self.next_speaker_id = data.get("next_id", 1)

                    # speaker_mapping에는 있지만 speakers 리스트에 없는 화자 생성
                    existing_ids = {s.speaker_id for s in self.speakers}
                    for speaker_id, display_name in self.speaker_mapping.items():
                        if speaker_id not in existing_ids:
                            new_speaker = Speaker(
                                speaker_id=speaker_id,
                                display_name=display_name,
                                embeddings=[],
                                confidence_scores=[]
                            )
                            self.speakers.append(new_speaker)
                            print(f"Created speaker from mapping: {speaker_id} ({display_name})")

            print(f"Loaded speaker mapping: {len(self.speaker_mapping)} mappings")
        except Exception as e:
            print(f"Error loading speaker mapping: {e}")
            self.speaker_mapping = {}
            self.next_speaker_id = 1

    def save_speaker_mapping(self):
        """화자 ID와 표시 이름 매핑 저장"""
        try:
            data = {
                "mapping": self.speaker_mapping,
                "next_id": self.next_speaker_id,
                "updated_at": datetime.now().isoformat()
            }
            with open(SPEAKER_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Speaker mapping saved.")
        except Exception as e:
            print(f"Error saving speaker mapping: {e}")

    def add_speaker_embedding(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """기존 화자에게 임베딩 추가"""
        speaker = self.get_speaker_by_id(speaker_id)
        if speaker:
            speaker.add_embedding(embedding, confidence)
            self.save_speakers()
            return speaker_id
        else:
            print(f"Warning: Speaker {speaker_id} not found")
            return None

    def create_new_speaker(self, embedding: np.ndarray, confidence: float = 1.0) -> str:
        """새로운 화자 생성 (자동으로 speaker_xx ID 할당)"""
        # 안전성 체크
        if not isinstance(self.speakers, list):
            print("ERROR: speakers is not a list in create_new_speaker, converting")
            self.speakers = list(self.speakers.values()) if isinstance(self.speakers, dict) else []

        speaker_id = f"speaker_{self.next_speaker_id:02d}"
        display_name = speaker_id  # 기본값으로 ID와 동일

        new_speaker = Speaker(
            speaker_id=speaker_id,
            display_name=display_name,
            embeddings=[embedding],
            confidence_scores=[confidence]
        )

        self.speakers.append(new_speaker)
        self.speaker_mapping[speaker_id] = display_name
        self.next_speaker_id += 1

        self.save_speakers()
        self.save_speaker_mapping()

        print(f"Created new speaker: {speaker_id}")
        return speaker_id

    def get_speaker_by_id(self, speaker_id: str) -> Optional[Speaker]:
        """ID로 화자 찾기"""
        for speaker in self.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None

    def get_speaker_by_name(self, display_name: str) -> Optional[Speaker]:
        """표시 이름으로 화자 찾기"""
        for speaker in self.speakers:
            if speaker.display_name == display_name:
                return speaker
        return None

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def identify_speaker(self, embedding: np.ndarray, threshold: float = 0.85) -> tuple[str, float]:
        """화자 식별 또는 새 화자 생성

        Returns:
            tuple: (speaker_id, confidence_score)
        """
        # 안전성 체크
        if not isinstance(self.speakers, list):
            print("ERROR: speakers is not a list in identify_speaker, converting")
            self.speakers = list(self.speakers.values()) if isinstance(self.speakers, dict) else []

        best_match_id = None
        max_similarity = -1.0

        for speaker in self.speakers:
            avg_embedding = speaker.get_average_embedding()
            if avg_embedding is None:
                continue

            similarity = self._cosine_similarity(embedding, avg_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = speaker.speaker_id

        if max_similarity > threshold:
            # 기존 화자에 임베딩 추가
            self.add_speaker_embedding(best_match_id, embedding, max_similarity)
            print(f"[화자 식별] 기존 화자: {best_match_id}, 유사도: {max_similarity:.3f} (임계값: {threshold})")
            return best_match_id, max_similarity
        else:
            # 새로운 화자 생성
            new_speaker_id = self.create_new_speaker(embedding, 1.0)
            print(f"[화자 식별] 새 화자 생성: {new_speaker_id}, 최대 유사도: {max_similarity:.3f} (임계값: {threshold})")
            return new_speaker_id, 1.0

    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """화자의 표시 이름 업데이트"""
        speaker = self.get_speaker_by_id(speaker_id)
        if speaker:
            old_name = speaker.display_name
            speaker.display_name = new_name
            self.speaker_mapping[speaker_id] = new_name
            self.save_speakers()
            self.save_speaker_mapping()
            print(f"Updated speaker {speaker_id}: {old_name} -> {new_name}")
            return True
        return False

    def get_all_speakers(self) -> List[tuple[str, str, int]]:
        """모든 화자 정보 반환 (ID, 이름, 임베딩 수)"""
        return [(s.speaker_id, s.display_name, len(s.embeddings)) for s in self.speakers]

    def get_speaker_display_name(self, speaker_id: str) -> str:
        """화자 ID에 대한 표시 이름 반환"""
        return self.speaker_mapping.get(speaker_id, speaker_id)

    def reset_all_speakers(self) -> bool:
        """모든 화자 정보 초기화"""
        try:
            # 메모리 초기화
            self.speakers = []
            self.speaker_mapping = {}
            self.next_speaker_id = 1

            # 파일 삭제
            import os
            if os.path.exists(SPEAKER_PROFILES_PATH):
                os.remove(SPEAKER_PROFILES_PATH)
                print(f"Deleted: {SPEAKER_PROFILES_PATH}")

            if os.path.exists(SPEAKER_MAPPING_PATH):
                os.remove(SPEAKER_MAPPING_PATH)
                print(f"Deleted: {SPEAKER_MAPPING_PATH}")

            print("All speakers have been reset.")
            return True
        except Exception as e:
            print(f"Error resetting speakers: {e}")
            return False
