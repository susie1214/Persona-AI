# core/speaker.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from core.voice_store import VoiceStore

@dataclass
class Speaker:
    """메모리 상에서 관리되는 화자 객체"""
    speaker_id: str
    display_name: str
    # 대표 임베딩. 지속적으로 갱신됨.
    embedding: Optional[np.ndarray] = None
    # 임베딩 갱신에 사용된 임베딩 수
    embedding_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_average_embedding(self) -> Optional[np.ndarray]:
        return self.embedding

    def update_embedding(self, new_embedding: np.ndarray):
        """새로운 임베딩으로 대표 임베딩을 갱신 (이동 평균)"""
        if self.embedding is None:
            self.embedding = new_embedding
            self.embedding_count = 1
        else:
            # 이동 평균으로 부드럽게 갱신
            self.embedding = (self.embedding * self.embedding_count + new_embedding) / (self.embedding_count + 1)
            self.embedding_count += 1

class SpeakerManager:
    """
    화자 정보 관리의 중심.
    - VoiceStore(Qdrant)를 통해 임베딩과 메타데이터를 영구 저장.
    - 메모리 상에 화자 객체(Speaker)를 캐싱하여 빠른 접근 제공.
    """
    def __init__(self, voice_store: VoiceStore = None):
        self.voice_store = voice_store if voice_store else VoiceStore()
        self.speakers: Dict[str, Speaker] = {}  # speaker_id -> Speaker object (in-memory cache)
        self.next_speaker_id_num = 1
        self._load_from_store()

    def _load_from_store(self):
        """VoiceStore에서 모든 화자 정보를 로드하여 메모리 캐시를 채움"""
        if not self.voice_store.ok:
            print("[WARN] VoiceStore is not OK. Cannot load speakers.")
            return

        all_speakers_data = self.voice_store.get_all_speakers()
        self.speakers = {}
        max_id = 0
        for data in all_speakers_data:
            speaker_id = data.get('speaker_id')
            if not speaker_id:
                continue
            
            # Qdrant에서 벡터를 직접 로드하지 않으므로, 초기 임베딩은 None으로 설정
            # 필요 시점에 lazy loading 하거나, identify 시점에 갱신
            speaker = Speaker(
                speaker_id=speaker_id,
                display_name=data.get('display_name', speaker_id)
            )
            self.speakers[speaker_id] = speaker

            # next_speaker_id 계산
            try:
                num = int(speaker_id.split('_')[-1])
                if num > max_id:
                    max_id = num
            except (ValueError, IndexError):
                continue
        
        self.next_speaker_id_num = max_id + 1
        print(f"[INFO] Loaded {len(self.speakers)} speakers from VoiceStore. Next ID: {self.next_speaker_id_num}")

    def identify_speaker(self, new_embedding: np.ndarray, threshold: float = 0.75) -> tuple[str, float]:
        """
        새로운 음성 임베딩을 기반으로 화자를 식별하거나 새로 생성.
        - 일치하는 화자를 찾으면 대표 임베딩을 갱신.
        - 일치하는 화자가 없으면 새로운 화자를 생성.
        """
        if not self.voice_store.ok:
            print("[WARN] VoiceStore not OK. Cannot identify speaker.")
            # VoiceStore가 없으면 임시 ID 반환
            return f"temp_{np.random.randint(1000)}", 0.5

        # 1. DB에서 유사 화자 검색
        match = self.voice_store.search_similar_speaker(new_embedding, threshold)

        if match:
            # 2a. 일치하는 화자 찾음
            speaker_id, confidence = match
            
            # 메모리 캐시에서 화자 객체 가져오기
            speaker = self.get_speaker_by_id(speaker_id)
            if speaker:
                # 대표 임베딩 갱신 (메모리)
                speaker.update_embedding(new_embedding)
                # 갱신된 임베딩을 DB에 저장
                self.voice_store.upsert_speaker_embedding(
                    speaker_id=speaker.speaker_id,
                    display_name=speaker.display_name,
                    embedding=speaker.embedding
                )
                print(f"[화자 식별] 기존 화자: {speaker_id}, 유사도: {confidence:.3f}")
            else:
                # DB에는 있지만 메모리에 없는 경우 (예: 다른 인스턴스에서 추가)
                # 새로 로드하거나, 여기서 새로 생성
                self._load_from_store() # 간단하게 전체 리로드
                print(f"[화자 식별] Stale cache hit. Reloaded speakers.")

            return speaker_id, confidence
        else:
            # 2b. 일치하는 화자 없음 -> 새로 생성
            new_speaker_id = self.create_new_speaker(new_embedding)
            print(f"[화자 식별] 새 화자 생성: {new_speaker_id}")
            return new_speaker_id, 1.0

    def create_new_speaker(self, embedding: np.ndarray) -> str:
        """새로운 화자를 생성하고 DB와 메모리에 추가"""
        speaker_id = f"speaker_{self.next_speaker_id_num:02d}"
        display_name = speaker_id  # 초기 표시 이름은 ID와 동일

        # 1. DB에 저장
        self.voice_store.upsert_speaker_embedding(speaker_id, display_name, embedding)

        # 2. 메모리 캐시에 추가
        new_speaker = Speaker(
            speaker_id=speaker_id,
            display_name=display_name,
            embedding=embedding,
            embedding_count=1
        )
        self.speakers[speaker_id] = new_speaker
        
        # 3. 다음 ID 증가
        self.next_speaker_id_num += 1

        return speaker_id

    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """화자의 표시 이름 업데이트"""
        if not self.voice_store.ok:
            return False

        # 1. DB 업데이트
        success = self.voice_store.update_speaker_name(speaker_id, new_name)
        
        if success:
            # 2. 메모리 캐시 업데이트
            speaker = self.get_speaker_by_id(speaker_id)
            if speaker:
                speaker.display_name = new_name
            print(f"[INFO] Updated speaker name for {speaker_id} to '{new_name}'")
        
        return success

    def get_speaker_by_id(self, speaker_id: str) -> Optional[Speaker]:
        return self.speakers.get(speaker_id)

    def get_speaker_display_name(self, speaker_id: str) -> str:
        speaker = self.get_speaker_by_id(speaker_id)
        return speaker.display_name if speaker else speaker_id

    def get_all_speakers(self) -> List[tuple[str, str, int]]:
        """모든 화자 정보 반환 (ID, 이름, 임베딩 수) - UI 표시용"""
        # 메모리 캐시의 최신 정보를 반환
        return [(s.speaker_id, s.display_name, s.embedding_count) for s in self.speakers.values()]

    def reset_all_speakers(self) -> bool:
        """모든 화자 정보 초기화"""
        if not self.voice_store.ok:
            return False
        
        # 1. DB 초기화
        self.voice_store.delete_all_speakers()
        
        # 2. 메모리 캐시 초기화
        self.speakers = {}
        self.next_speaker_id_num = 1
        
        print("[INFO] All speakers have been reset.")
        return True

    def reload(self):
        """DB에서 강제로 다시 로드"""
        print("[INFO] Reloading speakers from VoiceStore...")
        self._load_from_store()