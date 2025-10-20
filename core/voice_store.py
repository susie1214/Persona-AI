# -*- coding: utf-8 -*-
# core/voice_store.py
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    from qdrant_client import QdrantClient, models
    Q_OK = True
except ImportError:
    Q_OK = False
    # QdrantClient와 models를 mock 객체로 대체하여 AttributeError 방지
    class MockQdrantClient:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print(f"[WARN] Qdrant client not available. Method '{name}' was called but did nothing.")
                if name == "search":
                    return []
                if name == "scroll":
                    return [], None
                return None
            return method
    QdrantClient = MockQdrantClient
    models = type("models", (), {"Distance": type("Distance", (), {"COSINE": "cosine"})})


class VoiceStore:
    """
    화자의 음성 임베딩을 Qdrant에 저장하고 검색하는 스토어
    - pyannote/speaker-diarization-3.1 모델의 512 차원 임베딩 사용
    """

    def __init__(self, persist_path: str = "./qdrant_storage", collection_name: str = "voice_embeddings"):
        self.ok = False
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embed_dim = 512  # pyannote/speaker-diarization-3.1

        if not Q_OK:
            print("[WARN] Qdrant client not available for VoiceStore.")
            self.client = QdrantClient() # Mock client
            return

        try:
            self.client = QdrantClient(path=self.persist_path)
            self.ok = True
            print(f"[INFO] VoiceStore Qdrant client initialized with persist path: {self.persist_path}")
        except Exception as e:
            print(f"[WARN] VoiceStore Qdrant connection failed: {e}")
            self.ok = False
            return

        if self.ok:
            self._ensure_collection()

    def _ensure_collection(self):
        """컬렉션이 없으면 생성"""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embed_dim, distance=models.Distance.COSINE
                    ),
                )
                print(f"[INFO] VoiceStore created collection: {self.collection_name}")
            else:
                print(f"[INFO] VoiceStore using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"[WARN] VoiceStore create/check collection failed: {e}")
            self.ok = False

    def _speaker_id_to_uuid(self, speaker_id: str) -> str:
        """일관된 UUID 생성을 위해 speaker_id를 네임스페이스로 사용"""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, speaker_id))

    def upsert_speaker_embedding(self, speaker_id: str, display_name: str, embedding: np.ndarray):
        """화자의 대표 임베딩을 저장 (기존에 있으면 업데이트)"""
        if not self.ok:
            return

        point_id = self._speaker_id_to_uuid(speaker_id)
        vector = embedding.tolist()
        payload = {
            "speaker_id": speaker_id,
            "display_name": display_name,
            "updated_at": datetime.now().isoformat()
        }

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
                wait=True
            )
        except Exception as e:
            print(f"[WARN] VoiceStore upsert failed for speaker {speaker_id}: {e}")

    def search_similar_speaker(self, embedding: np.ndarray, threshold: float) -> Optional[Tuple[str, float]]:
        """주어진 임베딩과 가장 유사한 화자를 검색"""
        if not self.ok or not len(embedding):
            return None

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=1,
                score_threshold=threshold,
            )
            if not hits:
                return None

            best_hit = hits[0]
            speaker_id = best_hit.payload.get("speaker_id")
            score = best_hit.score
            return speaker_id, score
        except Exception as e:
            print(f"[WARN] VoiceStore search failed: {e}")
            return None

    def get_all_speakers(self) -> List[Dict]:
        """저장된 모든 화자 정보 반환"""
        if not self.ok:
            return []

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # 최대 1000명까지
                with_payload=True,
                with_vectors=False,
            )
            return [rec.payload for rec in records]
        except Exception as e:
            print(f"[WARN] VoiceStore failed to get all speakers: {e}")
            return []

    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """특정 화자 정보 조회"""
        if not self.ok:
            return None
        
        point_id = self._speaker_id_to_uuid(speaker_id)
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True
            )
            if records:
                return records[0].payload
            return None
        except Exception as e:
            print(f"[WARN] VoiceStore failed to get speaker {speaker_id}: {e}")
            return None

    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """화자의 표시 이름 업데이트"""
        if not self.ok:
            return False

        point_id = self._speaker_id_to_uuid(speaker_id)
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"display_name": new_name},
                points=[point_id],
                wait=True
            )
            return True
        except Exception as e:
            print(f"[WARN] VoiceStore failed to update name for speaker {speaker_id}: {e}")
            return False

    def delete_speaker(self, speaker_id: str) -> bool:
        """특정 화자 정보 삭제"""
        if not self.ok:
            return False

        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, speaker_id))
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
                wait=True
            )
            print(f"[INFO] VoiceStore deleted speaker: {speaker_id}")
            return True
        except Exception as e:
            print(f"[WARN] VoiceStore failed to delete speaker {speaker_id}: {e}")
            return False

    def delete_all_speakers(self):
        """모든 화자 정보 삭제 (컬렉션 재생성)"""
        if not self.ok:
            return

        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"[INFO] VoiceStore deleted collection: {self.collection_name}")
            self._ensure_collection()
        except Exception as e:
            print(f"[WARN] VoiceStore failed to delete collection: {e}")

