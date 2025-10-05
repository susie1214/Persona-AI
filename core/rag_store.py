# core/rag_store.py
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

try:
    from .audio import Segment
except ImportError:
    Segment = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range
    from sentence_transformers import SentenceTransformer

    Q_OK = True
except Exception:
    Q_OK = False


class RagStore:
    """
    Speaker별 발언을 저장하고 검색하는 RAG 스토어
    - Speaker ID 기반 필터링
    - 시간 범위 검색
    - Semantic 검색
    """

    def __init__(self, persist_path: str = "./qdrant_db"):
        self.ok = False
        self.persist_path = persist_path

        if not Q_OK:
            print("[WARN] Qdrant client or SentenceTransformer not available")
            return

        try:
            # 영구 저장소 사용 (메모리 대신)
            self.client = QdrantClient(path=persist_path)
            self.ok = True
            print(f"[INFO] Qdrant client initialized with persist path: {persist_path}")
        except Exception as e:
            print(f"[WARN] Qdrant connection failed: {e}")
            self.ok = False

        self.collection = "meeting_ctx"
        self.embed_dim = 384
        self.model = None

        if self.ok:
            try:
                self.model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                self.embed_dim = self.model.get_sentence_embedding_dimension()

                # 컬렉션 생성 (이미 존재하면 재사용)
                collections = self.client.get_collections().collections
                if not any(c.name == self.collection for c in collections):
                    self.client.create_collection(
                        self.collection,
                        vectors_config=VectorParams(
                            size=self.embed_dim, distance=Distance.COSINE
                        ),
                    )
                    print(f"[INFO] Created collection: {self.collection}")
                else:
                    print(f"[INFO] Using existing collection: {self.collection}")

            except Exception as e:
                print(f"[WARN] create collection failed: {e}")
                self.ok = False

        self._id_seq = 1

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩 벡터로 변환"""
        if self.model is None:
            # Fallback: 간단한 해시 기반 임베딩
            vecs = []
            for t in texts:
                v = np.zeros(256, dtype=np.float32)
                for i, ch in enumerate(t.encode("utf-8")):
                    v[i % 256] += (ch % 13) / 13.0
                n = np.linalg.norm(v) + 1e-9
                vecs.append((v / n).tolist())
            return vecs

        em = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return em.tolist() if isinstance(em, np.ndarray) else em

    def upsert_segments(self, segs: List) -> int:
        """
        발언 세그먼트를 벡터DB에 저장

        Args:
            segs: Segment 객체 또는 Dict 리스트

        Returns:
            저장된 세그먼트 수
        """
        if not self.ok:
            print("[WARN] RAG store not initialized")
            return 0

        payloads = []
        texts = []
        ids = []

        for s in segs:
            # Segment 객체 또는 Dict 처리
            if hasattr(s, 'speaker_name'):
                speaker_name = s.speaker_name
                speaker_id = s.speaker_id
                text = s.text
                start = s.start
                end = s.end
            else:
                speaker_name = s.get('speaker_name', s.get('speaker', 'Unknown'))
                speaker_id = s.get('speaker_id', s.get('speaker', 'Unknown'))
                text = s.get('text', '')
                start = s.get('start', 0.0)
                end = s.get('end', 0.0)

            if not text.strip():
                continue

            texts.append(f"[{speaker_name}] {text}")
            ids.append(self._id_seq)
            payloads.append({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "start": float(start),
                "end": float(end),
                "text": text,
                "timestamp": datetime.now().isoformat(),
            })
            self._id_seq += 1

        if not texts:
            return 0

        vecs = self._embed(texts)
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vecs, payloads)
        ]

        try:
            self.client.upsert(self.collection, points=points)
            print(f"[INFO] Upserted {len(points)} segments to RAG")
            return len(points)
        except Exception as e:
            print(f"[WARN] RAG upsert failed: {e}")
            return 0

    def search(
        self,
        query: str,
        topk: int = 5,
        speaker_id: Optional[str] = None,
        speaker_name: Optional[str] = None,
        time_range: Optional[tuple] = None
    ) -> List[Dict]:
        """
        발언 검색

        Args:
            query: 검색 쿼리
            topk: 반환할 결과 수
            speaker_id: 특정 화자 ID 필터링 (선택)
            speaker_name: 특정 화자 이름 필터링 (선택)
            time_range: (start_timestamp, end_timestamp) 시간 범위 (선택)

        Returns:
            검색 결과 리스트 (payload + score)
        """
        if not self.ok:
            return []

        vec = self._embed([query])[0]

        # 필터 조건 구성
        filter_conditions = []

        if speaker_id:
            filter_conditions.append(
                FieldCondition(
                    key="speaker_id",
                    match=MatchValue(value=speaker_id)
                )
            )

        if speaker_name:
            filter_conditions.append(
                FieldCondition(
                    key="speaker_name",
                    match=MatchValue(value=speaker_name)
                )
            )

        if time_range:
            start_time, end_time = time_range
            filter_conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(
                        gte=start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time,
                        lte=end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time
                    )
                )
            )

        # 필터 적용
        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        try:
            res = self.client.search(
                self.collection,
                query_vector=vec,
                limit=topk,
                query_filter=query_filter
            )

            out = []
            for r in res:
                pl = r.payload or {}
                pl["_score"] = r.score
                out.append(pl)

            return out

        except Exception as e:
            print(f"[WARN] RAG search failed: {e}")
            return []

    def search_by_speaker(self, speaker_id: str, query: str = "", topk: int = 10) -> List[Dict]:
        """
        특정 화자의 발언 검색

        Args:
            speaker_id: 화자 ID
            query: 검색 쿼리 (비어있으면 모든 발언)
            topk: 반환할 결과 수

        Returns:
            화자의 발언 리스트
        """
        if not query:
            # 쿼리 없이 화자의 모든 발언 가져오기
            query = speaker_id  # 화자 ID로 검색

        return self.search(query=query, topk=topk, speaker_id=speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """
        화자의 통계 정보

        Args:
            speaker_id: 화자 ID

        Returns:
            통계 정보 (발언 수, 평균 길이 등)
        """
        if not self.ok:
            return {}

        try:
            # 화자의 모든 발언 가져오기
            results = self.search_by_speaker(speaker_id, query=speaker_id, topk=1000)

            if not results:
                return {
                    "speaker_id": speaker_id,
                    "total_utterances": 0,
                    "avg_length": 0,
                    "total_duration": 0,
                }

            texts = [r["text"] for r in results]
            durations = [r["end"] - r["start"] for r in results]

            return {
                "speaker_id": speaker_id,
                "speaker_name": results[0].get("speaker_name", speaker_id),
                "total_utterances": len(results),
                "avg_length": sum(len(t) for t in texts) / len(texts),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations),
            }

        except Exception as e:
            print(f"[WARN] Failed to get speaker stats: {e}")
            return {}

    def get_all_speakers(self) -> List[str]:
        """
        저장된 모든 화자 ID 목록

        Returns:
            화자 ID 리스트
        """
        if not self.ok:
            return []

        try:
            # Scroll을 사용해 모든 포인트 가져오기
            records, _ = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            speakers = set()
            for record in records:
                if record.payload and "speaker_id" in record.payload:
                    speakers.add(record.payload["speaker_id"])

            return sorted(list(speakers))

        except Exception as e:
            print(f"[WARN] Failed to get all speakers: {e}")
            return []

    def clear_collection(self):
        """컬렉션 초기화"""
        if not self.ok:
            return False

        try:
            self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(
                    size=self.embed_dim, distance=Distance.COSINE
                ),
            )
            self._id_seq = 1
            print(f"[INFO] Cleared collection: {self.collection}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to clear collection: {e}")
            return False
