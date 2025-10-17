# # core/rag_store.py
# import numpy as np
# from typing import List, Dict, Optional
# from datetime import datetime
# import torch

# try:
#     from .audio import Segment
# except ImportError:
#     Segment = None

# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range
#     from sentence_transformers import SentenceTransformer

#     Q_OK = True
# except Exception:
#     Q_OK = False


# class RagStore:
#     """
#     Speaker별 발언을 저장하고 검색하는 RAG 스토어
#     - Speaker ID 기반 필터링
#     - 시간 범위 검색
#     - Semantic 검색
#     """

#     def __init__(self, persist_path: str = "./qdrant_db"):
#         self.ok = False
#         self.persist_path = persist_path

#         if not Q_OK:
#             print("[WARN] Qdrant client or SentenceTransformer not available")
#             return

#         try:
#             # 영구 저장소 사용 (메모리 대신)
#             self.client = QdrantClient(path=persist_path)
#             self.ok = True
#             print(f"[INFO] Qdrant client initialized with persist path: {persist_path}")
#         except Exception as e:
#             print(f"[WARN] Qdrant connection failed: {e}")
#             self.ok = False

#         self.collection = "meeting_ctx"
#         self.embed_dim = 2048
#         self.model = None

#         if self.ok:
#             try:
#                 self.model = SentenceTransformer(
#                     "dragonkue/BGE-m3-ko",
#                     device="cuda" if torch.cuda.is_available() else "cpu"
#                 ) # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#                 self.embed_dim = self.model.get_sentence_embedding_dimension()

#                 # 컬렉션 생성 (이미 존재하면 재사용)
#                 collections = self.client.get_collections().collections
#                 if not any(c.name == self.collection for c in collections):
#                     self.client.create_collection(
#                         self.collection,
#                         vectors_config=VectorParams(
#                             size=self.embed_dim, distance=Distance.COSINE
#                         ),
#                     )
#                     print(f"[INFO] Created collection: {self.collection}")
#                 else:
#                     print(f"[INFO] Using existing collection: {self.collection}")

#             except Exception as e:
#                 print(f"[WARN] create collection failed: {e}")
#                 self.ok = False

#         self._id_seq = 1

#     def _embed(self, texts: List[str]) -> List[List[float]]:
#         """텍스트를 임베딩 벡터로 변환"""
#         if self.model is None:
#             # Fallback: 간단한 해시 기반 임베딩
#             vecs = []
#             for t in texts:
#                 v = np.zeros(256, dtype=np.float32)
#                 for i, ch in enumerate(t.encode("utf-8")):
#                     v[i % 256] += (ch % 13) / 13.0
#                 n = np.linalg.norm(v) + 1e-9
#                 vecs.append((v / n).tolist())
#             return vecs

#         em = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
#         return em.tolist() if isinstance(em, np.ndarray) else em

#     def upsert_segments(self, segs: List) -> int:
#         """
#         발언 세그먼트를 벡터DB에 저장

#         Args:
#             segs: Segment 객체 또는 Dict 리스트

#         Returns:
#             저장된 세그먼트 수
#         """
#         if not self.ok:
#             print("[WARN] RAG store not initialized")
#             return 0

#         payloads = []
#         texts = []
#         ids = []

#         for s in segs:
#             # Segment 객체 또는 Dict 처리
#             if hasattr(s, 'speaker_name'):
#                 speaker_name = s.speaker_name
#                 speaker_id = s.speaker_id
#                 text = s.text
#                 start = s.start
#                 end = s.end
#             else:
#                 speaker_name = s.get('speaker_name', s.get('speaker', 'Unknown'))
#                 speaker_id = s.get('speaker_id', s.get('speaker', 'Unknown'))
#                 text = s.get('text', '')
#                 start = s.get('start', 0.0)
#                 end = s.get('end', 0.0)

#             if not text.strip():
#                 continue

#             texts.append(f"[{speaker_name}] {text}")
#             ids.append(self._id_seq)
#             payloads.append({
#                 "speaker_id": speaker_id,
#                 "speaker_name": speaker_name,
#                 "start": float(start),
#                 "end": float(end),
#                 "text": text,
#                 "timestamp": datetime.now().isoformat(),
#             })
#             self._id_seq += 1

#         if not texts:
#             return 0

#         vecs = self._embed(texts)
#         points = [
#             PointStruct(id=i, vector=v, payload=p)
#             for i, v, p in zip(ids, vecs, payloads)
#         ]

#         try:
#             self.client.upsert(self.collection, points=points)
#             print(f"[INFO] Upserted {len(points)} segments to RAG")
#             return len(points)
#         except Exception as e:
#             print(f"[WARN] RAG upsert failed: {e}")
#             return 0

#     def search(
#         self,
#         query: str,
#         topk: int = 5,
#         speaker_id: Optional[str] = None,
#         speaker_name: Optional[str] = None,
#         time_range: Optional[tuple] = None
#     ) -> List[Dict]:
#         """
#         발언 검색

#         Args:
#             query: 검색 쿼리
#             topk: 반환할 결과 수
#             speaker_id: 특정 화자 ID 필터링 (선택)
#             speaker_name: 특정 화자 이름 필터링 (선택)
#             time_range: (start_timestamp, end_timestamp) 시간 범위 (선택)

#         Returns:
#             검색 결과 리스트 (payload + score)
#         """
#         if not self.ok:
#             return []

#         vec = self._embed([query])[0]

#         # 필터 조건 구성
#         filter_conditions = []

#         if speaker_id:
#             filter_conditions.append(
#                 FieldCondition(
#                     key="speaker_id",
#                     match=MatchValue(value=speaker_id)
#                 )
#             )

#         if speaker_name:
#             filter_conditions.append(
#                 FieldCondition(
#                     key="speaker_name",
#                     match=MatchValue(value=speaker_name)
#                 )
#             )

#         if time_range:
#             start_time, end_time = time_range
#             filter_conditions.append(
#                 FieldCondition(
#                     key="timestamp",
#                     range=Range(
#                         gte=start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time,
#                         lte=end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time
#                     )
#                 )
#             )

#         # 필터 적용
#         query_filter = None
#         if filter_conditions:
#             query_filter = Filter(must=filter_conditions)

#         try:
#             res = self.client.search(
#                 self.collection,
#                 query_vector=vec,
#                 limit=topk,
#                 query_filter=query_filter
#             )

#             out = []
#             for r in res:
#                 pl = r.payload or {}
#                 pl["_score"] = r.score
#                 out.append(pl)

#             return out

#         except Exception as e:
#             print(f"[WARN] RAG search failed: {e}")
#             return []

#     def search_by_speaker(self, speaker_id: str, query: str = "", topk: int = 10) -> List[Dict]:
#         """
#         특정 화자의 발언 검색

#         Args:
#             speaker_id: 화자 ID
#             query: 검색 쿼리 (비어있으면 모든 발언)
#             topk: 반환할 결과 수

#         Returns:
#             화자의 발언 리스트
#         """
#         if not query:
#             # 쿼리 없이 화자의 모든 발언 가져오기
#             query = speaker_id  # 화자 ID로 검색

#         return self.search(query=query, topk=topk, speaker_id=speaker_id)

#     def get_speaker_stats(self, speaker_id: str) -> Dict:
#         """
#         화자의 통계 정보

#         Args:
#             speaker_id: 화자 ID

#         Returns:
#             통계 정보 (발언 수, 평균 길이 등)
#         """
#         if not self.ok:
#             return {}

#         try:
#             # 화자의 모든 발언 가져오기
#             results = self.search_by_speaker(speaker_id, query=speaker_id, topk=1000)

#             if not results:
#                 return {
#                     "speaker_id": speaker_id,
#                     "total_utterances": 0,
#                     "avg_length": 0,
#                     "total_duration": 0,
#                 }

#             texts = [r["text"] for r in results]
#             durations = [r["end"] - r["start"] for r in results]

#             return {
#                 "speaker_id": speaker_id,
#                 "speaker_name": results[0].get("speaker_name", speaker_id),
#                 "total_utterances": len(results),
#                 "avg_length": sum(len(t) for t in texts) / len(texts),
#                 "total_duration": sum(durations),
#                 "avg_duration": sum(durations) / len(durations),
#             }

#         except Exception as e:
#             print(f"[WARN] Failed to get speaker stats: {e}")
#             return {}

#     def get_all_speakers(self) -> List[str]:
#         """
#         저장된 모든 화자 ID 목록

#         Returns:
#             화자 ID 리스트
#         """
#         if not self.ok:
#             return []

#         try:
#             # Scroll을 사용해 모든 포인트 가져오기
#             records, _ = self.client.scroll(
#                 collection_name=self.collection,
#                 limit=10000,
#                 with_payload=True,
#                 with_vectors=False
#             )

#             speakers = set()
#             for record in records:
#                 if record.payload and "speaker_id" in record.payload:
#                     speakers.add(record.payload["speaker_id"])

#             return sorted(list(speakers))

#         except Exception as e:
#             print(f"[WARN] Failed to get all speakers: {e}")
#             return []

#     def clear_collection(self):
#         """컬렉션 초기화"""
#         if not self.ok:
#             return False

#         try:
#             self.client.delete_collection(self.collection)
#             self.client.create_collection(
#                 self.collection,
#                 vectors_config=VectorParams(
#                     size=self.embed_dim, distance=Distance.COSINE
#                 ),
#             )
#             self._id_seq = 1
#             print(f"[INFO] Cleared collection: {self.collection}")
#             return True
#         except Exception as e:
#             print(f"[WARN] Failed to clear collection: {e}")
#             return False
# core/rag_store.py
# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import torch

# ----- 선택 의존성 (있으면 사용) -----
try:
    import fitz  # PyMuPDF for PDF
except Exception:
    fitz = None

try:
    import docx  # python-docx for DOCX
except Exception:
    docx = None

# ----- 프로젝트 내부 의존 (없으면 안전하게 처리) -----
try:
    from .audio import Segment  # 회의 세그먼트 타입 (선택)
except ImportError:
    Segment = None

# ----- Qdrant / Embedding 모델 -----
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct,
        Filter, FieldCondition, MatchValue, Range
    )
    from sentence_transformers import SentenceTransformer
    Q_OK = True
except Exception:
    Q_OK = False


class RagStore:
    """
    회의 세그먼트 + 문서 기반 RAG를 '한 클래스'로 처리.
    - 기존 기능 그대로 유지(세그먼트 upsert/search 등)
    - 추가: 문서 업로드(.txt/.md/.docx/.pdf/.hwpx) → 청크 → 업서트
    - 추가: 문서 검색(search_docs), 챗봇 컨텍스트(build_rag_prompt), 근거 로깅(save_hits_log)
    """

    def __init__(self, persist_path: str = "./qdrant_db"):
        self.ok = False
        self.persist_path = persist_path

        if not Q_OK:
            print("[WARN] Qdrant client or SentenceTransformer not available")
            return

        try:
            # 로컬 파일 기반 영구 저장 (서버 없이 동작)
            self.client = QdrantClient(path=persist_path)
            self.ok = True
            print(f"[INFO] Qdrant client initialized with persist path: {persist_path}")
        except Exception as e:
            print(f"[WARN] Qdrant connection failed: {e}")
            self.ok = False

        # 기본 컬렉션(회의 세그먼트)
        self.collection = "meeting_ctx"

        # 문서용 컬렉션(업로드 문서 RAG)
        self.doc_collection = "project_docs"

        self.embed_dim = 2048
        self.model = None

        if self.ok:
            try:
                # 한국어/다국어 문서 모두 대응 가능한 멀티링구얼 모델
                self.model = SentenceTransformer(
                    "dragonkue/BGE-m3-ko",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.embed_dim = self.model.get_sentence_embedding_dimension()

                # 회의 세그먼트 컬렉션 준비
                self._ensure_collection(self.collection, self.embed_dim)
                # 문서 컬렉션 준비
                self._ensure_collection(self.doc_collection, self.embed_dim)

            except Exception as e:
                print(f"[WARN] create collection failed: {e}")
                self.ok = False

        self._id_seq = 1
        # 근거 로그 파일
        self._hits_log = os.path.join("data", "rag", "last_hits.jsonl")

    # -----------------------------
    # 내부 유틸
    # -----------------------------
    def _ensure_collection(self, name: str, dim: int):
        """컬렉션 없으면 생성, 있으면 재사용"""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == name for c in collections):
                self.client.create_collection(
                    name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                print(f"[INFO] Created collection: {name}")
            else:
                print(f"[INFO] Using existing collection: {name}")
        except Exception as e:
            print(f"[WARN] ensure_collection failed({name}): {e}")
            raise

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트 → 임베딩. 모델 없으면 간이 해시 벡터 사용(디버그용)."""
        if self.model is None:
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

    # -----------------------------
    # (A) 회의 세그먼트 저장/검색 (기존)
    # -----------------------------
    def upsert_segments(self, segs: List) -> int:
        """
        회의 발언 세그먼트를 meeting_ctx 컬렉션에 저장.
        segs: Segment 객체 또는 dict 리스트
        """
        if not self.ok:
            print("[WARN] RAG store not initialized")
            return 0

        payloads, texts, ids = [], [], []

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
                "_type": "segment",
            })
            self._id_seq += 1

        if not texts:
            return 0

        vecs = self._embed(texts)
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vecs, payloads)]

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
        """회의 발언 검색 (meeting_ctx 컬렉션 대상)"""
        if not self.ok:
            return []

        vec = self._embed([query])[0]

        # 필터 구성
        filter_conditions = []
        if speaker_id:
            filter_conditions.append(FieldCondition(key="speaker_id", match=MatchValue(value=speaker_id)))
        if speaker_name:
            filter_conditions.append(FieldCondition(key="speaker_name", match=MatchValue(value=speaker_name)))
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
        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            res = self.client.search(self.collection, query_vector=vec, limit=topk, query_filter=query_filter)
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
        """특정 화자 발언 검색(회의 컬렉션)"""
        if not query:
            query = speaker_id
        return self.search(query=query, topk=topk, speaker_id=speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """특정 화자 통계(회의 컬렉션)"""
        if not self.ok:
            return {}
        try:
            results = self.search_by_speaker(speaker_id, query=speaker_id, topk=1000)
            if not results:
                return {"speaker_id": speaker_id, "total_utterances": 0, "avg_length": 0, "total_duration": 0}
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
        """저장된 모든 화자 ID 목록(회의 컬렉션)"""
        if not self.ok:
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            speakers = {rec.payload.get("speaker_id") for rec in records if rec.payload and "speaker_id" in rec.payload}
            return sorted([s for s in speakers if s])
        except Exception as e:
            print(f"[WARN] Failed to get all speakers: {e}")
            return []

    def clear_collection(self):
        """회의 컬렉션 초기화"""
        if not self.ok:
            return False
        try:
            self.client.delete_collection(self.collection)
            self._ensure_collection(self.collection, self.embed_dim)
            self._id_seq = 1
            print(f"[INFO] Cleared collection: {self.collection}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to clear collection: {e}")
            return False

    # -----------------------------
    # (B) 문서 업로드/검색 (추가)
    # -----------------------------
    @staticmethod
    def _clean_text(t: str) -> str:
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip()

    def _read_docx(self, path: str) -> str:
        if docx:
            try:
                return "\n".join(p.text for p in docx.Document(path).paragraphs)
            except Exception:
                pass
        # fallback: 간소화 (문단 분리 보존 어려울 수 있음)
        return open(path, "rb").read().decode("utf-8", errors="ignore")

    def _read_pdf(self, path: str) -> str:
        if not fitz:
            raise RuntimeError("PyMuPDF(fitz) 미설치: pip install pymupdf")
        doc = fitz.open(path)
        return "\n".join(p.get_text("text") for p in doc)

    def _read_hwpx(self, path: str) -> str:
        # 매우 단순한 HWPX 스크랩(필요 시 개선)
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            xml = z.read("Contents/section0.xml")
        root = ET.fromstring(xml)
        texts = [n.text.strip() for n in root.iter() if n.text and n.text.strip()]
        return " ".join(texts)

    def _read_any(self, path: str) -> Tuple[str, str]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":  return "pdf",  self._read_pdf(path)
        if ext == ".docx": return "docx", self._read_docx(path)
        if ext in [".txt", ".md"]:
            return ext[1:], open(path, "r", encoding="utf-8", errors="ignore").read()
        if ext == ".hwpx": return "hwp",  self._read_hwpx(path)
        raise RuntimeError(f"지원하지 않는 확장자: {ext}")

    def _chunk(self, text: str, max_len: int = 900, overlap: int = 150) -> List[str]:
        text = self._clean_text(text)
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i+max_len])
            i += max_len - overlap
        return out

    def upsert_document_files(self, file_paths: List[str], project: str = "Persona-AI", tags: Optional[List[str]] = None) -> int:
        """
        문서 파일들을 읽어 청크로 나눈 뒤 Qdrant(project_docs 컬렉션)에 업서트.
        반환: 업서트된 청크 수
        """
        if not self.ok:
            print("[WARN] RAG store not initialized")
            return 0
        tags = tags or ["uploaded"]

        payloads, texts, ids = [], [], []
        for fp in file_paths:
            if not os.path.isfile(fp):
                continue
            try:
                stype, raw = self._read_any(fp)
            except Exception as e:
                print(f"[WARN] read failed: {fp} ({e})")
                continue
            chunks = self._chunk(raw)
            title = os.path.basename(fp)
            for ci, ch in enumerate(chunks):
                texts.append(ch)
                ids.append(int(datetime.now().timestamp()*1000) + len(ids))
                payloads.append({
                    "project": project,
                    "source_type": stype,
                    "source_path": os.path.abspath(fp),
                    "title": title,
                    "chunk": ch,
                    "chunk_id": str(ci),
                    "lang": "ko",
                    "tags": tags,
                    "_type": "doc",
                })

        if not texts:
            return 0

        vecs = self._embed(texts)
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vecs, payloads)]
        try:
            self.client.upsert(self.doc_collection, points=points)
            print(f"[INFO] Upserted {len(points)} doc-chunks to {self.doc_collection}")
            return len(points)
        except Exception as e:
            print(f"[WARN] Doc upsert failed: {e}")
            return 0

    def upsert_document_dir(self, dir_path: str, exts: Tuple[str, ...] = (".txt", ".md", ".docx", ".pdf", ".hwpx")) -> int:
        """디렉터리 전체를 스캔해 문서 업서트"""
        files = []
        for root, _, fs in os.walk(dir_path):
            for f in fs:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(root, f))
        return self.upsert_document_files(files)

    def search_docs(self, query: str, topk: int = 5, project: Optional[str] = None, tag: Optional[str] = None) -> List[Dict]:
        """
        업로드 문서 검색(project_docs 컬렉션)
        - project / tag 기반 필터 제공
        """
        if not self.ok:
            return []
        vec = self._embed([query])[0]
        conds = []
        if project:
            conds.append(FieldCondition(key="project", match=MatchValue(value=project)))
        if tag:
            conds.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
        qf = Filter(must=conds) if conds else None

        try:
            res = self.client.search(self.doc_collection, query_vector=vec, limit=topk, query_filter=qf)
            hits = []
            for rank, r in enumerate(res, 1):
                pl = r.payload or {}
                pl["_score"] = r.score
                pl["_rank"] = rank
                hits.append(pl)
            return hits
        except Exception as e:
            print(f"[WARN] Doc search failed: {e}")
            return []

    # -----------------------------
    # (C) 챗봇 컨텍스트 & 근거 로깅 (추가)
    # -----------------------------
    def build_rag_prompt(self, query: str, topk: int = 5, project: Optional[str] = None) -> Dict:
        """
        챗봇용 프롬프트(system/user)와 히트(hits)를 생성.
        - 업로드 문서(doc) 컬렉션을 우선 사용
        """
        hits = self.search_docs(query, topk=topk, project=project)
        context = "\n\n---\n\n".join(
            f"[{h.get('title')} / {h.get('source_path')}] (rank={h.get('_rank')}, score={h.get('_score'):.3f})\n{h.get('chunk')}"
            for h in hits
        )
        system = (
            "다음 <컨텍스트>는 업로드 문서에서 검색된 근거입니다. "
            "반드시 컨텍스트를 우선하여 한국어로 답변하고, 근거가 없으면 '문서 근거 없음'이라고 명시하세요."
        )
        user = f"질문: {query}\n\n<컨텍스트 시작>\n{context}\n<컨텍스트 끝>"
        return {"system": system, "user": user, "hits": hits}

    def save_hits_log(self, query: str, hits: List[Dict]):
        """최근 RAG 근거를 로그 파일로 저장 (UI에서 '근거 보기' 용)"""
        try:
            os.makedirs(os.path.dirname(self._hits_log), exist_ok=True)
            row = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "query": query, "hits": hits}
            with open(self._hits_log, "a", encoding="utf-8") as w:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] save_hits_log failed: {e}")
