# core/rag_store.py
import numpy as np
from typing import List, Dict
from .audio import Segment

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from sentence_transformers import SentenceTransformer

    Q_OK = True
except Exception:
    Q_OK = False


class RagStore:
    def __init__(self):
        self.ok = False
        if not Q_OK:
            return
        try:
            self.client = QdrantClient(":memory:")
            self.ok = True
        except Exception:
            try:
                self.client = QdrantClient(host="127.0.0.1", port=6333)
                self.ok = True
            except Exception as e:
                print("[WARN] Qdrant connection failed:", e)
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
                self.client.recreate_collection(
                    self.collection,
                    vectors_config=VectorParams(
                        size=self.embed_dim, distance=Distance.COSINE
                    ),
                )
            except Exception as e:
                print("[WARN] create collection failed:", e)
                self.ok = False
        self._id_seq = 1

    def _embed(self, texts: List[str]) -> List[List[float]]:
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

    def upsert_segments(self, segs: List[Segment]):
        if not self.ok:
            return
        payloads = []
        texts = []
        ids = []
        for s in segs:
            texts.append(f"[{s.speaker_name}] {s.text}")
            ids.append(self._id_seq)
            payloads.append(
                {
                    "speaker_id": s.speaker_id,
                    "speaker_name": s.speaker_name,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
            )
            self._id_seq += 1
        vecs = self._embed(texts)
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vecs, payloads)
        ]
        try:
            self.client.upsert(self.collection, points=points)
        except Exception as e:
            print("[WARN] RAG upsert failed:", e)

    def search(self, query: str, topk=5) -> List[Dict]:
        if not self.ok:
            return []
        vec = self._embed([query])[0]
        try:
            res = self.client.search(self.collection, query_vector=vec, limit=topk)
            out = []
            for r in res:
                pl = r.payload or {}
                pl["_score"] = r.score
                out.append(pl)
            return out
        except Exception as e:
            print("[WARN] RAG search failed:", e)
            return []
