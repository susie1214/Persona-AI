# core/rag_pipeline_qdrant.py
# -*- coding: utf-8 -*-
"""
PyQt6 데스크톱 앱에서 쓰는 Qdrant RAG 파이프라인.
- 업로드된 문서 → 청크(JSONL) → 임베딩 → Qdrant 업로드/검색
- 지원 확장자: .txt .md .docx .pdf .hwpx (HWP는 HWPX로 저장 권장)
"""
from __future__ import annotations
import os, re, json, uuid, hashlib, zipfile, traceback
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

# 선택 의존성
try:
    import fitz  # PyMuPDF: PDF 텍스트
except Exception:
    fitz = None
try:
    import docx  # python-docx: DOCX 텍스트
except Exception:
    docx = None

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

DATA_DIR      = "data"
UPLOAD_DIR    = os.path.join(DATA_DIR, "docs", "uploaded")
RAG_JSONL     = os.path.join(DATA_DIR, "rag", "rag_all.jsonl")
COLLECTION    = "persona_ai_rag"
EMB_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RAG_JSONL), exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(text: str, max_len=900, overlap=150) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    i = 0; chunks=[]
    while i < len(text):
        chunks.append(text[i:i+max_len])
        i += max_len - overlap
    return chunks

def read_docx(path: str) -> str:
    if docx:
        try:
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        except Exception:
            pass
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    texts=[]
    for n in root.iter():
        if n.tag.endswith("}t"): texts.append(n.text or "")
        elif n.tag.endswith("}p"): texts.append("\n")
    return "".join(texts)

def read_pdf(path: str) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF(fitz) 필요: pip install pymupdf")
    doc = fitz.open(path)
    return "\n".join(p.get_text("text") for p in doc)

def read_hwpx(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("Contents/section0.xml")
    root = ET.fromstring(xml)
    texts=[n.text.strip() for n in root.iter() if n.text and n.text.strip()]
    return " ".join(texts)

def read_any(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext==".pdf":  return "pdf",  read_pdf(path)
    if ext==".docx": return "docx", read_docx(path)
    if ext in [".txt",".md"]:
        return ext[1:], open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext==".hwpx": return "hwp",  read_hwpx(path)
    raise RuntimeError(f"지원하지 않는 확장자: {ext}")

def ingest_file_to_rows(path: str, project="Persona-AI", lang="ko", tags=None) -> List[dict]:
    tags = tags or []
    stype, text = read_any(path)
    rows=[]
    for i, chunk in enumerate(chunk_text(text)):
        rows.append({
            "id": str(uuid.uuid4()),
            "project": project,
            "source_type": stype,
            "source_path": path,
            "title": os.path.basename(path),
            "chunk": chunk,
            "chunk_id": str(i),
            "hash": sha1(chunk),
            "lang": lang,
            "tags": tags,
            "meta": {}
        })
    return rows

def rebuild_jsonl(indir=UPLOAD_DIR, out_jsonl=RAG_JSONL) -> int:
    ensure_dirs()
    open(out_jsonl, "w", encoding="utf-8").close()
    cnt=0
    for root,_,files in os.walk(indir):
        for f in files:
            if f.lower().split(".")[-1] in ["pdf","docx","md","txt","hwpx"]:
                path = os.path.join(root, f)
                try:
                    rows = ingest_file_to_rows(path)
                    with open(out_jsonl, "a", encoding="utf-8") as w:
                        for r in rows: w.write(json.dumps(r, ensure_ascii=False)+"\n")
                    cnt += len(rows)
                except Exception:
                    print(f"[WARN] ingest 실패: {path}\n{traceback.format_exc()}")
    return cnt

class QdrantRAG:
    """Qdrant 업로드/검색 헬퍼 (로컬 qdrant 서버 필요: docker run -p 6333:6333 qdrant/qdrant)"""
    def __init__(self, host="localhost", port=6333, collection=COLLECTION, model_name=EMB_MODEL):
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name)
        self.collection = collection
        # 컬렉션 준비
        names = [c.name for c in self.client.get_collections().collections]
        if collection not in names:
            dim = self.model.get_sentence_embedding_dimension()
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert_jsonl(self, jsonl_path=RAG_JSONL) -> int:
        pts=[]
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                vec = self.model.encode(r["chunk"]).tolist()
                pts.append(PointStruct(id=r["id"], vector=vec, payload=r))
        if pts:
            self.client.upsert(collection_name=self.collection, points=pts)
        return len(pts)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        qv = self.model.encode(query).tolist()
        res = self.client.search(collection_name=self.collection, query_vector=qv, limit=k)
        hits=[]
        for rank, r in enumerate(res):
            p = r.payload
            hits.append({
                "rank": rank+1, "score": r.score,
                "title": p.get("title"), "chunk": p.get("chunk"),
                "source": p.get("source_path")
            })
        return hits
