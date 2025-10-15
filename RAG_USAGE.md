# RAG (Retrieval-Augmented Generation) 사용 가이드

## 개요

Persona-AI 프로젝트는 **회의 요약 문서**를 벡터 데이터베이스에 저장하고, 의미 기반 검색(Semantic Search)을 통해 관련 정보를 빠르게 찾을 수 있는 RAG 기능을 제공합니다.

## 🎯 데이터 흐름 설계

### RAG (회의록 검색용)
- **저장 대상**: 요약 문서, 액션 아이템, 회의록
- **용도**: 과거 회의 내용 검색, 의사결정 근거 추적
- **저장 시점**: "Summarize" 버튼 클릭 시 자동

### QLoRA (페르소나 학습용)
- **저장 대상**: 실시간 대화 원본 (`live_segments`)
- **용도**: 화자별 말투/성격 학습, 개인화된 AI 어시스턴트 생성
- **저장 위치**: 메모리 (`MeetingState.live_segments`)

> ⚠️ **중요**: 실시간 대화 원본은 RAG에 저장되지 않습니다. 요약 문서만 RAG에 저장되어 검색에 사용됩니다.

## 주요 기능

### 1. **회의 요약 문서 자동 인덱싱**
- 요약 생성 시 자동으로 RAG에 저장
- 회의 메타데이터 포함 (날짜, 참석자)
- 액션 아이템 개별 추적
- Qdrant 영구 저장소 (재시작 후에도 데이터 유지)
- Sentence-Transformers 기반 다국어 임베딩 (한국어 지원)

### 2. **회의록 검색 및 분석**
- 과거 회의 내용 의미 기반 검색
- 특정 주제에 대한 논의 이력 추적
- 액션 아이템 담당자별 검색
- 시간 범위별 회의 검색

### 3. **컨텍스트 기반 LLM 질의응답**
- RAG로 검색한 관련 회의록을 컨텍스트로 활용
- LLM이 과거 회의 근거를 바탕으로 답변 생성
- 출처 추적 (어떤 회의의 어떤 내용을 참조했는지 표시)

## 설치

RAG 기능을 사용하려면 다음 패키지가 필요합니다:

```bash
pip install qdrant-client sentence-transformers
```

## 파일 구조

```
Persona-AI/
├── core/
│   ├── rag_store.py              # RAG Store 핵심 구현
│   └── summarizer.py             # RAG 활용 함수들
├── data/
│   └── qdrant_db/                # 벡터 DB 영구 저장소 (요약 문서만)
├── example_persona_rag.py        # RAG 전체 워크플로우 예제
└── test_rag_speaker.py           # RAG 기능 테스트
```

## 사용 방법

### 1. UI에서 사용 (meeting_console.py)

#### 자동 초기화
회의 앱 실행 시 자동으로 RAG Store가 초기화됩니다:

```python
# 자동 초기화 (data/qdrant_db 경로)
self.rag = RagStore(persist_path="data/qdrant_db")
```

#### 요약 생성 및 RAG 저장
1. **"Start Recording"** 버튼으로 회의 시작
2. 실시간 대화가 `live_segments`에 저장됨 (QLoRA 학습용)
3. **"Summarize"** 버튼 클릭
   - 회의 요약 HTML 생성
   - 액션 아이템 추출
   - **자동으로 RAG에 저장** ✨

#### 테스트용 기능
**"Index to RAG"** 버튼은 개발/테스트 전용입니다. 실제 운영에서는 사용하지 않습니다.

### 2. 프로그래밍 방식으로 사용

#### 기본 사용 예제 (요약 문서 저장)

```python
from core.rag_store import RagStore

# RAG Store 초기화
rag = RagStore(persist_path="./data/qdrant_db")

# 요약 문서 저장
summary_doc = {
    "speaker_id": "SYSTEM",
    "speaker_name": "회의 요약",
    "text": "[2025-01-15] 프로젝트 킥오프 회의 - 참석자: 김태진, 이현택\n\n주요 논의사항: 데이터베이스 최적화...",
    "start": 0.0,
    "end": 0.0
}

rag.upsert_segments([summary_doc])

# 회의록 검색
results = rag.search("데이터베이스 최적화", topk=5)
for r in results:
    print(f"[{r['speaker_name']}] {r['text']} (score: {r['_score']:.3f})")
```

#### RAG + LLM 통합 사용

```python
from core.summarizer import llm_summarize_with_rag

# RAG 기반 질의응답
answer = llm_summarize_with_rag(
    query="데이터베이스 성능 개선에 대한 과거 논의 내용을 요약해주세요",
    rag_store=rag,
    topk=5
)
print(answer)
```

### 3. 전체 워크플로우 실행

```bash
# 예제 스크립트 실행 (테스트 데이터 사용)
python example_persona_rag.py

# RAG 기능 테스트
python test_rag_speaker.py
```

## API 레퍼런스

### RagStore 클래스

#### `__init__(persist_path: str = "./qdrant_db")`
- RAG Store 초기화
- **persist_path**: 벡터 DB 저장 경로

#### `upsert_segments(segs: List) -> int`
- 요약 문서를 벡터 DB에 저장
- **segs**: Segment 객체 또는 Dict 리스트
- **반환**: 저장된 세그먼트 수

#### `search(query: str, topk: int = 5, speaker_id: Optional[str] = None, time_range: Optional[tuple] = None) -> List[Dict]`
- 의미 기반 검색
- **query**: 검색 쿼리
- **topk**: 반환할 결과 수
- **speaker_id**: 특정 담당자 필터링 (선택)
- **time_range**: 시간 범위 필터링 (선택)
- **반환**: 검색 결과 리스트 (payload + score)

#### `get_all_speakers() -> List[str]`
- 저장된 모든 화자/담당자 ID 목록
- **반환**: ID 리스트 (SYSTEM, 담당자 이름 등)

#### `clear_collection()`
- 컬렉션 초기화 (모든 데이터 삭제)

### Summarizer 함수들

#### `llm_summarize_with_rag(query, rag_store, speaker_id=None, backend=None, topk=5) -> str`
- RAG 기반 LLM 요약 생성
- 관련 회의록을 자동으로 검색하여 LLM에 제공
- 출처 정보 포함

#### `get_speaker_context_summary(rag_store, speaker_id, topic=None, backend=None) -> Dict`
- 담당자의 액션 아이템 및 발언 패턴 AI 분석
- **반환**: 담당자 정보 + AI 분석 결과

## 실전 활용 사례

### 사례 1: 과거 회의에서 특정 주제 검색

```python
# "마케팅 전략"에 대한 과거 논의 검색
results = rag.search("마케팅 전략", topk=10)

for r in results:
    print(f"[{r['speaker_name']}]")
    print(f"{r['text']}\n")
```

### 사례 2: 액션 아이템 담당자별 검색

```python
# 특정 담당자의 액션 아이템 검색
results = rag.search("[액션아이템]", speaker_id="김태진", topk=20)

for r in results:
    print(f"- {r['text']}")
```

### 사례 3: AI 기반 회의록 질의응답

```python
from core.summarizer import llm_summarize_with_rag

# 과거 회의 내용 기반 답변 생성
answer = llm_summarize_with_rag(
    query="프로젝트 일정 지연 이슈에 대해 팀에서 어떤 대응책을 논의했나요?",
    rag_store=rag,
    topk=10
)

print(answer)
# AI가 관련 회의록을 검색하여 종합 답변 생성
# 출처(어떤 회의에서 논의되었는지)도 함께 제공
```

### 사례 4: 시간 범위 검색

```python
from datetime import datetime

# 특정 기간의 회의 검색
start = datetime(2025, 1, 1)
end = datetime(2025, 1, 31)

results = rag.search(
    query="프로젝트 진행상황",
    time_range=(start, end),
    topk=10
)
```

## 성능 및 최적화

### 임베딩 모델
- **기본 모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- **차원**: 384차원
- **특징**: 한국어 지원, 빠른 속도, 경량

### 저장소 크기
- 100개 회의 요약 ≈ 10-20MB
- 1,000개 회의 요약 ≈ 100-200MB

### 검색 속도
- 1,000개 문서 기준 < 50ms
- 10,000개 문서 기준 < 100ms

## 문제 해결

### RAG Store 초기화 실패
```
[WARN] Qdrant client or SentenceTransformer not available
```

**해결책**: 필수 패키지 설치
```bash
pip install qdrant-client sentence-transformers
```

### 검색 결과가 없음
- 요약 문서가 저장되었는지 확인 (Summarize 버튼 클릭했는지)
- `rag.get_all_speakers()`로 저장된 데이터 확인
- 쿼리 문구를 더 일반적으로 변경
- `topk` 값을 증가

### LLM 모듈 오류
```
⚠️ LLM 모듈을 사용할 수 없습니다
```

**해결책**: LLMRouter 및 필수 패키지 확인
- OpenAI API 키 설정 (.env 파일에 `OPENAI_API_KEY` 추가)
- `core/llm_router.py` 파일 존재 확인

## 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                   실시간 회의 진행                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   live_segments       │ ◄── 실시간 대화 저장
         │   (메모리)             │     (QLoRA 학습 데이터)
         └───────────┬───────────┘
                     │
                     │ "Summarize" 클릭
                     ▼
         ┌───────────────────────┐
         │  요약 문서 생성        │
         │  • 회의 요약           │
         │  • 액션 아이템         │
         └───────────┬───────────┘
                     │
                     │ 자동 저장
                     ▼
         ┌───────────────────────┐
         │   RAG Store           │ ◄── 요약 문서만 저장
         │   (Qdrant DB)         │     (검색 & 질의응답용)
         └───────────────────────┘
```

## QLoRA 어댑터와 통합

RAG로 검색한 회의록을 QLoRA 어댑터와 결합하여 개인화된 답변 생성:

```python
from core.adapter import AdapterManager
from core.summarizer import llm_summarize_with_rag

adapter_mgr = AdapterManager()
adapter_mgr.load_base('Qwen/Qwen2.5-3B-Instruct')
adapter_mgr.load_adapter('김태진', 'adapters/speaker_01/final')

# 1. RAG로 관련 회의록 검색
context = rag.search("데이터베이스 최적화", topk=3)

# 2. QLoRA 어댑터로 김태진 스타일의 답변 생성
response = adapter_mgr.respond_with_context(
    query="데이터베이스 최적화 방안을 제안해주세요",
    rag_context=context,
    speaker_id='speaker_01'
)

print(response)  # 김태진의 말투와 전문성을 반영한 답변
```

## 참고 자료

- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
- [Sentence Transformers 모델](https://www.sbert.net/docs/pretrained_models.html)
- [example_persona_rag.py](example_persona_rag.py) - 전체 워크플로우 예제
- [test_rag_speaker.py](test_rag_speaker.py) - 기능 테스트 코드

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
