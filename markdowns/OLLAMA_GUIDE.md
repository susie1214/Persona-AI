# Ollama Python 패키지 사용 가이드

`ollama` Python 패키지를 사용하여 Ollama 모델과 통신하는 방법입니다.

## 1. 설치

### Ollama 설치 (필수)
먼저 Ollama를 설치해야 합니다:
- **Windows/Mac/Linux**: https://ollama.com/download

### Python 패키지 설치
```bash
pip install ollama
```

## 2. Ollama 모델 다운로드

Ollama를 실행한 후 원하는 모델을 다운로드합니다:

```bash
# Llama 3 (8B) - 추천
ollama pull llama3

# Llama 3.1 (8B)
ollama pull llama3.1

# Llama 3 (70B) - 대용량
ollama pull llama3:70b

# Mistral (7B)
ollama pull mistral

# Gemma 2 (9B)
ollama pull gemma2

# 한국어 특화 모델
ollama pull llama3-korean
```

### 사용 가능한 모델 확인
```bash
ollama list
```

## 3. Ollama 서버 실행

Ollama는 백그라운드에서 자동 실행되지만, 수동으로 실행할 수도 있습니다:

```bash
# 서버 실행
ollama serve

# 특정 포트로 실행
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

## 4. 코드에서 사용

### 기본 사용법

```python
from core.llm_router import LLMRouter

# Llama3 모델 사용
router = LLMRouter(default_backend="ollama:llama3")

# 텍스트 생성
response = router.complete(
    backend="ollama:llama3",
    prompt="한국의 수도는 어디인가요?",
    temperature=0.7
)
print(response)
```

### 다양한 모델 사용

```python
# Mistral 모델
response = router.complete(
    backend="ollama:mistral",
    prompt="Explain quantum computing",
    temperature=0.3
)

# Gemma2 모델
response = router.complete(
    backend="ollama:gemma2",
    prompt="Write a Python function",
    temperature=0.0
)

# Llama3 70B (대용량)
response = router.complete(
    backend="ollama:llama3:70b",
    prompt="Create a business plan",
    temperature=0.8
)
```

### 커스텀 호스트 지정

```python
# 원격 Ollama 서버 사용
backend = "ollama:llama3?host=http://192.168.1.100:11434"
response = router.complete(backend=backend, prompt="Hello!")
```

## 5. 페르소나별 모델 설정

기존 코드와 호환됩니다:

```python
# survey_wizard.py 또는 chat_dock.py에서
persona_config = {
    "조진경": {
        "backend": "ollama:llama3",
        "temperature": 0.7,
        "style": "professional"
    },
    "신우택": {
        "backend": "ollama:mistral",
        "temperature": 0.5,
        "style": "friendly"
    }
}
```

## 6. 채팅 형식 사용 (선택)

OllamaLLM 클래스는 채팅 형식도 지원합니다:

```python
from core.llm_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

# 채팅 형식
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = llm.chat(messages, temperature=0.7)
print(response)
```

## 7. 모델별 특징 및 추천 사용처

### Llama 3 (8B)
- **크기**: 4.7GB
- **용도**: 범용 작업
- **장점**: 빠른 속도, 좋은 품질
- **추천**: 일반 채팅, 요약, Q&A

### Llama 3.1 (8B)
- **크기**: 4.7GB
- **용도**: 최신 버전, 향상된 성능
- **장점**: Llama 3보다 개선된 추론
- **추천**: 복잡한 추론, 코드 생성

### Llama 3 (70B)
- **크기**: 40GB
- **용도**: 고급 작업
- **장점**: 최고 품질
- **추천**: 전문적인 문서 작성, 복잡한 분석
- **주의**: GPU 필요 (최소 48GB VRAM)

### Mistral (7B)
- **크기**: 4.1GB
- **용도**: 효율적인 추론
- **장점**: 빠른 속도, 낮은 메모리
- **추천**: 실시간 응답, 리소스 제한 환경

### Gemma 2 (9B)
- **크기**: 5.4GB
- **용도**: Google의 최신 모델
- **장점**: 균형잡힌 성능
- **추천**: 다양한 언어 작업

## 8. 성능 최적화

### GPU 가속
Ollama는 자동으로 GPU를 감지하여 사용합니다:

```bash
# GPU 사용 확인
ollama ps

# CUDA 설정 (Linux)
export CUDA_VISIBLE_DEVICES=0
ollama serve
```

### 병렬 요청
여러 모델을 동시에 로드할 수 있습니다:

```python
# 모델 캐싱으로 빠른 응답
router = LLMRouter()

# 첫 번째 요청 (모델 로딩)
r1 = router.complete("ollama:llama3", "Question 1")

# 두 번째 요청 (캐시된 모델 사용)
r2 = router.complete("ollama:llama3", "Question 2")  # 빠름!
```

### 컨텍스트 길이 조정
```bash
# 더 긴 컨텍스트 허용 (메모리 증가)
ollama pull llama3
ollama run llama3 --ctx-size 8192
```

## 9. 환경 변수 설정

```bash
# .env 파일에 추가
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODELS_PATH=/path/to/models
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=2
```

## 10. 트러블슈팅

### 문제: "ollama package not installed"
```bash
pip install ollama
```

### 문제: "Ollama 서버에 연결할 수 없습니다"
```bash
# Ollama 서버 실행 확인
ollama list

# 수동 실행
ollama serve
```

### 문제: "모델을 찾을 수 없습니다"
```bash
# 모델 다운로드
ollama pull llama3

# 다운로드된 모델 확인
ollama list
```

### 문제: "응답이 너무 느림"
- GPU 드라이버 업데이트
- 더 작은 모델 사용 (70B → 8B)
- `num_parallel` 설정 조정

### 문제: "메모리 부족"
```bash
# 더 작은 모델 사용
ollama pull llama3  # 대신 7B/8B 모델

# 또는 양자화된 모델
ollama pull llama3:7b-q4_0
```

## 11. 예제: 실전 사용

### 회의록 요약
```python
from core.llm_router import LLMRouter

router = LLMRouter(default_backend="ollama:llama3")

meeting_transcript = """
[김철수] 프로젝트 진행 상황을 공유드리겠습니다.
[이영희] 네, 잘 들었습니다. 다음 단계는 무엇인가요?
"""

summary_prompt = f"""
다음 회의 내용을 요약해주세요:

{meeting_transcript}

요약:
"""

summary = router.complete(
    backend="ollama:llama3",
    prompt=summary_prompt,
    temperature=0.3
)
print(summary)
```

### 페르소나 응답 생성
```python
persona_prompt = f"""
당신은 친절한 AI 어시스턴트 '조진경'입니다.
사용자의 질문에 전문적이고 정중하게 답변하세요.

사용자 질문: {user_question}

조진경의 답변:
"""

response = router.complete(
    backend="ollama:llama3",
    prompt=persona_prompt,
    temperature=0.7
)
```

### 코드 생성
```python
code_prompt = """
Python으로 피보나치 수열을 생성하는 함수를 작성하세요.
재귀 방식과 반복 방식 두 가지를 모두 구현하세요.
"""

code = router.complete(
    backend="ollama:llama3.1",  # 코드 생성에 최적화
    prompt=code_prompt,
    temperature=0.0  # 결정적 출력
)
print(code)
```

## 12. Ollama vs OpenAI 비교

| 항목 | Ollama | OpenAI |
|------|--------|--------|
| 비용 | 무료 (로컬) | 유료 (API 요금) |
| 속도 | GPU 의존 | 일관되게 빠름 |
| 프라이버시 | 완벽 (로컬) | 데이터 전송 |
| 모델 선택 | 제한적 | 다양함 |
| 설정 난이도 | 중간 | 쉬움 |
| 인터넷 필요 | X | O |

## 13. 추천 설정

### 개발 환경
```python
# 빠른 응답, 로컬 테스트
router = LLMRouter(default_backend="ollama:llama3")
```

### 프로덕션 환경
```python
# 안정성 우선
router = LLMRouter(default_backend="ollama:llama3.1")
# 또는 OpenAI 백업
# router = LLMRouter(default_backend="openai:gpt-4o-mini")
```

### 고성능 서버
```python
# 대용량 모델, GPU 가속
router = LLMRouter(default_backend="ollama:llama3:70b")
```

---

**참고 자료:**
- Ollama 공식 문서: https://ollama.com
- Ollama Python 라이브러리: https://github.com/ollama/ollama-python
- 모델 라이브러리: https://ollama.com/library
