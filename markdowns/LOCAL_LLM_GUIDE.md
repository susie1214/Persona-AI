# 로컬 LLM 모델 사용 가이드

Ollama 서버 없이 GGUF 포맷 모델을 직접 로드하여 사용하는 방법입니다.

## 1. 패키지 설치

### CPU 버전 (기본)
```bash
pip install llama-cpp-python
```

### GPU 버전 (CUDA 지원)
```bash
# Windows/Linux
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# macOS (Metal 지원)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

자세한 설치 방법: https://github.com/abetlen/llama-cpp-python

## 2. 모델 다운로드

GGUF 포맷 모델을 다운로드합니다. 추천 소스:

### 추천 모델 (HuggingFace)

**한국어 지원 모델:**
- **EEVE-Korean-10.8B** (추천)
  - https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0-GGUF
  - 파일: `EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf` (6.7GB)

- **Llama-3-Korean-8B**
  - https://huggingface.co/beomi/Llama-3-Open-Ko-8B-GGUF
  - 파일: `Llama-3-Open-Ko-8B-Q4_K_M.gguf` (4.9GB)

**영어 모델:**
- **Llama 3.1 8B**
  - https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  - 파일: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.9GB)

- **Mistral 7B**
  - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  - 파일: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (4.4GB)

### 모델 저장 위치
```bash
# 프로젝트 루트에 models 폴더 생성
mkdir models
cd models

# 예: EEVE 모델 다운로드
wget https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0-GGUF/resolve/main/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf
```

## 3. 설정 방법

### 방법 1: 기본 사용 (코드에서)

```python
from core.llm_router import LLMRouter

# 기본 설정 (CPU, 4096 컨텍스트)
router = LLMRouter(default_backend="local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf")

# 모델 사용
response = router.complete(
    backend="local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf",
    prompt="안녕하세요, 자기소개 해주세요.",
    temperature=0.7
)
print(response)
```

### 방법 2: GPU 가속 사용

```python
# GPU 레이어 지정 (NVIDIA GPU 필요)
backend = "local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf?gpu_layers=35"

# 전체 모델을 GPU로 (-1 = 모든 레이어)
backend = "local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf?gpu_layers=-1"
```

### 방법 3: 고급 파라미터 설정

```python
# 파라미터 조합
# - gpu_layers: GPU 레이어 수 (0=CPU only, -1=모두)
# - ctx: 컨텍스트 길이 (기본값: 4096)
# - threads: CPU 스레드 수

backend = "local:models/llama-3.1-8b.gguf?gpu_layers=35&ctx=8192&threads=8"

router = LLMRouter(default_backend=backend)
```

### 방법 4: 페르소나별 모델 지정

```python
# survey_wizard.py 또는 chat_dock.py에서
persona_config = {
    "조진경": {
        "backend": "local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf?gpu_layers=20",
        "prompt_style": "formal"
    },
    "신우택": {
        "backend": "openai:gpt-4o-mini",  # OpenAI 사용 가능
        "prompt_style": "casual"
    }
}
```

## 4. 성능 최적화

### GPU 메모리별 추천 설정

| GPU VRAM | 모델 크기 | gpu_layers | 예상 속도 |
|----------|----------|------------|-----------|
| 4GB | 7B Q4 | 15-20 | 보통 |
| 6GB | 7B Q4 | 25-30 | 빠름 |
| 8GB | 7B Q4 | 35 (전체) | 매우 빠름 |
| 8GB | 10B Q4 | 20-25 | 보통 |
| 12GB+ | 10B Q4 | 35 (전체) | 매우 빠름 |

### CPU 전용 최적화

```python
# CPU 스레드 수 조정 (물리 코어 수 추천)
backend = "local:models/model.gguf?threads=8"

# 작은 컨텍스트로 메모리 절약
backend = "local:models/model.gguf?ctx=2048&threads=8"
```

## 5. 모델 양자화 레벨 선택

GGUF 파일명에 포함된 양자화 레벨:

- **Q4_K_M** (추천): 균형잡힌 품질/크기 (약 4.5GB for 7B)
- **Q5_K_M**: 더 높은 품질 (약 5.5GB for 7B)
- **Q8_0**: 최고 품질 (약 7.5GB for 7B)
- **Q3_K_M**: 메모리 절약 (약 3.5GB for 7B, 품질 저하)

## 6. 기존 Ollama 서버 방식과 비교

### 장점
✅ 서버 실행 불필요 (독립 실행)
✅ 모델 로딩이 한 번만 발생 (캐싱)
✅ 세밀한 파라미터 제어
✅ GPU 메모리 최적화
✅ Python 환경에서 완전 통합

### 단점
❌ 첫 로딩 시간 (10-30초, 모델 크기에 따라)
❌ 메모리 사용량 증가 (프로세스 내 로딩)
❌ 멀티모델 동시 사용 시 메모리 부담

## 7. 트러블슈팅

### 문제: "llama-cpp-python not installed"
```bash
pip install llama-cpp-python
```

### 문제: "Model file not found"
- 모델 파일 경로 확인
- 절대 경로 사용 권장: `local:D:/models/model.gguf`

### 문제: GPU 인식 안 됨
```bash
# CUDA 버전 재설치
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### 문제: 메모리 부족
- `gpu_layers` 값을 낮춤 (예: 35 → 20)
- `ctx` 값을 낮춤 (예: 8192 → 2048)
- 더 작은 양자화 모델 사용 (Q3_K_M)

### 문제: 응답이 너무 느림
- `gpu_layers`를 -1로 설정 (전체 GPU 사용)
- CPU 스레드 수 증가
- 더 작은 모델 사용 (7B 대신 3B)

## 8. 예제 코드

### 간단한 테스트
```python
from core.llm_ollama import OllamaLLM

# 모델 로드
llm = OllamaLLM(
    model_path="models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf",
    n_gpu_layers=35,  # GPU 사용
    n_ctx=4096,
    verbose=True
)

# 질문
response = llm.complete(
    "한국의 수도는 어디인가요?",
    temperature=0.3
)
print(response)
```

### 채팅봇 통합
```python
# chat_dock.py에서
router = LLMRouter(
    default_backend="local:models/EEVE-Korean-10.8B-v1.0-Q4_K_M.gguf?gpu_layers=35&ctx=8192"
)

# 페르소나 응답 생성
answer = router.complete(
    backend=persona.llm_backend,
    prompt=prompt_template.format(question=user_query),
    temperature=0.7
)
```

## 9. 추천 워크플로우

1. **모델 다운로드**: EEVE-Korean-10.8B (한국어) 또는 Llama-3.1-8B (영어)
2. **GPU 테스트**: `gpu_layers=-1`로 전체 GPU 사용 시도
3. **메모리 조정**: VRAM 부족 시 `gpu_layers` 점진적 감소
4. **성능 측정**: 응답 시간 확인 후 최적값 설정
5. **프로덕션 배포**: 안정적인 설정으로 고정

---

**참고 자료:**
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python
- GGUF 모델 검색: https://huggingface.co/models?library=gguf
- 한국어 LLM 리더보드: https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard
