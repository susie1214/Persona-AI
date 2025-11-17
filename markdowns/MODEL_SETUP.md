# 모델 설정 가이드

## 📁 모델 저장 위치

모든 로컬 모델은 **D 드라이브**에 저장됩니다:
```
D:/models/
├── skt_A.X-4.0/              # A.X-4.0 모델
└── K-intelligence_Midm-2.0-Mini-Instruct/  # Midm-2.0 모델
```

이 경로는 `config.py`에서 중앙 관리됩니다.

## 🔧 경로 변경 방법

모델 저장 위치를 변경하려면 `config.py` 파일의 `LOCAL_MODELS_DIR` 변수를 수정하세요:

```python
# config.py
LOCAL_MODELS_DIR = Path("D:/models")  # 원하는 경로로 변경
```

## 📦 지원 모델

### 1. OpenAI (gpt-4o-mini)
- **요구사항**: OPENAI_API_KEY
- **설정 위치**: `.env` 파일
- **사용법**: 채팅에서 "openai:gpt-4o-mini" 선택

### 2. Ollama (llama3)
- **요구사항**: Ollama 로컬 서버 실행
- **설치**: [ollama.ai](https://ollama.ai) 참고
- **서버 실행**: `ollama serve`
- **사용법**: 채팅에서 "ollama:llama3" 선택

### 3. A.X-4.0 (SKT)
- **모델 크기**: ~28GB
- **요구사항**: HF_TOKEN (최초 다운로드 시)
- **저장 위치**: `D:/models/skt_A.X-4.0/`
- **사용법**: 채팅에서 "ax:A.X-4.0" 선택

### 4. Midm-2.0-Mini-Instruct
- **모델 크기**: ~7GB
- **요구사항**: 없음 (공개 모델)
- **저장 위치**: `D:/models/K-intelligence_Midm-2.0-Mini-Instruct/`
- **사용법**: 채팅에서 "midm:Midm-2.0-Mini-Instruct" 선택

## 🚀 모델 다운로드

### 자동 다운로드 (권장)
앱을 실행하고 채팅에서 모델을 선택하면 자동으로 다운로드됩니다:
```bash
python app.py
# 채팅에서 원하는 백엔드 선택 → 자동 다운로드 시작
```

### 수동 다운로드
미리 다운로드하려면:
```bash
python download_models.py
```
대화형 메뉴가 나타나면:
- `1`: A.X-4.0만 다운로드
- `2`: Midm-2.0만 다운로드
- `3`: 모두 다운로드

## 🔐 환경 변수 설정

`.env` 파일 작성:
```bash
# HuggingFace 토큰 (A.X-4.0 다운로드 시 필요)
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# OpenAI API 키 (OpenAI 모델 사용 시 필요)
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxx"
```

**주의**: 등호 앞뒤에 공백을 넣지 마세요!

## ✅ 설치 확인

### 1. 디렉토리 생성 확인
```bash
python config.py
```
출력 예시:
```
📋 현재 설정:
  로컬 모델 디렉토리: D:\models
  A.X-4.0 경로: D:\models\skt_A.X-4.0
  Midm-2.0 경로: D:\models\K-intelligence_Midm-2.0-Mini-Instruct

[INFO] Directories initialized:
  - Models: D:\models
  - Qdrant: .\qdrant_storage
  - QLoRA: .\adapters
```

### 2. 모델 테스트
```bash
python test_models.py
```
각 모델을 개별적으로 테스트하여 정상 작동 여부를 확인합니다.

## 🐛 문제 해결

### 토큰 오류
```
❌ HuggingFace 토큰 오류
```
**해결**: `.env` 파일에 `HF_TOKEN` 설정 확인

### 모델 없음 오류
```
❌ 모델을 찾을 수 없습니다
```
**해결**: `python download_models.py` 실행하여 모델 다운로드

### 연결 오류 (Ollama)
```
❌ 연결 오류
```
**해결**: Ollama 서버 실행 확인 (`ollama serve`)

### OpenAI API 키 오류
```
❌ OpenAI API 키 오류
```
**해결**: `.env` 파일에 `OPENAI_API_KEY` 설정 확인

## 💾 디스크 공간 요구사항

- **A.X-4.0**: 약 28GB
- **Midm-2.0**: 약 7GB
- **Qdrant 데이터**: 가변 (회의 데이터에 따라)
- **QLoRA 어댑터**: 가변 (화자당 ~1GB)

**권장**: D 드라이브에 최소 50GB 여유 공간

## 🔄 모델 업데이트

로컬 모델을 재다운로드하려면:
1. 기존 모델 폴더 삭제: `D:/models/skt_A.X-4.0/`
2. `python download_models.py` 재실행
3. 또는 앱에서 해당 백엔드 선택 시 자동 재다운로드

## 📚 추가 리소스

- **HuggingFace 토큰 생성**: https://huggingface.co/settings/tokens
- **Ollama 설치**: https://ollama.ai
- **A.X-4.0 모델 페이지**: https://huggingface.co/skt/A.X-4.0
- **Midm-2.0 모델 페이지**: https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct
