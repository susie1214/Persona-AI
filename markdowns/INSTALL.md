# Persona-AI 설치 가이드

## 플랫폼별 설치 방법

### Windows

1. **CUDA 설치** (GPU 사용 시)
   ```bash
   # CUDA 12.8 설치
   # https://developer.nvidia.com/cuda-downloads
   ```

2. **Python 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 변수 설정**
   ```bash
   # HuggingFace 토큰 설정
   set HF_TOKEN=your_huggingface_token
   ```

---

### macOS

1. **Homebrew 설치** (없는 경우)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **PortAudio 설치** (선택사항 - PyAudio 사용 시)
   ```bash
   brew install portaudio
   ```

3. **Python 패키지 설치**
   ```bash
   pip install -r requirements_macos.txt
   ```

4. **환경 변수 설정**
   ```bash
   # HuggingFace 토큰 설정
   export HF_TOKEN=your_huggingface_token

   # .zshrc 또는 .bash_profile에 추가
   echo 'export HF_TOKEN=your_huggingface_token' >> ~/.zshrc
   ```

---

### Linux

1. **시스템 패키지 설치**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install portaudio19-dev python3-pyaudio

   # Fedora/RHEL
   sudo dnf install portaudio-devel
   ```

2. **CUDA 설치** (GPU 사용 시)
   ```bash
   # NVIDIA CUDA Toolkit 설치
   # https://developer.nvidia.com/cuda-downloads
   ```

3. **Python 패키지 설치**
   ```bash
   # GPU 있는 경우
   pip install -r requirements.txt

   # GPU 없는 경우 (macOS용 사용)
   pip install -r requirements_macos.txt
   ```

4. **환경 변수 설정**
   ```bash
   export HF_TOKEN=your_huggingface_token
   echo 'export HF_TOKEN=your_huggingface_token' >> ~/.bashrc
   ```

---

## 오디오 백엔드 선택

프로그램은 자동으로 사용 가능한 오디오 라이브러리를 감지합니다:

- **Windows**: PyAudio (기본)
- **macOS**: sounddevice (권장) 또는 PyAudio
- **Linux**: PyAudio 또는 sounddevice

### 오디오 테스트

```python
# 사용 가능한 오디오 백엔드 확인
python -c "
try:
    import pyaudio
    print('PyAudio: 사용 가능')
except:
    print('PyAudio: 사용 불가')

try:
    import sounddevice
    print('sounddevice: 사용 가능')
except:
    print('sounddevice: 사용 불가')
"
```

---

## GPU 지원

### Windows/Linux (NVIDIA GPU)
- CUDA 12.8 지원
- faster-whisper가 자동으로 GPU 사용

### macOS (Apple Silicon)
- MPS 백엔드 감지됨
- faster-whisper는 MPS 미지원으로 CPU 사용
- PyTorch 모델은 MPS 가속 가능

### GPU 테스트

```python
import torch

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"MPS 사용 가능: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"사용 중인 디바이스: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 실행

```bash
# GUI 실행
python run_gui.py

# 또는
python -m ui.meeting_console
```

---

## 문제 해결

### macOS: PyAudio 설치 실패
```bash
# portaudio 설치 후 재시도
brew install portaudio
pip install pyaudio

# 또는 sounddevice 사용
pip install sounddevice
```

### macOS: SSL 인증서 오류
```bash
# Python 인증서 설치
/Applications/Python\ 3.x/Install\ Certificates.command
```

### Windows: CUDA 인식 안됨
1. NVIDIA 드라이버 최신 버전 설치
2. CUDA Toolkit 12.8 설치
3. `nvcc --version` 명령으로 확인

### Linux: 마이크 권한 오류
```bash
# 사용자를 audio 그룹에 추가
sudo usermod -a -G audio $USER
# 로그아웃 후 재로그인
```

---

## HuggingFace 토큰 발급

1. https://huggingface.co/ 가입
2. Settings → Access Tokens → New token 생성
3. 환경 변수에 설정 (위 참조)

---

## 의존성 요약

| 항목 | Windows | macOS | Linux |
|------|---------|-------|-------|
| Python | 3.8+ | 3.8+ | 3.8+ |
| PyAudio | ✅ | ⚠️ (portaudio 필요) | ✅ |
| sounddevice | ✅ | ✅ 권장 | ✅ |
| CUDA | ✅ (GPU) | ❌ | ✅ (GPU) |
| MPS | ❌ | ✅ (M1/M2/M3) | ❌ |
