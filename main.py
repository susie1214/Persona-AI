# main.py
"""
실시간 화자분리 + 요약 GUI 애플리케이션

실행 방법:
python main.py

필요한 패키지:
- PyQt6
- pyaudio
- numpy
- torch
- pyannote.audio
- faster_whisper
- openai
- python-dotenv
- pyyaml

환경 설정:
1. .env 파일에 OPENAI_API_KEY 설정
2. models/ 디렉토리에 pyannote와 whisper 모델 배치
"""

import sys
import os
import onnxruntime as ort
from pathlib import Path

# 현재 스크립트의 디렉토리를 파이썬 패스에 추가
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def check_dependencies():
    """필수 의존성 검사"""
    required_packages = [
        'PyQt6',
        'pyaudio', 
        'numpy',
        'librosa',
        'soundfile',
        'torch',
        'pyannote.audio',
        'faster_whisper',
        'openai',
        'dotenv',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                import dotenv
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model_paths():
    """모델 경로 확인"""
    from config import Config
    
    config = Config()
    
    issues = []
    
    if not config.PYANNOTE_MODEL_PATH.exists():
        issues.append(f"Pyannote 모델 경로가 존재하지 않습니다: {config.PYANNOTE_MODEL_PATH}")
    
    if not config.WHISPER_MODEL_PATH.exists():
        issues.append(f"Whisper 모델 경로가 존재하지 않습니다: {config.WHISPER_MODEL_PATH}")
    
    if not config.api_key:
        issues.append("OpenAI API 키가 설정되지 않았습니다. (.env 파일에 OPENAI_API_KEY 설정)")
    
    if issues:
        print("설정 문제:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n※ 모델 경로 문제는 첫 실행 시 무시하고 진행할 수 있습니다.")
        return False
    
    return True

def main():
    """메인 함수"""
    CHECK = False
    
    if CHECK:
        print("실시간 화자분리 + 요약 GUI 시작 중...")
        
        # 의존성 검사
        print("1. 의존성 검사 중...")
        if not check_dependencies():
            print("필수 패키지가 설치되지 않았습니다. 설치 후 다시 실행하세요.")
            sys.exit(1)
        
        # 모델 및 설정 검사
        print("2. 모델 및 설정 검사 중...")
        check_model_paths()  # 경고만 표시하고 계속 진행
    
        # GUI 애플리케이션 시작
        print("3. GUI 시작 중...")
        
    try:
        from gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"애플리케이션 시작 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()