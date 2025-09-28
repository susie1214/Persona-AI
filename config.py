import os
import torch
from pathlib import Path
from dataclasses import dataclass
import pyaudio
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # 모델 경로
    PYANNOTE_MODEL_PATH: Path = Path("./models/diart_model")
    WHISPER_MODEL_PATH: Path = Path("./models/whisper-small-ct2")
    
    # Whisper 설정
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    
    # 오디오 설정
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    CHANNELS: int = 1
    AUDIO_FORMAT: int = pyaudio.paInt16
    
    # 처리 설정
    BUFFER_DURATION: float = 30.0
    PROCESS_INTERVAL: float = 10.0
    MIN_SEG_DUR: float = 0.35
    OVERLAP_DURATION: float = 5.0
    
    # OpenAI API 키
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    def __post_init__(self):
        """설정 유효성 검사"""
        if not self.api_key:
            print("Warning: OpenAI API 키가 설정되지 않았습니다.")
        
        # 모델 경로 존재 확인
        if not self.PYANNOTE_MODEL_PATH.exists():
            print(f"Warning: Pyannote 모델 경로가 존재하지 않습니다: {self.PYANNOTE_MODEL_PATH}")
        
        if not self.WHISPER_MODEL_PATH.exists():
            print(f"Warning: Whisper 모델 경로가 존재하지 않습니다: {self.WHISPER_MODEL_PATH}")