# config.py
# 프로젝트 전체 설정 관리

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import torch

@dataclass
class AudioConfig:
    """오디오 처리 관련 설정"""
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2  # 16-bit
    CHUNK_SIZE: int = 1024
    BUFFER_DURATION: float = 30.0
    PROCESS_INTERVAL: float = 10.0
    MIN_SEG_DUR: float = 0.35
    OVERLAP_DURATION: float = 5.0

@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # Whisper
    WHISPER_MODEL: str = "medium"
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    
    # Pyannote
    PYANNOTE_MODEL_PATH: Path = Path("./models/diart_model")
    PYANNOTE_PIPELINE_NAME: str = "pyannote/speaker-diarization-3.1"
    HF_TOKEN_ENV: str = "HF_TOKEN"
    
    # Embedding
    EMBEDDING_MODEL: str = "dragonkue/BGE-m3-ko"

@dataclass
class StorageConfig:
    """저장소 관련 설정"""
    OUTPUT_DIR: Path = Path("./output")
    QDRANT_PATH: Path = Path("./qdrant_storage")
    SPEAKER_PROFILES_PATH: Path = Path("./speaker_profiles.pkl")
    COLLECTION_NAME: str = "meeting_embeddings"

@dataclass
class SpeakerConfig:
    """화자 인식 관련 설정"""
    SIMILARITY_THRESHOLD: float = 0.75
    MAX_SPEAKER_EMBEDDINGS: int = 10
    SPEAKER_TIMEOUT: float = 3600.0
    
@dataclass
class UIConfig:
    """UI 테마 설정"""
    THEME: Dict[str, str] = None
    
    def __post_init__(self):
        if self.THEME is None:
            self.THEME = {
                "bg": "#e6f5e6",
                "pane": "#99cc99", 
                "light_bg": "#fafffa",
                "btn": "#ffe066",
                "btn_hover": "#ffdb4d",
                "btn_border": "#cccc99",
            }

@dataclass
class AppConfig:
    """애플리케이션 전체 설정"""
    audio: AudioConfig = None
    model: ModelConfig = None
    storage: StorageConfig = None
    speaker: SpeakerConfig = None
    ui: UIConfig = None
    
    def __post_init__(self):
        self.audio = self.audio or AudioConfig()
        self.model = self.model or ModelConfig()
        self.storage = self.storage or StorageConfig()
        self.speaker = self.speaker or SpeakerConfig()
        self.ui = self.ui or UIConfig()
        
        # 디렉토리 생성
        self.storage.OUTPUT_DIR.mkdir(exist_ok=True)
        self.storage.QDRANT_PATH.mkdir(exist_ok=True)

# 전역 설정 인스턴스
config = AppConfig()

# 독립 실행 테스트
if __name__ == "__main__":
    import json
    
    print("=" * 50)
    print("Config Module Test")
    print("=" * 50)
    
    # 설정 정보 출력
    print("📁 Storage Configuration:")
    print(f"  - Output Directory: {config.storage.OUTPUT_DIR}")
    print(f"  - Qdrant Path: {config.storage.QDRANT_PATH}")
    print(f"  - Speaker Profiles: {config.storage.SPEAKER_PROFILES_PATH}")
    
    print("\n🎤 Audio Configuration:")
    print(f"  - Sample Rate: {config.audio.SAMPLE_RATE}Hz")
    print(f"  - Channels: {config.audio.CHANNELS}")
    print(f"  - Buffer Duration: {config.audio.BUFFER_DURATION}s")
    print(f"  - Process Interval: {config.audio.PROCESS_INTERVAL}s")
    
    print("\n🤖 Model Configuration:")
    print(f"  - Whisper Model: {config.model.WHISPER_MODEL}")
    print(f"  - Device: {config.model.WHISPER_DEVICE}")
    print(f"  - Compute Type: {config.model.WHISPER_COMPUTE_TYPE}")
    print(f"  - Language: {config.model.WHISPER_LANG}")
    print(f"  - Embedding Model: {config.model.EMBEDDING_MODEL}")
    
    print("\n👥 Speaker Configuration:")
    print(f"  - Similarity Threshold: {config.speaker.SIMILARITY_THRESHOLD}")
    print(f"  - Max Embeddings: {config.speaker.MAX_SPEAKER_EMBEDDINGS}")
    print(f"  - Timeout: {config.speaker.SPEAKER_TIMEOUT}s")
    
    print("\n🎨 UI Configuration:")
    for key, value in config.ui.THEME.items():
        print(f"  - {key}: {value}")
    
    # 디렉토리 생성 확인
    print(f"\n📂 Directory Creation:")
    print(f"  - Output dir exists: {config.storage.OUTPUT_DIR.exists()}")
    print(f"  - Qdrant dir exists: {config.storage.QDRANT_PATH.exists()}")
    
    # 설정을 JSON으로 저장 테스트
    try:
        config_dict = {
            "audio": {
                "sample_rate": config.audio.SAMPLE_RATE,
                "channels": config.audio.CHANNELS,
                "buffer_duration": config.audio.BUFFER_DURATION
            },
            "model": {
                "whisper_model": config.model.WHISPER_MODEL,
                "device": config.model.WHISPER_DEVICE,
                "embedding_model": config.model.EMBEDDING_MODEL
            }
        }
        
        with open(config.storage.OUTPUT_DIR / "config_test.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"\n✅ Configuration test file saved: config_test.json")
    except Exception as e:
        print(f"\n❌ Configuration save failed: {e}")
    
    print("\n" + "=" * 50)
    print("Config Module Test Complete!")