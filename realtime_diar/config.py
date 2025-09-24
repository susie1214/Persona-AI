# realtime_diar/config.py
from pathlib import Path
import torch
import pyaudio
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for real-time diarization and transcription."""
    PYANNOTE_MODEL_PATH: Path = Path("./models/diart_model")
    WHISPER_MODEL_PATH: Path = Path("./models/whisper-small-ct2")
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    # Audio
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    CHANNELS: int = 1
    AUDIO_FORMAT: int = pyaudio.paInt16
    # Processing
    BUFFER_DURATION: float = 30.0
    PROCESS_INTERVAL: float = 10.0
    MIN_SEG_DUR: float = 0.35
    OVERLAP_DURATION: float = 5.0
