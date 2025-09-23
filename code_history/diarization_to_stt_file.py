import os
import librosa
import numpy as np
from datetime import timedelta
import yaml
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    """Configuration for the diarization and transcription script."""
    AUDIO_FILE: Path = Path("./output/meeting_merged.wav")
    PYANNOTE_MODEL_PATH: Path = Path("D:/Persona-AI/models/diart_model")
    WHISPER_MODEL_PATH: Path = Path("D:/Persona-AI/models/whisper-small-ct2")
    
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    
    MIN_SEG_DUR: float = 0.35

def hhmmss(seconds: float) -> str:
    """Converts seconds to HH:MM:SS format."""
    return str(timedelta(seconds=round(seconds)))

def load_diarization_pipeline(config: Config) -> SpeakerDiarization:
    """Loads the pyannote speaker diarization pipeline."""
    print("[INFO] Loading local Pyannote model...")
    segmentation_model = Model.from_pretrained(config.PYANNOTE_MODEL_PATH / "segmentation-3.0" / "pytorch_model.bin")
    embedding_model = Model.from_pretrained(config.PYANNOTE_MODEL_PATH / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin")

    with open(config.PYANNOTE_MODEL_PATH / "config.yaml", "r") as f:
        pipeline_config = yaml.safe_load(f)

    diar_pipeline = SpeakerDiarization(
        segmentation=segmentation_model,
        embedding=embedding_model,
    )
    diar_pipeline.instantiate(pipeline_config['params'])
    return diar_pipeline

def load_whisper_model(config: Config) -> WhisperModel:
    """Loads the faster-whisper model."""
    print("[INFO] Loading Whisper model...")
    return WhisperModel(
        str(config.WHISPER_MODEL_PATH),
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE
    )

def process_audio(diar_pipeline: SpeakerDiarization, whisper_model: WhisperModel, config: Config):
    """Diarizes and transcribes the audio file."""
    print(f"[INFO] Loading audio file: {config.AUDIO_FILE}")
    wav, sr = librosa.load(config.AUDIO_FILE, sr=16000, mono=True)
    print(f"[INFO] Audio loaded. Duration: {len(wav)/sr:.2f}s")

    print("[INFO] Diarizing speakers...")
    diar_result = diar_pipeline(str(config.AUDIO_FILE))

    print("[INFO] Transcribing speaker segments...")
    for turn, _, speaker in diar_result.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        if end - start < config.MIN_SEG_DUR:
            continue

        i0 = int(start * sr)
        i1 = int(end * sr)
        segment_wav = wav[i0:i1].copy()
        if segment_wav.size == 0:
            continue

        segments, _ = whisper_model.transcribe(
            segment_wav.astype(np.float32),
            language=config.WHISPER_LANG,
            vad_filter=True,
            beam_size=1,
            word_timestamps=False
        )
        text = " ".join([s.text.strip() for s in segments if s.text])

        print(f"[{hhmmss(start)} - {hhmmss(end)}] {speaker}: {text}")

def main():
    """Main function to run the diarization and transcription process."""
    config = Config()
    
    try:
        diar_pipeline = load_diarization_pipeline(config)
        whisper_model = load_whisper_model(config)
        process_audio(diar_pipeline, whisper_model, config)
        print("[INFO] Done.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()