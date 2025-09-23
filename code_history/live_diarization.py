# diarize_stt_qdrant.py
import os
import sys
import platform
import threading
import traceback
import uuid
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import librosa

from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter, _extract_prediction

from faster_whisper import WhisperModel

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# =========================
# 사용자 설정
# =========================
DEBUG = True

# "mic" 또는 "file"
MODE = "mic"

# 마이크 모드
DEVICE_ID: Optional[int] = None
OUT_DIR = "./outputs"
REC_SR = 16000                 # STT용 녹음/분석 기준 SR(Whisper 권장 16k)
TAIL_GUARD = 0.20              # 마이크에서 끝부분 보류 시간(초) — 버퍼 안정화

# 파일 모드
FILE_PATH = "./audio/sample.wav"
OUT_PATH: Optional[str] = None  # None이면 파일명 기반 .rttm
PIPELINE_SR: Optional[int] = None  # None이면 파이프라인 기본 SR

# Whisper (로컬 모델 경로 또는 모델명)
WHISPER_MODEL_PATH = "./models/whisper-small-ct2"  # CT2 디렉토리 또는 "small"/"medium"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"
WHISPER_LANG = "ko"

# Qdrant 설정 (임베딩은 placeholder)
QDRANT_URL = "./qdrant_db"
COLLECTION_NAME = "speakers"
EMBED_DIM = 1024  # placeholder 차원

# 출력
PRINT_TO_CONSOLE = True
WRITE_TXT = True
TXT_PATH = "./outputs/"   # None이면 RTTM 경로 기반 자동

MIN_SEG_DUR = 0.35  # 너무 짧은 구간 무시

# =========================
# 로깅/유틸
# =========================
def log_env():
    if DEBUG:
        dev = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        print(f"[DEBUG] OS: {platform.system()}")
        print(f"[DEBUG] Torch device available: {dev}")

def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def hhmmss(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600); m = int((x % 3600) // 60); s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# =========================
# 파이프라인/모델
# =========================
def build_pipeline() -> SpeakerDiarization:
    # diart 파이프라인은 .to(dev) 사용하지 않음 (내부에서 적절한 디바이스 사용)
    return SpeakerDiarization()

def build_whisper():
    try:
        print(f"[INFO] init whisper: model={WHISPER_MODEL_PATH}, device={WHISPER_DEVICE}, type={WHISPER_COMPUTE_TYPE}")
        return WhisperModel(WHISPER_MODEL_PATH, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    except Exception as e:
        print("[WARN] Whisper init failed → fallback to CPU int8:", e)
        return WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")

whisper = build_whisper()

# =========================
# Qdrant 준비
# =========================
qdrant_client = QdrantClient(path = QDRANT_URL)
if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            # "image": VectorParams(size=IMG_DIM, distance=Distance.COSINE),
    },
)

def embed_text(text: str):
    # TODO: 실제 임베딩 모델로 교체 (예: BGE-m3-ko)
    return np.random.rand(EMBED_DIM).astype(np.float32).tolist()

def add_to_qdrant(speaker_id: str, text: str):
    vector = embed_text(text)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "speaker_id": speaker_id,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
    if DEBUG:
        print(f"[DEBUG] Qdrant upsert: {speaker_id} -> {text[:60]}{'...' if len(text)>60 else ''}")

# =========================
# 마이크 STT용 버퍼
# =========================
class AudioBuffer:
    def __init__(self, sr: int):
        self.sr = sr
        self._data = np.zeros((0,), dtype=np.float32)

    def append(self, chunk: np.ndarray):
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)
        self._data = np.concatenate([self._data, chunk], axis=0)

    def duration(self) -> float:
        return len(self._data) / float(self.sr)

    def slice(self, start_s: float, end_s: float) -> np.ndarray:
        i0 = max(0, int(start_s * self.sr))
        i1 = min(len(self._data), int(end_s * self.sr))
        if i1 <= i0:
            return np.zeros((0,), dtype=np.float32)
        return self._data[i0:i1].copy()

class MicRecorder:
    def __init__(self, audio_buf: AudioBuffer, sr=16000, channels=1):
        self.audio_buf = audio_buf
        self.sr = sr
        self.channels = channels
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status and DEBUG:
            print(f"[DEBUG] sounddevice: {status}", file=sys.stderr)
        self.audio_buf.append(indata.copy())

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sr, channels=self.channels, dtype="float32", callback=self._callback, blocksize=0)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

# =========================
# 공용 STT+색인 옵저버
# =========================
speaker_data = {}  # {label: [{start, end, text}, ...]}

class STTAndIndexWriter:
    """
    StreamingInference 옵저버:
    - 새로 확정된 세그먼트를 받아, 오디오 슬라이스 → Whisper STT → 콘솔/파일 출력 → Qdrant 색인
    """
    def __init__(
        self,
        get_clip: Callable[[float, float], np.ndarray],  # (start,end)->wav(float32,16k)
        get_available_dur: Callable[[], float],         # 현재 사용 가능 길이(s)
        txt_path: Optional[str],
        prefix: str = "",
        tail_guard: float = 0.0,                        # 마이크에서 끝 보류
        language: Optional[str] = None,
    ):
        self.get_clip = get_clip
        self.get_available_dur = get_available_dur
        self.txt_path = txt_path
        self.prefix = prefix
        self.tail_guard = tail_guard
        self.language = language
        self.seen = set()
        if self.txt_path:
            ensure_parent(self.txt_path)
            if not os.path.exists(self.txt_path):
                with open(self.txt_path, "w", encoding="utf-8") as f:
                    f.write("# Diarization + STT Transcript\n")

    def _stt(self, wav: np.ndarray) -> str:
        if wav.size == 0:
            return ""
        segs, _ = whisper.transcribe(
            wav.astype(np.float32),
            language=self.language,
            vad_filter=True,
            beam_size=1,
            word_timestamps=False,
        )
        texts = [s.text.strip() for s in segs if s.text]
        return " ".join(texts).strip()

    def on_next(self, value):
        try:
            ann = _extract_prediction(value)  # Annotation or tuple→Annotation
            avail = self.get_available_dur()

            new_lines = []
            for segment, _, label in ann.itertracks(yield_label=True):
                start = float(segment.start); end = float(segment.end)
                if end - start < MIN_SEG_DUR:
                    continue
                if self.tail_guard > 0.0 and end > (avail - self.tail_guard):
                    # 마이크 끝부분은 버퍼가 더 모일 때까지 보류
                    continue

                key = (round(start, 3), round(end, 3), str(label))
                if key in self.seen:
                    continue
                self.seen.add(key)

                wav = self.get_clip(start, end)
                text = self._stt(wav)

                # 결과 저장/출력/색인
                speaker_data.setdefault(label, []).append({"start": start, "end": end, "text": text})
                add_to_qdrant(label, text)

                line = f"{self.prefix}[{hhmmss(start)}–{hhmmss(end)}] {label}"
                if text:
                    line += f": {text}"
                if PRINT_TO_CONSOLE:
                    print(line)
                new_lines.append(line + "\n")

            if new_lines and self.txt_path and WRITE_TXT:
                with open(self.txt_path, "a", encoding="utf-8") as f:
                    f.writelines(new_lines)

        except Exception as e:
            self.on_error(e)

    def on_error(self, error: Exception):
        msg = f"{self.prefix}Observer error: {error}"
        print(msg, file=sys.stderr)
        if self.txt_path:
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write(f"# ERROR: {error}\n")
        if DEBUG:
            traceback.print_exc()

    def on_completed(self):
        if DEBUG and self.txt_path:
            print(f"[DEBUG] Transcript saved to: {self.txt_path}")

    # 호환용
    def on_complete(self):
        self.on_completed()

# =========================
# 실행 함수
# =========================
def run_file_mode(file_path: str, out_path: Optional[str] = None, pipeline_sr: Optional[int] = None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    pipeline = build_pipeline()
    src_sr = pipeline_sr if pipeline_sr is not None else pipeline.config.sample_rate
    src = FileAudioSource(file_path, sample_rate=src_sr)

    # STT용 전체 오디오 로드(16k 변환)
    wav, in_sr = sf.read(file_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if in_sr != REC_SR:
        wav = librosa.resample(wav, orig_sr=in_sr, target_sr=REC_SR)

    def get_clip(start, end):
        i0 = max(0, int(start * REC_SR))
        i1 = min(len(wav), int(end * REC_SR))
        if i1 <= i0:
            return np.zeros((0,), dtype=np.float32)
        return wav[i0:i1].copy()

    def get_avail():
        return len(wav) / float(REC_SR)

    if out_path is None:
        stem, _ = os.path.splitext(file_path)
        out_path = stem + ".rttm"
    ensure_parent(out_path)
    txt_path = TXT_PATH or (os.path.splitext(out_path)[0] + ".txt")

    inf = StreamingInference(pipeline, src, do_plot=False)
    inf.attach_observers(RTTMWriter(src.uri, out_path))
    inf.attach_observers(STTAndIndexWriter(
        get_clip=get_clip,
        get_available_dur=get_avail,
        txt_path=txt_path,
        prefix="[FILE] ",
        tail_guard=0.0,
        language=WHISPER_LANG,
    ))

    if DEBUG:
        print(f"[DEBUG] File diarization+STT 시작: {file_path}, pipeline_sr={src_sr}, stt_sr={REC_SR}")
    inf()
    if DEBUG:
        print(f"[DEBUG] RTTM 저장: {out_path}")
        if WRITE_TXT:
            print(f"[DEBUG] TXT 저장:  {txt_path}")
    return out_path

def run_mic_mode(device_id: Optional[int] = None, out_dir: str = "./outputs"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"microphone_{ts}.rttm")
    txt_path = TXT_PATH or os.path.join(out_dir, f"microphone_{ts}.txt")

    # STT용 16k 마이크 버퍼
    audio_buf = AudioBuffer(sr=REC_SR)
    rec = MicRecorder(audio_buf, sr=REC_SR, channels=1)
    rec.start()

    pipeline = build_pipeline()
    # diart 입력 스트림(이쪽은 sample_rate 인자를 받지 않음)
    src = MicrophoneAudioSource(device=device_id)

    def get_clip(start, end):
        return audio_buf.slice(start, end)

    def get_avail():
        return audio_buf.duration()

    inf = StreamingInference(pipeline, src, do_plot=False)
    inf.attach_observers(RTTMWriter(src.uri, out_path))
    inf.attach_observers(STTAndIndexWriter(
        get_clip=get_clip,
        get_available_dur=get_avail,
        txt_path=txt_path,
        prefix="[MIC ] ",
        tail_guard=TAIL_GUARD,
        language=WHISPER_LANG,
    ))

    print("마이크 화자 분리 + STT 시작 (종료: Ctrl+C)")
    try:
        inf()  # Ctrl+C로 종료
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        rec.stop()
        print(f"RTTM 저장: {out_path}")
        if WRITE_TXT:
            print(f"TXT 저장:  {txt_path}")
    return out_path

def main():
    log_env()
    if MODE == "mic":
        run_mic_mode(device_id=DEVICE_ID, out_dir=OUT_DIR)
    elif MODE == "file":
        run_file_mode(FILE_PATH, out_path=OUT_PATH, pipeline_sr=PIPELINE_SR)
    else:
        raise ValueError('MODE 값이 잘못되었습니다. "mic" 또는 "file" 중 하나로 설정하세요.')

if __name__ == "__main__":
    main()