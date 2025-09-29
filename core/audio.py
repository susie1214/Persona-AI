# -*- coding: utf-8 -*-
# core/audio.py
import io, os, tempfile, threading, time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np, soundfile as sf

try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

from PySide6.QtCore import QObject, Signal

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_SECONDS = 6


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str = "Unknown"
    speaker_name: str = "Unknown"


@dataclass
class MeetingState:
    live_segments: List[Segment] = field(default_factory=list)
    diar_segments: List[Tuple[float, float, str]] = field(default_factory=list)
    speaker_map: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    actions: List[str] = field(default_factory=list)
    schedule_note: str = ""
    diarization_enabled: bool = False
    use_gpu: bool = True
    asr_model: str = "medium"
    raw_audio_path: str = ""
    audio_time_elapsed: float = 0.0


def now_str():
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


class AudioWorker(QObject):
    sig_transcript = Signal(object)  # Segment
    sig_status = Signal(str)

    def __init__(self, state: MeetingState):
        super().__init__()
        self.state = state
        self._stop = threading.Event()
        self.audio = None
        self.stream = None
        self._buf = io.BytesIO()
        self._buf_lock = threading.Lock()
        self._frames_elapsed = 0
        self.model = None

        # ▼ 추가: 선택된 입력 디바이스 인덱스 (None이면 기본장치)
        self._input_device_index: Optional[int] = None

        # (선택) 임시폴더를 프로젝트 내부로 고정하고 싶다면 주석 해제
        # os.makedirs("output/tmp", exist_ok=True)
        # tempfile.tempdir = os.path.abspath("output/tmp")

    # ====== Public API ======
    def set_input_device_index(self, idx: Optional[int]):
        """UI에서 마이크 인덱스를 전달 (None=기본장치)"""
        self._input_device_index = idx

    # ====== ASR ======
    def init_asr(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper 미설치")
        device = "cuda" if self.state.use_gpu else "cpu"
        compute = "float16" if self.state.use_gpu else "int8"
        self.sig_status.emit(f"Loading Whisper '{self.state.asr_model}' on {device}...")
        try:
            self.model = WhisperModel(
                self.state.asr_model, device=device, compute_type=compute
            )
        except Exception as e:
            self.sig_status.emit(f"ASR GPU 실패 -> CPU 재시도 ({e})")
            self.model = WhisperModel(
                self.state.asr_model, device="cpu", compute_type="int8"
            )
        self.sig_status.emit("ASR model ready.")

    # ====== Lifecycle ======
    def start(self):
        if pyaudio is None:
            raise RuntimeError("PyAudio 미설치")
        self._stop.clear()
        self.init_asr()

        # rolling wav 파일 준비
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="raw_meeting_")
        os.close(fd)
        self.state.raw_audio_path = path
        sf.write(
            path,
            np.zeros((1, CHANNELS), dtype=np.float32),
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        # PyAudio 오픈
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                output=False,  # ★ 출력 장치 강제 미사용 (-9996 회피)
                input_device_index=self._input_device_index,  # ★ 선택 마이크(없으면 기본)
                frames_per_buffer=int(SAMPLE_RATE * 0.2),  # 200ms
                stream_callback=self._on_audio,
            )
        except Exception as e:
            # 입력 장치 인덱스가 문제면 기본장치로 재시도
            if self._input_device_index is not None:
                self.sig_status.emit(f"Input device {self._input_device_index} 실패 -> 기본장치 재시도")
                self.stream = self.audio.open(
                    format=self.audio.get_format_from_width(SAMPLE_WIDTH),
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=False,
                    frames_per_buffer=int(SAMPLE_RATE * 0.2),
                    stream_callback=self._on_audio,
                )
            else:
                raise
        self.stream.start_stream()
        threading.Thread(target=self._chunk_loop, daemon=True).start()
        self.sig_status.emit("Audio capture started.")

    def stop(self):
        self._stop.set()
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except Exception:
            pass
        self.sig_status.emit("Audio capture stopped.")

    # ====== Audio callback / buffering ======
    def _on_audio(self, in_data, frame_count, time_info, status):
        # 버퍼 누적
        with self._buf_lock:
            self._buf.write(in_data)
            self._frames_elapsed += frame_count
            self.state.audio_time_elapsed += frame_count / SAMPLE_RATE

        # rolling wav append (데모용 간단 구현)
        data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        data_np = data_np.reshape(-1, CHANNELS) if CHANNELS == 1 else data_np
        try:
            existing, sr = sf.read(
                self.state.raw_audio_path, dtype="float32", always_2d=True
            )
            sf.write(
                self.state.raw_audio_path,
                np.vstack([existing, data_np]),
                SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception as e:
            self.sig_status.emit(f"WAV append fail: {e}")

        return (None, pyaudio.paContinue)

    def _pull_chunk_wav(self) -> Optional[bytes]:
        # CHUNK_SECONDS 만큼의 PCM을 잘라서 WAV 바이트로 반환
        with self._buf_lock:
            seconds = self._frames_elapsed / SAMPLE_RATE
            if seconds < CHUNK_SECONDS:
                return None
            need_frames = int(CHUNK_SECONDS * SAMPLE_RATE)
            raw = self._buf.getvalue()
            need_bytes = need_frames * SAMPLE_WIDTH * CHANNELS
            if len(raw) < need_bytes:
                return None
            chunk = raw[:need_bytes]
            remain = raw[need_bytes:]
            self._buf = io.BytesIO()
            self._buf.write(remain)
            self._frames_elapsed -= need_frames

        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = audio_np.reshape(-1, CHANNELS)
        mem = io.BytesIO()
        sf.write(mem, audio_np, SAMPLE_RATE, format="WAV")
        return mem.getvalue()

    # ====== Speaker mapping (approx) ======
    def _infer_speaker_id(self, s: float, e: float) -> str:
        overlaps = [
            spk for (ds, de, spk, _) in self.state.diar_segments if not (e < ds or s > de)
        ]
        if not overlaps:
            return "Unknown"
        return max(set(overlaps), key=overlaps.count)

    # ====== STT loop ======
    def _chunk_loop(self):
        chunk_offset_seconds = 0.0
        while not self._stop.is_set():
            wav_bytes = self._pull_chunk_wav()
            if wav_bytes is None:
                time.sleep(0.1)
                continue

            path = None
            try:
                # Windows 안전: 핸들을 닫고 경로만 써서 권한 문제 방지
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="chunk_")
                os.close(fd)
                with open(path, "wb") as f:
                    f.write(wav_bytes)

                segments, info = self.model.transcribe(
                    path, language="ko", vad_filter=True
                )
                for s in segments:
                    seg = Segment(
                        start=chunk_offset_seconds + s.start,
                        end=chunk_offset_seconds + s.end,
                        text=(s.text or "").strip(),
                    )
                    spk_id = self._infer_speaker_id(seg.start, seg.end)
                    seg.speaker_id = spk_id
                    seg.speaker_name = self.state.speaker_map.get(spk_id, spk_id)
                    self.sig_transcript.emit(seg)

                chunk_offset_seconds += CHUNK_SECONDS

            except Exception as e:
                self.sig_status.emit(f"STT error: {e}")
                time.sleep(0.2)
            finally:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
