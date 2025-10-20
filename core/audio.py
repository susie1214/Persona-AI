# -*- coding: utf-8 -*-
# core/audio.py
import io, os, tempfile, threading, time, sys, platform
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np, soundfile as sf

# Cross-platform audio library support
AUDIO_BACKEND = None
try:
    import pyaudio
    AUDIO_BACKEND = "pyaudio"
except Exception:
    # pyaudio = None
    pass

# Fallback to sounddevice for macOS/Linux
if AUDIO_BACKEND is None:
    try:
        import sounddevice as sd
        AUDIO_BACKEND = "sounddevice"
    except Exception:
        # sd = None
        # need logging for error at sound device
        pass 

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

from PySide6.QtCore import QObject, Signal
from core.speaker import SpeakerManager

# Detect platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

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
    diar_segments: List[Tuple[float, float, str, float]] = field(default_factory=list)  # (start, end, speaker_id, confidence)
    speaker_map: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    actions: List[str] = field(default_factory=list)
    schedule_note: str = ""
    diarization_enabled: bool = False
    use_gpu: bool = True
    asr_model: str = "medium"
    raw_audio_path: str = ""
    audio_time_elapsed: float = 0.0
    speaker_counter: int = 0  # 순차적 화자 ID 생성용 카운터


def now_str():
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


class AudioWorker(QObject):
    sig_transcript = Signal(object)  # Segment
    sig_status = Signal(str)
    sig_new_speaker_detected = Signal(str)  # 새 화자 감지 신호

    def __init__(
        self,
        state: MeetingState,
        speaker_manager: Optional[SpeakerManager] = None,
        persona_manager=None
    ):
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

        # SpeakerManager 통합 (임베딩 기반 화자 인식)
        self.speaker_manager = speaker_manager if speaker_manager else SpeakerManager()

        # DigitalPersonaManager 통합 (Phase 1)
        self.persona_manager = persona_manager

        # 화자 연속성 추적
        self._last_speaker_id = None
        self._last_speech_time = 0.0
        self._continuity_threshold = 5.0  # 5초 이내 발화는 같은 화자로 간주 (더 길게 조정)
        self._min_text_length = 3  # 최소 텍스트 길이

        # Embedding inference 모델 (lazy loading)
        self._embedding_inference = None

        # 녹음 관련 변수
        self._recording = False
        self._recording_path = None
        self._recording_frames = []
        self._recording_lock = threading.Lock()

    # ====== Public API ======
    def set_input_device_index(self, idx: Optional[int]):
        """UI에서 마이크 인덱스를 전달 (None=기본장치)"""
        self._input_device_index = idx

    def start_recording(self, output_path: str):
        """녹음 시작"""
        with self._recording_lock:
            if self._recording:
                return False
            self._recording = True
            self._recording_path = output_path
            self._recording_frames = []
        self.sig_status.emit(f"녹음 시작: {output_path}")
        return True

    def stop_recording(self) -> Optional[str]:
        """녹음 중지 및 파일 저장"""
        with self._recording_lock:
            if not self._recording:
                return None
            self._recording = False
            frames = self._recording_frames
            path = self._recording_path
            self._recording_frames = []
            self._recording_path = None

        if not frames or not path:
            return None

        try:
            # 녹음된 오디오를 WAV 파일로 저장
            audio_data = np.concatenate(frames, axis=0)
            sf.write(path, audio_data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            self.sig_status.emit(f"녹음 저장 완료: {path}")
            return path
        except Exception as e:
            self.sig_status.emit(f"녹음 저장 실패: {e}")
            return None

    def get_or_assign_speaker_id(self, original_speaker_id: str) -> str:
        """화자 ID를 받아서 순차적인 speaker_XX 형태로 변환/할당"""
        # 이미 매핑된 화자라면 그대로 반환
        if original_speaker_id in self.state.speaker_map:
            return self.state.speaker_map[original_speaker_id]

        # Unknown이거나 새로운 화자인 경우 새 speaker_XX 할당
        new_speaker_id = f"speaker_{self.state.speaker_counter:02d}"
        self.state.speaker_map[original_speaker_id] = new_speaker_id
        self.state.speaker_counter += 1

        # 새 화자 감지 신호 발송
        print(f"[DEBUG AudioWorker] Emitting sig_new_speaker_detected: {new_speaker_id}")
        self.sig_new_speaker_detected.emit(new_speaker_id)
        self.sig_status.emit(f"새 화자 할당: {original_speaker_id} -> {new_speaker_id}")

        return new_speaker_id

    def _select_best_overlapping_speaker(self, overlaps: list, start: float, end: float) -> Optional[str]:
        """겹치는 화자들 중 가장 적합한 화자 선택"""
        if not overlaps:
            return None

        # 각 화자별 점수 계산
        speaker_scores = {}

        for speaker_id, ds, de, confidence in overlaps:
            score = 0.0

            # 1. diarization 신뢰도 점수 (가장 중요)
            score += confidence * 0.5

            # 2. 시간적 겹침 정도
            overlap_start = max(start, ds)
            overlap_end = min(end, de)
            overlap_duration = max(0, overlap_end - overlap_start)
            segment_duration = end - start
            overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0
            score += overlap_ratio * 0.3

            # 3. 최근 화자 연속성 (같은 화자가 계속 말하는 경우 가산점)
            if speaker_id == self._last_speaker_id:
                score += 0.2

            speaker_scores[speaker_id] = score

        # 가장 높은 점수의 화자 선택
        if speaker_scores:
            best_speaker = max(speaker_scores, key=speaker_scores.get)
            self.sig_status.emit(f"겹침 감지: {len(overlaps)}명 중 {best_speaker} 선택 (점수: {speaker_scores[best_speaker]:.2f})")
            return best_speaker

        return None

    def _get_overlapping_speakers(self, start: float, end: float) -> list:
        """해당 시간 구간에 겹치는 모든 화자 반환"""
        overlaps = [
            (spk, ds, de, conf) for (ds, de, spk, conf) in self.state.diar_segments
            if not (end < ds or start > de)
        ]
        return overlaps

    def _extract_speaker_audio(self, wav_bytes: bytes, segment_start: float, segment_end: float,
                               diar_start: float, diar_end: float) -> Optional[bytes]:
        """특정 화자의 오디오 구간만 추출"""
        try:
            # WAV 바이트를 numpy 배열로 변환
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f)

            # 전체 청크에서 해당 화자 구간의 상대적 위치 계산
            chunk_duration = len(audio) / sr

            # diarization 구간을 chunk 내 상대 위치로 변환
            relative_start = max(0, diar_start - segment_start)
            relative_end = min(chunk_duration, diar_end - segment_start)

            if relative_end <= relative_start:
                return None

            # 샘플 인덱스로 변환
            start_sample = int(relative_start * sr)
            end_sample = int(relative_end * sr)

            # 해당 구간 추출
            speaker_audio = audio[start_sample:end_sample]

            # 너무 짧은 구간은 제외
            if len(speaker_audio) < sr * 0.3:  # 0.3초 미만
                return None

            # WAV 바이트로 변환
            mem = io.BytesIO()
            sf.write(mem, speaker_audio, sr, format="WAV")
            return mem.getvalue()

        except Exception as e:
            self.sig_status.emit(f"화자 오디오 추출 실패: {e}")
            return None

    def _transcribe_audio_segment(self, wav_bytes: bytes) -> str:
        """오디오 세그먼트를 STT 처리"""
        temp_path = None
        try:
            if not self.model or not wav_bytes:
                return ""

            # 임시 파일로 저장 (macOS 호환)
            fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="overlap_")
            try:
                os.write(fd, wav_bytes)
            finally:
                os.close(fd)

            # Whisper로 전사
            segments, _ = self.model.transcribe(temp_path, language="ko", vad_filter=True)
            text_parts = [s.text.strip() for s in segments if hasattr(s, 'text')]
            return " ".join(text_parts).strip()

        except Exception as e:
            self.sig_status.emit(f"겹침 구간 STT 실패: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _handle_overlapping_speech(self, segment: Segment, overlapping_speakers: list, wav_bytes: bytes):
        """겹치는 음성 구간 처리 - 화자별로 오디오 분리 후 개별 STT"""

        processed_speakers = []

        # 각 화자별로 처리
        for speaker_id, diar_start, diar_end, confidence in overlapping_speakers:
            # 겹치는 시간 계산
            overlap_start = max(segment.start, diar_start)
            overlap_end = min(segment.end, diar_end)

            if overlap_end <= overlap_start:
                continue

            # 해당 화자의 오디오만 추출
            speaker_wav = self._extract_speaker_audio(
                wav_bytes, segment.start, segment.end, diar_start, diar_end
            )

            if speaker_wav:
                # 추출된 오디오를 STT 처리
                speaker_text = self._transcribe_audio_segment(speaker_wav)

                if speaker_text:
                    # 텍스트가 있으면 세그먼트 생성
                    speaker_seg = Segment(
                        start=overlap_start,
                        end=overlap_end,
                        text=speaker_text,
                        speaker_id=speaker_id,
                        speaker_name=self.get_or_assign_speaker_id(speaker_id)
                    )
                    self.sig_transcript.emit(speaker_seg)
                    self._populate_persona_utterance(speaker_seg)  # 디지털 페르소나에 발언 추가
                    processed_speakers.append(self.get_or_assign_speaker_id(speaker_id))
                else:
                    # STT 실패 시 기본 표시
                    overlap_seg = Segment(
                        start=overlap_start,
                        end=overlap_end,
                        text="[음성 감지]",
                        speaker_id=speaker_id,
                        speaker_name=self.get_or_assign_speaker_id(speaker_id)
                    )
                    self.sig_transcript.emit(overlap_seg)
                    processed_speakers.append(self.get_or_assign_speaker_id(speaker_id))

        # 겹침 상태 로깅
        if processed_speakers:
            self.sig_status.emit(f"대화 겹침 분리 처리: {', '.join(processed_speakers)}")

    def _populate_persona_utterance(self, segment: Segment):
        """
        디지털 페르소나에 발언 추가 (Phase 1)

        Args:
            segment: 발언 세그먼트
        """
        if not self.persona_manager or not segment.text or segment.text == "[음성 감지]":
            return

        try:
            speaker_id = segment.speaker_name
            if not speaker_id or speaker_id == "Unknown":
                return

            # 1. 페르소나가 없으면 생성
            persona = self.persona_manager.get_persona(speaker_id)
            if not persona:
                speaker = self.speaker_manager.get_speaker_by_id(speaker_id)
                if speaker and speaker.embedding is not None:
                    display_name = self.speaker_manager.get_speaker_display_name(speaker_id)
                    self.persona_manager.create_persona(
                        speaker_id=speaker_id,
                        display_name=display_name,
                        voice_embedding=speaker.embedding,
                        llm_backend="openai:gpt-4o-mini"
                    )
                    print(f"[INFO] Created digital persona for {speaker_id} (live)")

            # 2. 발언을 RAG에 추가
            self.persona_manager.add_utterance(
                speaker_id=speaker_id,
                text=segment.text,
                start=segment.start,
                end=segment.end
            )

        except Exception as e:
            print(f"[WARN] Failed to populate persona utterance: {e}")

    def _get_embedding_inference(self):
        """임베딩 추출 모델 lazy loading"""
        if self._embedding_inference is None:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                self.sig_status.emit("[WARNING] HF_TOKEN이 설정되지 않아 화자 임베딩 기능을 사용할 수 없습니다. Settings에서 HuggingFace 토큰을 입력하세요.")
                return None

            try:
                from pyannote.audio import Inference
                self.sig_status.emit("화자 임베딩 모델 로딩 중...")
                self._embedding_inference = Inference(
                    "pyannote/embedding",
                    use_auth_token=hf_token
                )
                self.sig_status.emit("✓ 화자 임베딩 모델 로드 완료")
            except Exception as e:
                self.sig_status.emit(f"✗ 임베딩 모델 로드 실패: {e}")
                return None
        return self._embedding_inference

    def _extract_speaker_embedding(self, wav_bytes: bytes) -> Optional[np.ndarray]:
        """오디오에서 화자 임베딩 추출 (pyannote 사용)"""
        temp_path = None
        try:
            inference = self._get_embedding_inference()
            if inference is None:
                return None

            # WAV 바이트를 임시 파일로 저장 (macOS 호환)
            fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='embed_')
            try:
                os.write(fd, wav_bytes)
            finally:
                os.close(fd)

            # 임베딩 추출
            embedding = inference(temp_path)

            # 평균 임베딩 계산 (다차원인 경우)
            if hasattr(embedding, 'data'):
                # SlidingWindowFeature 객체인 경우
                embedding_data = embedding.data
                if len(embedding_data.shape) > 1:
                    embedding = np.mean(embedding_data, axis=0)
                else:
                    embedding = embedding_data
            elif isinstance(embedding, np.ndarray):
                if len(embedding.shape) > 1:
                    embedding = np.mean(embedding, axis=0)

            return embedding

        except Exception as e:
            self.sig_status.emit(f"임베딩 추출 실패: {e}")
            return None
        finally:
            # 임시 파일 삭제
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _smart_infer_speaker_id(self, start: float, end: float, wav_bytes: bytes, text: str) -> str:
        """임베딩 기반 화자 ID 추론 (diarization 결과 + 화자 연속성 강화)"""

        # 1. 텍스트 검증 (먼저 체크)
        if not text or len(text.strip()) < self._min_text_length:
            # 텍스트가 너무 짧으면 무조건 이전 화자 유지
            if self._last_speaker_id:
                return self._last_speaker_id
            return "Unknown"

        # 2. 화자 연속성 확인 (가장 중요! - 우선순위 상향)
        silence_duration = start - self._last_speech_time

        # 침묵이 짧으면 같은 화자일 가능성이 매우 높음 - 무조건 유지
        if self._last_speaker_id and silence_duration <= self._continuity_threshold:
            self._last_speech_time = end
            # 연속성 유지 로그
            # self.sig_status.emit(f"[화자 연속] {self._last_speaker_id} 유지 (침묵: {silence_duration:.1f}초)")
            return self._last_speaker_id

        # 3. diarization 결과 확인 (화자 변경 가능성이 있을 때만)
        overlaps = [
            (spk, ds, de, conf) for (ds, de, spk, conf) in self.state.diar_segments
            if not (end < ds or start > de)
        ]

        # 4. diarization 결과가 있으면 사용
        if overlaps:
            if len(overlaps) == 1:
                # 단일 화자
                speaker_id = overlaps[0][0]
                # 이전 화자와 같으면 그대로, 다르면 변경
                if self._last_speaker_id and speaker_id == self._last_speaker_id:
                    self._last_speech_time = end
                    return self._last_speaker_id
                else:
                    self._last_speech_time = end
                    self._last_speaker_id = speaker_id
                    return speaker_id
            else:
                # 여러 화자가 겹치는 경우
                best_speaker = self._select_best_overlapping_speaker(overlaps, start, end)
                if best_speaker:
                    self._last_speech_time = end
                    self._last_speaker_id = best_speaker
                    return best_speaker

        # 5. 임베딩 기반 화자 식별 (침묵이 길거나 첫 발화인 경우만)
        # diarization이 비활성화되어도 임베딩 기반 식별 시도
        if wav_bytes:
            embedding = self._extract_speaker_embedding(wav_bytes)
            if embedding is not None:
                # SpeakerManager로 화자 식별 (낮은 임계값 사용)
                speaker_id, confidence = self.speaker_manager.identify_speaker(
                    embedding,
                    threshold=0.55  # 더 낮춘 임계값 (0.70 -> 0.55)
                )

                self._last_speaker_id = speaker_id
                self._last_speech_time = end
                diar_status = "활성화" if self.state.diarization_enabled else "비활성화"
                self.sig_status.emit(f"[화자 식별] {speaker_id} (신뢰도: {confidence:.2f}, Diar: {diar_status})")
                return speaker_id

        # 6. 임베딩 추출 실패 시 연속성만으로 판단
        if self._last_speaker_id:
            self._last_speech_time = end
            return self._last_speaker_id

        # 7. 최후 수단: Unknown
        return "Unknown"

    # ====== ASR ======
    def init_asr(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper 미설치")

        # 플랫폼별 GPU 설정
        if self.state.use_gpu:
            if IS_MAC:
                # macOS: MPS 또는 CPU
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        device = "cpu"  # faster-whisper는 MPS 직접 지원 안함, CPU 사용
                        compute = "int8"
                        self.sig_status.emit("macOS MPS 감지됨, CPU로 실행 (faster-whisper MPS 미지원)")
                    else:
                        device = "cpu"
                        compute = "int8"
                except ImportError:
                    device = "cpu"
                    compute = "int8"
            else:
                # Windows/Linux: CUDA 또는 CPU
                device = "cuda"
                compute = "float16"
        else:
            device = "cpu"
            compute = "int8"

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
        if AUDIO_BACKEND is None:
            raise RuntimeError("오디오 라이브러리 미설치 (PyAudio 또는 sounddevice 필요)")

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

        if AUDIO_BACKEND == "pyaudio":
            self._start_pyaudio()
        else:
            self._start_sounddevice()

        threading.Thread(target=self._chunk_loop, daemon=True).start()
        self.sig_status.emit(f"Audio capture started ({AUDIO_BACKEND}).")

    def _start_pyaudio(self):
        """PyAudio 기반 오디오 캡처 시작 (Windows 주로 사용)"""
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                output=False,
                input_device_index=self._input_device_index,
                frames_per_buffer=int(SAMPLE_RATE * 0.2),
                stream_callback=self._on_audio_pyaudio,
            )
        except Exception as e:
            if self._input_device_index is not None:
                self.sig_status.emit(f"Input device {self._input_device_index} 실패 -> 기본장치 재시도")
                self.stream = self.audio.open(
                    format=self.audio.get_format_from_width(SAMPLE_WIDTH),
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=False,
                    frames_per_buffer=int(SAMPLE_RATE * 0.2),
                    stream_callback=self._on_audio_pyaudio,
                )
            else:
                raise
        self.stream.start_stream()

    def _start_sounddevice(self):
        """sounddevice 기반 오디오 캡처 시작 (macOS/Linux)"""
        def callback(indata, frames, time_info, status):
            if status:
                self.sig_status.emit(f"sounddevice status: {status}")
            # indata는 이미 float32 numpy array
            self._on_audio_sounddevice(indata.copy(), frames)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=int(SAMPLE_RATE * 0.2),
            device=self._input_device_index,
            callback=callback
        )
        self.stream.start()

    def stop(self):
        self._stop.set()
        try:
            if AUDIO_BACKEND == "pyaudio":
                if self.stream:
                    # self.stream.stop_stream()
                    self.stream.close()
                if self.audio:
                    self.audio.terminate()
            else:
                if self.stream:
                    # self.stream.stop()
                    self.stream.close()
        except Exception:
            pass
        self.sig_status.emit("Audio capture stopped.")

    # ====== Audio callback / buffering ======
    def _on_audio_pyaudio(self, in_data, frame_count, time_info, status):
        """PyAudio 콜백"""
        # 오디오 데이터를 numpy 배열로 변환
        data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        data_np = data_np.reshape(-1, CHANNELS) if CHANNELS == 1 else data_np

        self._process_audio_data(data_np, frame_count)
        return (None, pyaudio.paContinue)

    def _on_audio_sounddevice(self, indata, frame_count):
        """sounddevice 콜백"""
        # indata는 이미 float32 numpy array
        data_np = indata.reshape(-1, CHANNELS) if CHANNELS == 1 else indata
        self._process_audio_data(data_np, frame_count)

    def _process_audio_data(self, data_np: np.ndarray, frame_count: int):
        """공통 오디오 데이터 처리"""
        # 버퍼 누적 (PyAudio용 - sounddevice는 바이트가 아닌 float)
        if AUDIO_BACKEND == "pyaudio":
            # int16 -> bytes 변환
            in_data = (data_np * 32768.0).astype(np.int16).tobytes()
            with self._buf_lock:
                self._buf.write(in_data)
                self._frames_elapsed += frame_count
                self.state.audio_time_elapsed += frame_count / SAMPLE_RATE
        else:
            # sounddevice는 float32로 직접 처리
            with self._buf_lock:
                # float32를 int16 bytes로 변환해서 저장 (기존 로직 호환)
                in_data = (data_np * 32768.0).astype(np.int16).tobytes()
                self._buf.write(in_data)
                self._frames_elapsed += frame_count
                self.state.audio_time_elapsed += frame_count / SAMPLE_RATE

        # 녹음 중이면 프레임 저장
        with self._recording_lock:
            if self._recording:
                self._recording_frames.append(data_np.copy())

        # WAV 파일에 추가
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
            spk for (ds, de, spk, conf) in self.state.diar_segments if not (e < ds or s > de)
        ]
        if not overlaps:
            # 화자 분리가 없으면 기본 화자로 처리
            if self.state.speaker_map:
                # 등록된 화자가 있으면 첫 번째 화자 사용
                return list(self.state.speaker_map.keys())[0]
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

            temp_path = None
            try:
                # macOS/Windows 호환: 안전한 tempfile 처리
                fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="chunk_")
                try:
                    os.write(fd, wav_bytes)
                finally:
                    os.close(fd)

                segments, info = self.model.transcribe(
                    temp_path, language="ko", vad_filter=True
                )
                for s in segments:
                    seg = Segment(
                        start=chunk_offset_seconds + s.start,
                        end=chunk_offset_seconds + s.end,
                        text=(s.text or "").strip(),
                    )

                    # diarization 결과에서 겹치는 화자가 있는지 확인
                    overlapping_speakers = self._get_overlapping_speakers(seg.start, seg.end)

                    if len(overlapping_speakers) > 1:
                        # 여러 화자가 동시에 말하는 경우: 분할 처리
                        self._handle_overlapping_speech(seg, overlapping_speakers, wav_bytes)
                    else:
                        # 단일 화자 또는 겹침 없음: 기존 로직
                        original_spk_id = self._smart_infer_speaker_id(
                            seg.start, seg.end, wav_bytes, seg.text
                        )
                        seg.speaker_id = original_spk_id
                        seg.speaker_name = self.get_or_assign_speaker_id(original_spk_id)
                        self.sig_transcript.emit(seg)
                        self._populate_persona_utterance(seg)  # 디지털 페르소나에 발언 추가

                chunk_offset_seconds += CHUNK_SECONDS

            except Exception as e:
                self.sig_status.emit(f"STT error: {e}")
                time.sleep(0.2)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass