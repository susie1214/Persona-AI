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

        # 간단한 화자 구분을 위한 오디오 특성 저장
        self._speaker_features = {}  # {speaker_id: [audio_features]}
        self._last_speaker_id = None
        self._last_speech_time = 0.0
        self._silence_threshold = 4.0  # 4초 침묵 후 새 화자 가능성
        self._min_text_length = 5  # 최소 텍스트 길이
        self._audio_similarity_threshold = 0.7  # 오디오 특성 차이 임계값 (높을수록 보수적)

        # (선택) 임시폴더를 프로젝트 내부로 고정하고 싶다면 주석 해제
        # os.makedirs("output/tmp", exist_ok=True)
        # tempfile.tempdir = os.path.abspath("output/tmp")

    # ====== Public API ======
    def set_input_device_index(self, idx: Optional[int]):
        """UI에서 마이크 인덱스를 전달 (None=기본장치)"""
        self._input_device_index = idx

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
        self.sig_new_speaker_detected.emit(new_speaker_id)
        self.sig_status.emit(f"새 화자 할당: {original_speaker_id} -> {new_speaker_id}")

        return new_speaker_id

    def _select_best_overlapping_speaker(self, overlaps: list, start: float, end: float,
                                       wav_bytes: bytes, text: str) -> Optional[str]:
        """겹치는 화자들 중 가장 적합한 화자 선택"""
        if not overlaps:
            return None

        # 각 화자별 점수 계산
        speaker_scores = {}

        for speaker_id, ds, de, confidence in overlaps:
            score = 0.0

            # 1. diarization 신뢰도 점수
            score += confidence * 0.4

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

            # 4. 오디오 특성 유사성 (가능한 경우)
            if wav_bytes and speaker_id in self._speaker_features:
                try:
                    current_features = self._extract_audio_features(wav_bytes)
                    speaker_features = self._speaker_features.get(speaker_id, [])
                    if speaker_features:
                        avg_features = np.mean(speaker_features, axis=0)
                        similarity = 1.0 / (1.0 + np.linalg.norm(current_features - avg_features))
                        score += similarity * 0.1
                except Exception:
                    pass

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

    def _handle_overlapping_speech(self, segment: Segment, overlapping_speakers: list, wav_bytes: bytes):
        """겹치는 음성 구간 처리 - 각 화자별로 세그먼트 생성"""
        # 주요 화자 선택
        main_speaker = self._select_best_overlapping_speaker(
            overlapping_speakers, segment.start, segment.end, wav_bytes, segment.text
        )

        if main_speaker:
            # 주요 화자의 세그먼트 생성
            main_seg = Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=main_speaker,
                speaker_name=self.get_or_assign_speaker_id(main_speaker)
            )
            self.sig_transcript.emit(main_seg)

            # 다른 화자들도 표시 (텍스트는 [겹침]으로 표시)
            for speaker_id, ds, de, conf in overlapping_speakers:
                if speaker_id != main_speaker:
                    # 겹치는 시간 계산
                    overlap_start = max(segment.start, ds)
                    overlap_end = min(segment.end, de)

                    if overlap_end > overlap_start:
                        overlap_seg = Segment(
                            start=overlap_start,
                            end=overlap_end,
                            text=f"[{main_seg.speaker_name}와 겹침]",
                            speaker_id=speaker_id,
                            speaker_name=self.get_or_assign_speaker_id(speaker_id)
                        )
                        self.sig_transcript.emit(overlap_seg)

            # 겹침 상태 로깅
            speaker_names = [self.get_or_assign_speaker_id(spk) for spk, _, _, _ in overlapping_speakers]
            self.sig_status.emit(f"대화 겹침 감지: {', '.join(speaker_names)}")

    def _extract_audio_features(self, wav_bytes: bytes) -> np.ndarray:
        """간단한 오디오 특성 추출 (에너지, 주파수 특성 등)"""
        try:
            import librosa
            # WAV 바이트를 numpy 배열로 변환
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f)

            # 기본 특성 추출
            energy = np.mean(audio ** 2)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))

            return np.array([energy, zcr, spectral_centroid])
        except Exception:
            # librosa가 없거나 오류 시 기본값 반환
            return np.array([0.1, 0.1, 1000.0])

    def _smart_infer_speaker_id(self, start: float, end: float, wav_bytes: bytes, text: str) -> str:
        """향상된 화자 ID 추론 (시간 기반 + 오디오 특성 + diarization)"""
        # 1. diarization 결과 확인 (겹치는 구간 처리)
        overlaps = [
            (spk, ds, de, conf) for (ds, de, spk, conf) in self.state.diar_segments
            if not (end < ds or start > de)
        ]

        if overlaps:
            # 겹치는 구간이 있는 경우 처리
            if len(overlaps) == 1:
                # 단일 화자: 그대로 사용
                speaker_id = overlaps[0][0]
                self._last_speech_time = end
                self._last_speaker_id = speaker_id
                return speaker_id
            else:
                # 여러 화자가 겹치는 경우: 가장 적합한 화자 선택
                best_speaker = self._select_best_overlapping_speaker(overlaps, start, end, wav_bytes, text)
                if best_speaker:
                    self._last_speech_time = end
                    self._last_speaker_id = best_speaker
                    return best_speaker

        # 2. 텍스트가 충분히 길어야 새 화자로 판단 (단, 첫 화자는 예외)
        if not text.strip() or (len(text.strip()) < self._min_text_length and self._last_speaker_id is not None):
            return self._last_speaker_id or "Unknown"

        # 3. 화자 연속성 우선 확인
        silence_duration = start - self._last_speech_time

        # 동일 화자 연속 발언 감지 강화
        if self._last_speaker_id is not None and silence_duration <= self._silence_threshold:
            # 침묵이 짧으면 일단 같은 화자로 가정
            if wav_bytes and self._last_speaker_id in self._speaker_features:
                try:
                    current_features = self._extract_audio_features(wav_bytes)
                    last_features = self._speaker_features[self._last_speaker_id]
                    if last_features:
                        avg_last = np.mean(last_features, axis=0)
                        feature_diff = np.linalg.norm(current_features - avg_last)

                        # 오디오 특성이 크게 다르지 않으면 같은 화자로 처리
                        if feature_diff < self._audio_similarity_threshold:
                            # 기존 화자의 특성에 추가
                            self._speaker_features[self._last_speaker_id].append(current_features)
                            self._last_speech_time = end
                            return self._last_speaker_id
                except Exception:
                    # 오디오 분석 실패 시 시간만으로 판단
                    self._last_speech_time = end
                    return self._last_speaker_id
            else:
                # 오디오 특성이 없으면 시간만으로 판단 (같은 화자로 처리)
                self._last_speech_time = end
                return self._last_speaker_id

        # 4. 새로운 화자 생성 조건
        should_create_new_speaker = False

        if self._last_speaker_id is None:
            # 첫 번째 화자
            should_create_new_speaker = True
        elif silence_duration > self._silence_threshold:
            # 충분한 침묵 후 발화 → 새 화자 가능성
            should_create_new_speaker = True

        if should_create_new_speaker:
            # 새로운 화자 ID 생성
            new_speaker_id = f"time_speaker_{len(self._speaker_features)}"

            # 오디오 특성 저장
            if wav_bytes:
                try:
                    audio_features = self._extract_audio_features(wav_bytes)
                    self._speaker_features[new_speaker_id] = [audio_features]
                except Exception:
                    pass

            self._last_speaker_id = new_speaker_id
            self._last_speech_time = end

            return new_speaker_id

        # 5. 기존 화자 계속 (새 화자 생성 조건에 해당하지 않는 경우)
        if self._last_speaker_id:
            # 기존 화자의 특성에 현재 오디오 특성 추가
            if wav_bytes:
                try:
                    current_features = self._extract_audio_features(wav_bytes)
                    if self._last_speaker_id in self._speaker_features:
                        self._speaker_features[self._last_speaker_id].append(current_features)
                    else:
                        self._speaker_features[self._last_speaker_id] = [current_features]
                except Exception:
                    pass

            self._last_speech_time = end
            return self._last_speaker_id

        # 6. 최후의 수단: Unknown
        return "Unknown"

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
