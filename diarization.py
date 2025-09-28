# diarization.py
import os
import time
import wave
import queue
import threading
import tempfile
from datetime import timedelta, datetime
from collections import deque
from typing import Callable, Optional, Dict

import numpy as np
import torch
import yaml
import pyaudio

from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel

from config import Config

class RealTimeDiarization:
    def __init__(self, config: Config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.BUFFER_DURATION * config.SAMPLE_RATE))
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None

        # 콜백 함수들
        self.on_transcription: Optional[Callable[[str, str, str], None]] = None
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # 모델 관련
        self.diar_pipeline = None
        self.whisper_model = None
        self.models_loaded = False

        # 화자별 텍스트 저장
        self.speaker_texts: Dict[str, list[str]] = {}

        # 처리 스레드
        self._processing_thread = None

    def _emit_status(self, message: str):
        """상태 메시지 발생"""
        if self.on_status_change:
            self.on_status_change(message)
        else:
            print("[STATUS]", message)

    def _emit_error(self, message: str):
        """오류 메시지 발생"""
        if self.on_error:
            self.on_error(message)
        else:
            print("[ERROR]", message)

    def _emit_transcription(self, timestamp: str, speaker: str, text: str):
        """전사 결과 발생"""
        if self.on_transcription:
            self.on_transcription(timestamp, speaker, text)
        else:
            print(f"[{timestamp}] {speaker}: {text}")

        # 화자별 텍스트 저장
        if speaker not in self.speaker_texts:
            self.speaker_texts[speaker] = []
        self.speaker_texts[speaker].append(text)

    def hhmmss(self, seconds: float) -> str:
        """초를 HH:MM:SS 형식으로 변환"""
        return str(timedelta(seconds=round(seconds)))

    def load_models(self) -> bool:
        """모델 로딩"""
        try:
            self._emit_status("모델 로딩 시작...")
            
            # Pyannote 모델 로딩
            seg_dir = str(self.config.PYANNOTE_MODEL_PATH / "segmentation-3.0" / "pytorch_model.bin")
            emb_dir = str(self.config.PYANNOTE_MODEL_PATH / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin")
            segmentation_model = Model.from_pretrained(seg_dir)
            embedding_model = Model.from_pretrained(emb_dir)

            # 파이프라인 설정 로딩
            cfg_path = self.config.PYANNOTE_MODEL_PATH / "config.yaml"
            pipeline_config = {}
            if cfg_path.exists():
                with open(cfg_path, "r") as f:
                    pipeline_config = yaml.safe_load(f) or {}
            
            self.diar_pipeline = SpeakerDiarization(
                segmentation=segmentation_model, 
                embedding=embedding_model
            )
            if 'params' in pipeline_config:
                self.diar_pipeline.instantiate(pipeline_config['params'])

            # Whisper 모델 로딩
            self._emit_status("Whisper 모델 로드 중...")
            self.whisper_model = WhisperModel(
                str(self.config.WHISPER_MODEL_PATH),
                device=self.config.WHISPER_DEVICE,
                compute_type=self.config.WHISPER_COMPUTE_TYPE
            )

            self.models_loaded = True
            self._emit_status("모델 로딩 완료")
            return True
        except Exception as e:
            self._emit_error(f"모델 로딩 실패: {e}")
            return False

    def get_available_audio_devices(self) -> Dict[int, str]:
        """사용 가능한 오디오 디바이스 조회"""
        devices = {}
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices[i] = info['name']
            p.terminate()
        except Exception as e:
            self._emit_error(f"오디오 디바이스 조회 실패: {e}")
        return devices

    def audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 콜백 함수"""
        if status:
            self._emit_error(f"Audio callback status: {status}")
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_data)
            self.audio_queue.put(audio_data.copy())
        except Exception as e:
            self._emit_error(f"오디오 콜백 오류: {e}")
        return (in_data, pyaudio.paContinue)

    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """녹음 시작"""
        if not self.models_loaded:
            self._emit_error("모델이 아직 로드되지 않았습니다.")
            return False
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.config.AUDIO_FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.CHUNK_SIZE,
                stream_callback=self.audio_callback
            )
            self.is_recording = True
            self.audio_stream.start_stream()
            self._processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self._processing_thread.start()
            self._emit_status("녹음 시작")
            return True
        except Exception as e:
            self._emit_error(f"녹음 시작 실패: {e}")
            self.cleanup()
            return False

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        try:
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            self._emit_status("녹음 중지됨")
        except Exception as e:
            self._emit_error(f"녹음 중지 오류: {e}")

    def process_audio_segment(self, audio_data: np.ndarray, base_time: float):
        """오디오 세그먼트 처리"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                tmp_path = tmpf.name
            
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(self.config.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.config.SAMPLE_RATE)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            # 화자분리 수행
            diar_result = self.diar_pipeline(tmp_path)

            now = datetime.now()
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
                if (turn.end - turn.start) < self.config.MIN_SEG_DUR:
                    continue
                
                # 오디오 세그먼트 추출
                i0 = max(0, int(turn.start * self.config.SAMPLE_RATE))
                i1 = min(len(audio_data), int(turn.end * self.config.SAMPLE_RATE))
                seg = audio_data[i0:i1]
                if seg.size == 0:
                    continue

                # 음성인식 수행
                try:
                    segments, _ = self.whisper_model.transcribe(
                        seg,
                        language=self.config.WHISPER_LANG,
                        vad_filter=True,
                        beam_size=1,
                        word_timestamps=False
                    )
                    text = " ".join([s.text.strip() for s in segments if s.text])
                except Exception as e:
                    self._emit_error(f"Whisper 전사 오류: {e}")
                    text = ""

                # 전사 결과 발생
                if text.strip():
                    timestamp = now.strftime("%H:%M:%S")
                    self._emit_transcription(timestamp, speaker, text)

            os.unlink(tmp_path)
        except Exception as e:
            self._emit_error(f"처리 중 예외: {e}")

    def processing_loop(self):
        """오디오 처리 루프"""
        process_buffer = []
        last_process_time = time.time()
        
        while self.is_recording:
            try:
                # 큐에서 오디오 데이터 가져오기
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    process_buffer.extend(chunk)
                except queue.Empty:
                    pass

                current_time = time.time()
                buffer_duration = len(process_buffer) / self.config.SAMPLE_RATE

                # 처리 조건 확인
                should_process = (
                    buffer_duration >= self.config.PROCESS_INTERVAL or 
                    current_time - last_process_time > self.config.PROCESS_INTERVAL
                )

                if should_process and process_buffer:
                    audio_array = np.array(process_buffer, dtype=np.float32)
                    base_time = current_time - buffer_duration
                    self.process_audio_segment(audio_array, base_time)
                    process_buffer = []
                    last_process_time = current_time
                    
            except Exception as e:
                self._emit_error(f"Processing loop 예외: {e}")
        
        self._emit_status("Processing loop 종료")

    def get_speaker_texts(self) -> Dict[str, list[str]]:
        """화자별 텍스트 반환"""
        return self.speaker_texts.copy()

    def clear_speaker_texts(self):
        """화자별 텍스트 초기화"""
        self.speaker_texts.clear()

    def cleanup(self):
        """리소스 정리"""
        self.stop_recording()