# realtime_diar_gui_with_summary.py
import os
import sys
import time
import wave
import queue
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta, datetime
from collections import deque
from typing import Callable, Optional, Dict, Any

import numpy as np
import torch
import yaml

# audio libs
import pyaudio

# speech libs
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel

# PyQt6
from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QTextEdit,
    QPushButton, QLabel, QComboBox, QSplitter, QMessageBox, QFrame
)

# ---------------- Configuration ----------------
@dataclass
class Config:
    PYANNOTE_MODEL_PATH: Path = Path("./models/diart_model")
    WHISPER_MODEL_PATH: Path = Path("./models/whisper-small-ct2")
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    CHANNELS: int = 1
    AUDIO_FORMAT: int = pyaudio.paInt16
    BUFFER_DURATION: float = 30.0
    PROCESS_INTERVAL: float = 10.0
    MIN_SEG_DUR: float = 0.35
    OVERLAP_DURATION: float = 5.0

# ---------------- RealTimeDiarization ----------------
class RealTimeDiarization:
    def __init__(self, config: Config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.BUFFER_DURATION * config.SAMPLE_RATE))
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None

        self.on_transcription: Optional[Callable[[str, str, str], None]] = None
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        self.diar_pipeline = None
        self.whisper_model = None
        self.models_loaded = False

        self.speaker_texts: Dict[str, list[str]] = {}  # 화자별 전사 저장

        self._processing_thread = None

    # ---------- callback helpers ----------
    def _emit_status(self, message: str):
        if self.on_status_change:
            self.on_status_change(message)
        else:
            print("[STATUS]", message)

    def _emit_error(self, message: str):
        if self.on_error:
            self.on_error(message)
        else:
            print("[ERROR]", message)

    def _emit_transcription(self, timestamp: str, speaker: str, text: str):
        if self.on_transcription:
            self.on_transcription(timestamp, speaker, text)
        else:
            print(f"[{timestamp}] {speaker}: {text}")

        # 화자별 전사 저장
        if speaker not in self.speaker_texts:
            self.speaker_texts[speaker] = []
        self.speaker_texts[speaker].append(text)

    def hhmmss(self, seconds: float) -> str:
        return str(timedelta(seconds=round(seconds)))

    # ---------- model loading ----------
    def load_models(self) -> bool:
        try:
            self._emit_status("모델 로딩 시작...")
            seg_dir = str(self.config.PYANNOTE_MODEL_PATH / "segmentation-3.0" / "pytorch_model.bin")
            emb_dir = str(self.config.PYANNOTE_MODEL_PATH / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin")
            segmentation_model = Model.from_pretrained(seg_dir)
            embedding_model = Model.from_pretrained(emb_dir)

            cfg_path = self.config.PYANNOTE_MODEL_PATH / "config.yaml"
            pipeline_config = {}
            if cfg_path.exists():
                with open(cfg_path, "r") as f:
                    pipeline_config = yaml.safe_load(f) or {}
            self.diar_pipeline = SpeakerDiarization(segmentation=segmentation_model, embedding=embedding_model)
            if 'params' in pipeline_config:
                self.diar_pipeline.instantiate(pipeline_config['params'])

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

    # ---------- audio device helpers ----------
    def get_available_audio_devices(self) -> Dict[int, str]:
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

    # ---------- pyaudio callback ----------
    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            self._emit_error(f"Audio callback status: {status}")
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_data)
            self.audio_queue.put(audio_data.copy())
        except Exception as e:
            self._emit_error(f"오디오 콜백 오류: {e}")
        return (in_data, pyaudio.paContinue)

    # ---------- recording control ----------
    def start_recording(self, device_index: Optional[int] = None) -> bool:
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

    # ---------- audio processing ----------
    def process_audio_segment(self, audio_data: np.ndarray, base_time: float):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                tmp_path = tmpf.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(self.config.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.config.SAMPLE_RATE)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            diar_result = self.diar_pipeline(tmp_path)

            now = datetime.now()
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
                if (turn.end - turn.start) < self.config.MIN_SEG_DUR:
                    continue
                i0 = max(0, int(turn.start * self.config.SAMPLE_RATE))
                i1 = min(len(audio_data), int(turn.end * self.config.SAMPLE_RATE))
                seg = audio_data[i0:i1]
                if seg.size == 0:
                    continue

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

                if text.strip():
                    timestamp = now.strftime("%H:%M:%S")
                    self._emit_transcription(timestamp, speaker, text)

            os.unlink(tmp_path)
        except Exception as e:
            self._emit_error(f"처리 중 예외: {e}")

    def processing_loop(self):
        process_buffer = []
        last_process_time = time.time()
        while self.is_recording:
            try:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    process_buffer.extend(chunk)
                except queue.Empty:
                    pass

                current_time = time.time()
                buffer_duration = len(process_buffer) / self.config.SAMPLE_RATE

                if (buffer_duration >= self.config.PROCESS_INTERVAL) or (current_time - last_process_time > self.config.PROCESS_INTERVAL):
                    audio_array = np.array(process_buffer, dtype=np.float32)
                    base_time = current_time - buffer_duration
                    self.process_audio_segment(audio_array, base_time)
                    process_buffer = []
                    last_process_time = current_time
            except Exception as e:
                self._emit_error(f"Processing loop 예외: {e}")
        self._emit_status("Processing loop 종료")

    # ---------- cleanup ----------
    def cleanup(self):
        self.stop_recording()

# ---------------- PyQt GUI ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 화자분리 + 요약 GUI")
        self.resize(900, 700)

        self.config = Config()
        self.rt = RealTimeDiarization(self.config)

        # 채팅 뷰
        self.chat_view = QTextBrowser()
        self.chat_view.setFrameShape(QFrame.Shape.StyledPanel)
        self.chat_view.setOpenExternalLinks(False)

        # 요약 뷰
        self.summary_view = QTextBrowser()
        self.summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        self.summary_view.setOpenExternalLinks(False)

        # 입력창
        self.input_edit = QTextEdit()
        self.input_edit.setFixedHeight(80)
        self.send_button = QPushButton("보내기")
        self.start_button = QPushButton("시작")
        self.stop_button = QPushButton("중지")
        self.summary_button = QPushButton("화자별 요약")
        self.device_combo = QComboBox()
        self.status_label = QLabel("상태: 준비됨")

        # 상단 버튼 레이아웃
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("입력 디바이스:"))
        top_layout.addWidget(self.device_combo)
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.stop_button)
        top_layout.addWidget(self.summary_button)
        top_layout.addStretch()
        top_layout.addWidget(self.status_label)

        # 입력창 레이아웃
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.input_edit)
        right_buttons = QVBoxLayout()
        right_buttons.addWidget(self.send_button)
        right_buttons.addStretch()
        bottom_layout.addLayout(right_buttons)

        # main splitter: 채팅창 + 요약창
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.chat_view)
        splitter.addWidget(self.summary_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # 메인 레이아웃
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addLayout(bottom_layout)

        # signals
        self.send_button.clicked.connect(self.on_send_clicked)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        self.summary_button.clicked.connect(self.on_summary_clicked)

        self.rt.on_transcription = self.on_transcription
        self.rt.on_status_change = self.on_status
        self.rt.on_error = self.on_error

        # populate devices
        self.populate_devices()
        
        
        from openai import OpenAI
        from dotenv import load_dotenv
        import os

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key = api_key)

    # ---------- device helpers ----------
    def populate_devices(self):
        devices = self.rt.get_available_audio_devices()
        self.device_combo.clear()
        for idx, name in devices.items():
            self.device_combo.addItem(f"{name} ({idx})", idx)

    # ---------- callbacks ----------
    def on_transcription(self, timestamp: str, speaker: str, text: str):
        self.chat_view.append(f"[{timestamp}] <b>{speaker}</b>: {text}")
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def on_status(self, message: str):
        self.status_label.setText(f"상태: {message}")

    def on_error(self, message: str):
        self.chat_view.append(f"<span style='color:red'>[ERROR] {message}</span>")

    # ---------- button actions ----------
    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.chat_view.append(f"[{timestamp}] <b>사용자</b>: {text}")
            self.input_edit.clear()

    def on_start(self):
        device_index = self.device_combo.currentData()
        if not self.rt.models_loaded:
            loaded = self.rt.load_models()
            if not loaded:
                QMessageBox.critical(self, "오류", "모델 로딩 실패")
                return
        self.rt.start_recording(device_index)

    def on_stop(self):
        self.rt.stop_recording()

    # ---------- summary ----------
    def on_summary_clicked(self):
        self.summary_view.clear()
        if not self.rt.speaker_texts:
            self.summary_view.append("<i>아직 전사된 데이터가 없습니다.</i>")
            return

        for speaker, texts in self.rt.speaker_texts.items():
            combined_text = " ".join(texts)
            
            print(f"[DEBUG] combined_text : {combined_text}")
            # OpenAI API로 요약
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "한국어 텍스트를 간결하게 요약해줘."},
                        {"role": "user", "content": combined_text}
                    ],
                    max_tokens=1024
                )
                summary = response.choices[0].message.content
            except Exception as e:
                summary = f"(요약 실패: {e})"

            self.summary_view.append(f"<b>{speaker} 요약:</b> {summary}\n")

# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
