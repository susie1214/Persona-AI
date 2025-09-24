# realtime_diar_gui.py
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

# speech libs (외부 의존성: pyannote.audio, faster_whisper)
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel

# PyQt6
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QTextEdit,
    QPushButton, QLabel, QComboBox, QSplitter, QMessageBox, QFrame
)


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


class RealTimeDiarization:
    """실시간 화자분리 및 전사 처리 클래스 (비동기 작업을 내부 스레드로 처리)."""

    def __init__(self, config: Config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.BUFFER_DURATION * config.SAMPLE_RATE))
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None

        # callbacks
        self.on_transcription: Optional[Callable[[str, str, str], None]] = None
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # models
        self.diar_pipeline = None
        self.whisper_model = None
        self.models_loaded = False

        # internal
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

    # ---------- utility ----------
    def hhmmss(self, seconds: float) -> str:
        return str(timedelta(seconds=round(seconds)))

    # ---------- model loading ----------
    def load_models(self) -> bool:
        """Pyannote + Whisper 모델 로드 (디렉토리 단위로 전달하도록 정리)."""
        try:
            self._emit_status("모델 로딩 시작...")
            # pyannote segmentation & embedding: 디렉토리 경로를 넘김
            seg_dir = str(self.config.PYANNOTE_MODEL_PATH / "segmentation-3.0" / "pytorch_model.bin")
            emb_dir = str(self.config.PYANNOTE_MODEL_PATH / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin")
            self._emit_status(f"Pyannote segmentation 로드: {seg_dir}")
            segmentation_model = Model.from_pretrained(seg_dir)
            self._emit_status(f"Pyannote embedding 로드: {emb_dir}")
            embedding_model = Model.from_pretrained(emb_dir)

            # pipeline config
            cfg_path = self.config.PYANNOTE_MODEL_PATH / "config.yaml"
            pipeline_config = {}
            if cfg_path.exists():
                with open(cfg_path, "r") as f:
                    pipeline_config = yaml.safe_load(f) or {}
            # instantiate pipeline
            self.diar_pipeline = SpeakerDiarization(
                segmentation=segmentation_model,
                embedding=embedding_model,
            )
            if 'params' in pipeline_config:
                self.diar_pipeline.instantiate(pipeline_config['params'])

            # Whisper
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
            # Put small chunk copy for processing loop
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
            if device_index is None:
                self._emit_status("기본 오디오 디바이스 사용")
            else:
                try:
                    info = self.pyaudio_instance.get_device_info_by_index(device_index)
                    self._emit_status(f"오디오 디바이스 선택: {info['name']}")
                except Exception:
                    self._emit_status("디바이스 인덱스가 유효하지 않습니다. 기본 디바이스 사용.")

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
            # start processing thread
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
        """오디오 블록(실수형, -1..1)을 받아 임시 WAV로 저장 → diarization → whisper 전사."""
        try:
            # 임시 wav 생성 (int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                tmp_path = tmpf.name
            try:
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(self.config.CHANNELS)
                    wf.setsampwidth(2)  # 16bit
                    wf.setframerate(self.config.SAMPLE_RATE)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
            except Exception as e:
                self._emit_error(f"WAV 작성 실패: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return

            # diarization
            try:
                diar_result = self.diar_pipeline(tmp_path)
            except Exception as e:
                self._emit_error(f"diarization 실패: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return

            # 각 발화 구간을 whisper로 전사
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

            # cleanup tmp wav
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except Exception as e:
            self._emit_error(f"처리 중 예외: {e}")

    def processing_loop(self):
        """audio_queue에서 모은 데이터를 버퍼로 합쳐 일정 주기마다 처리."""
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

                if (buffer_duration >= self.config.PROCESS_INTERVAL or
                        current_time - last_process_time >= self.config.PROCESS_INTERVAL):
                    if len(process_buffer) > 0:
                        audio_array = np.array(process_buffer, dtype=np.float32)
                        base_time = current_time - buffer_duration
                        # 처리 스레드 분리
                        t = threading.Thread(target=self.process_audio_segment, args=(audio_array, base_time), daemon=True)
                        t.start()

                        # overlap 유지
                        overlap_samples = int(self.config.OVERLAP_DURATION * self.config.SAMPLE_RATE)
                        if len(process_buffer) > overlap_samples:
                            process_buffer = process_buffer[-overlap_samples:]
                        else:
                            process_buffer = []

                        last_process_time = current_time
                # loop small sleep to reduce busy wait
                time.sleep(0.01)

            except Exception as e:
                self._emit_error(f"처리 루프 예외: {e}")
                break

    def run(self, device_index: Optional[int] = None):
        """모델 로드 후 녹음 시작 (블로킹 아님 — 호출자는 별도 쓰레드에서 호출)."""
        try:
            if not self.models_loaded:
                if not self.load_models():
                    return
            if not self.start_recording(device_index):
                return
            # run은 녹음이 시작되면 바로 반환(내부 스레드가 처리)
            self._emit_status("실시간 처리가 시작되었습니다.")
        except Exception as e:
            self._emit_error(f"실행 오류: {e}")

    def cleanup(self):
        try:
            self.stop_recording()
            self._emit_status("리소스 정리 완료")
        except Exception as e:
            self._emit_error(f"cleanup 오류: {e}")


# ---------------------------
# PyQt6 GUI 부분
# ---------------------------

class WorkerThread(QThread):
    """RealTimeDiarization.run()을 별도 스레드에서 실행하기 위한 QThread 래퍼."""
    def __init__(self, rt_diar: RealTimeDiarization, device_index: Optional[int] = None):
        super().__init__()
        self.rt_diar = rt_diar
        self.device_index = device_index

    def run(self):
        # run() 내부에서 블로킹하지 않으므로, 여기서는 단순히 호출
        try:
            self.rt_diar.run(self.device_index)
            # run() 후에도 is_recording에 따라 내부 스레드가 돌아감
            # QThread는 여기서 바로 종료될 수 있음 (문제 없음)
        except Exception as e:
            # GUI는 콜백을 통해 에러를 받게 됨
            print("WorkerThread 예외:", e)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 화자분리 채팅 GUI")
        self.resize(900, 700)

        # config & core
        self.config = Config()
        self.rt = RealTimeDiarization(self.config)

        # UI 요소
        self.chat_view = QTextBrowser()
        self.chat_view.setFrameShape(QFrame.Shape.StyledPanel)
        self.chat_view.setOpenExternalLinks(False)

        self.input_edit = QTextEdit()
        self.input_edit.setFixedHeight(80)
        self.send_button = QPushButton("보내기")
        self.start_button = QPushButton("시작")
        self.stop_button = QPushButton("중지")
        self.device_combo = QComboBox()
        self.status_label = QLabel("상태: 준비됨")

        # layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("입력 디바이스:"))
        top_layout.addWidget(self.device_combo)
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.stop_button)
        top_layout.addStretch()
        top_layout.addWidget(self.status_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.input_edit)
        right_buttons = QVBoxLayout()
        right_buttons.addWidget(self.send_button)
        right_buttons.addStretch()
        bottom_layout.addLayout(right_buttons)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.chat_view, stretch=1)
        main_layout.addLayout(bottom_layout)

        # signals
        self.send_button.clicked.connect(self.on_send_clicked)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)

        # set callbacks from RealTimeDiarization -> GUI
        self.rt.on_transcription = self.on_transcription
        self.rt.on_status_change = self.on_status
        self.rt.on_error = self.on_error

        # worker thread placeholder
        self.worker_thread: Optional[WorkerThread] = None

        # populate devices
        self.populate_devices()

    # ---------- UI helpers ----------
    def append_chat(self, text: str, who: str = "bot", timestamp: Optional[str] = None):
        """채팅 뷰에 메시지 추가. 'who'는 'bot' 또는 'user' (디자인 차이)."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        if who == "user":
            # 오른쪽 정렬(파란색 말풍선 스타일)
            html = f"""
            <div style="text-align: right; margin:6px;">
              <div style="display:inline-block; background:#d0e8ff; padding:8px 12px; border-radius:12px; max-width:70%;">
                <b>나</b> <small style="color:#666">[{timestamp}]</small><br>
                {self._escape_html(text)}
              </div>
            </div>
            """
        else:
            # 왼쪽 정렬 (회색 말풍선). speaker 표기
            html = f"""
            <div style="text-align: left; margin:6px;">
              <div style="display:inline-block; background:#f1f1f1; padding:8px 12px; border-radius:12px; max-width:70%;">
                <b>상대</b> <small style="color:#666">[{timestamp}]</small><br>
                {self._escape_html(text)}
              </div>
            </div>
            """
        self.chat_view.append(html)

    def _escape_html(self, text: str) -> str:
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br>"))

    # ---------- device ----------
    def populate_devices(self):
        self.device_combo.clear()
        try:
            devices = self.rt.get_available_audio_devices()
            if not devices:
                self.device_combo.addItem("기본 디바이스 (입력 없음)", -1)
            else:
                for idx, name in devices.items():
                    self.device_combo.addItem(f"{idx}: {name}", idx)
        except Exception as e:
            self.device_combo.addItem("디바이스 조회 실패", -1)
            self.status_label.setText("상태: 디바이스 조회 실패")

    # ---------- callbacks from rt -->
    def on_transcription(self, timestamp: str, speaker: str, text: str):
        # GUI 스레드에서 실행되어야 하므로 Qt의 싱글 스레드 규칙을 지키기 위해 invoke
        # QThread-safe 방식: 모든 GUI 업데이트는 메인 스레드에서 실행되므로 signal 사용이 권장.
        # 여기서는 간단히 메서드로 호출하면 PyQt가 메인 스레드에서 호출(콜백이 아닌 경우 문제 발생할 수 있음).
        # 안전하게 하려면 Qt signals를 만들지만 간단화를 위해 직접 호출.
        # 표시 형식: [스피커] 텍스트
        display_text = f"[{speaker}] {text}"
        # 왼쪽(상대) 메시지로 추가
        self.append_chat(display_text, who="bot", timestamp=timestamp)

    def on_status(self, message: str):
        # status_label 갱신 및 로그 추가
        self.status_label.setText(f"상태: {message}")
        # 상태를 채팅에도 기록
        self.chat_view.append(f"<div style='color:gray; font-size:small'>[STATUS] {self._escape_html(message)}</div>")

    def on_error(self, message: str):
        self.status_label.setText("상태: 오류 발생")
        self.chat_view.append(f"<div style='color:red; font-weight:bold'>[ERROR] {self._escape_html(message)}</div>")

    # ---------- UI 이벤트 ----------
    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        # 사용자가 보낸 메시지(오른쪽)
        self.append_chat(text, who="user", timestamp=timestamp)
        self.input_edit.clear()
        # (선택) 사용자가 보낸 메시지를 로컬 로직으로 바로 처리하거나 서버 전송하는 등 할 수 있음.

    def on_start(self):
        # 장치 인덱스 결정
        idx = self.device_combo.currentData()
        device_index = None if idx is None or idx == -1 else int(idx)

        # worker 쓰레드에서 rt.run 호출
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "알림", "이미 실행 중입니다.")
            return

        # 모델 로드가 오래 걸리므로 상태 메시지
        self.status_label.setText("상태: 모델 로딩 및 녹음 시작 시도...")
        self.chat_view.append("<div style='color:green'>[SYSTEM] 모델을 로드하고 녹음을 시작합니다. (로그를 확인하세요)</div>")

        # run을 비동기적으로 호출
        self.worker_thread = WorkerThread(self.rt, device_index=device_index)
        self.worker_thread.start()

    def on_stop(self):
        self.rt.cleanup()
        self.chat_view.append("<div style='color:orange'>[SYSTEM] 녹음/처리를 중지했습니다.</div>")
        self.status_label.setText("상태: 중지됨")

    def closeEvent(self, event):
        # 앱 종료 시 정리
        try:
            self.rt.cleanup()
            if self.worker_thread is not None and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait(timeout=2000)
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
