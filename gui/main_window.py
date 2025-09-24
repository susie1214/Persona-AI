# gui/main_window.py
import sys
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QTextEdit,
    QPushButton, QLabel, QComboBox, QMessageBox, QFrame
)
from PyQt6.QtCore import QThread

from realtime_diar.config import Config
from realtime_diar.core import RealTimeDiarization


class WorkerThread(QThread):
    def __init__(self, rt_diar: RealTimeDiarization, device_index: Optional[int] = None):
        super().__init__()
        self.rt_diar = rt_diar
        self.device_index = device_index

    def run(self):
        self.rt_diar.run(self.device_index)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 화자분리 채팅 GUI")
        self.resize(900, 700)

        self.config = Config()
        self.rt = RealTimeDiarization(self.config)

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

        self.send_button.clicked.connect(self.on_send_clicked)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)

        self.rt.on_transcription = self.on_transcription
        self.rt.on_status_change = self.on_status
        self.rt.on_error = self.on_error

        self.worker_thread: Optional[WorkerThread] = None
        self.populate_devices()

    def append_chat(self, text: str, who: str = "bot", timestamp: Optional[str] = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        if who == "user":
            html = f"""
            <div style="text-align: right; margin:6px;">
              <div style="display:inline-block; background:#d0e8ff; padding:8px 12px; border-radius:12px; max-width:70%;">
                <b>나</b> <small style="color:#666">[{timestamp}]</small><br>
                {self._escape_html(text)}
              </div>
            </div>
            """
        else:
            html = f"""
            <div style="text-align: left; margin:6px;">
              <div style="display:inline-block; background:#f1f1f1; padding:8px 12px; border-radius:12px; max-width:70%;">
                <text style="color:#ffffff">{self._escape_html(text)}</text>
              </div>
            </div>
            """
        self.chat_view.append(html)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def _escape_html(self, text: str) -> str:
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br>"))

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

    def on_transcription(self, timestamp: str, speaker: str, text: str):
        display_text = f"[{speaker}] {text}"
        self.append_chat(display_text, who="bot", timestamp=timestamp)

    def on_status(self, message: str):
        self.status_label.setText(f"상태: {message}")
        self.chat_view.append(f"<div style='color:gray; font-size:small'>[STATUS] {self._escape_html(message)}</div>")
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def on_error(self, message: str):
        self.status_label.setText("상태: 오류 발생")
        self.chat_view.append(f"<div style='color:red; font-weight:bold'>[ERROR] {self._escape_html(message)}</div>")
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append_chat(text, who="user", timestamp=timestamp)
        self.input_edit.clear()

    def on_start(self):
        idx = self.device_combo.currentData()
        device_index = None if idx is None or idx == -1 else int(idx)
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "알림", "이미 실행 중입니다.")
            return
        self.status_label.setText("상태: 모델 로딩 및 녹음 시작 시도...")
        self.chat_view.append("<div style='color:green'>[SYSTEM] 모델을 로드하고 녹음을 시작합니다.</div>")
        self.worker_thread = WorkerThread(self.rt, device_index=device_index)
        self.worker_thread.start()

    def on_stop(self):
        self.rt.cleanup()
        self.chat_view.append("<div style='color:orange'>[SYSTEM] 녹음/처리를 중지했습니다.</div>")
        self.status_label.setText("상태: 중지됨")

    def closeEvent(self, event):
        try:
            self.rt.cleanup()
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait()
        except Exception:
            pass
        super().closeEvent(event)
