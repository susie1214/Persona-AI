# ui/meeting_notes.py
# -*- coding: utf-8 -*-
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QLineEdit
)
import os, datetime

from core.offline_meeting import process_audio_file

class _SummWorker(QObject):
    sig_done = Signal(dict)
    sig_error = Signal(str)
    def __init__(self, path, settings):
        super().__init__()
        self.path = path
        self.settings = settings or {}
    def run(self):
        try:
            # mp3/wav/mp4/m4a 등 ffmpeg로 처리됨
            res = process_audio_file(
                self.path, asr_model="medium", use_gpu=(os.getenv("FORCE_CPU","0")!="1"),
                diarize=True, settings=self.settings
            )
            self.sig_done.emit(res)
        except Exception as e:
            self.sig_error.emit(str(e))

class MeetingNotesView(QWidget):
    """업로드 → 요약/회의록 → TXT/MD/HTML 저장 & 클립보드 복사"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings_cache = {}  # meeting_settings에서 가져다 넣어도 됨
        L = QVBoxLayout(self)

        # 헤더
        head = QHBoxLayout()
        head.addWidget(QLabel("회의 자료 업로드(.wav/.mp3/.m4a/.mp4 등)"))
        head.addStretch(1)
        self.btn_upload = QPushButton("파일 선택")
        head.addWidget(self.btn_upload)
        L.addLayout(head)

        # 진행 표시
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        L.addWidget(self.progress)

        # 제목 입력/수정
        ti = QHBoxLayout()
        ti.addWidget(QLabel("제목"))
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("회의록 제목 (자동 생성 가능)")
        ti.addWidget(self.edit_title)
        L.addLayout(ti)

        # 결과 표시
        self.txt = QTextEdit()
        self.txt.setReadOnly(False)
        L.addWidget(self.txt, 1)

        # 액션 버튼
        actions = QHBoxLayout()
        self.btn_copy = QPushButton("복사")
        self.btn_save_txt = QPushButton("TXT 저장")
        self.btn_save_md = QPushButton("Markdown 저장")
        self.btn_save_html = QPushButton("HTML 저장")
        actions.addWidget(self.btn_copy)
        actions.addStretch(1)
        actions.addWidget(self.btn_save_txt)
        actions.addWidget(self.btn_save_md)
        actions.addWidget(self.btn_save_html)
        L.addLayout(actions)

        # 연결
        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_copy.clicked.connect(self.copy_to_clip)
        self.btn_save_txt.clicked.connect(lambda: self.save_text("txt"))
        self.btn_save_md.clicked.connect(lambda: self.save_text("md"))
        self.btn_save_html.clicked.connect(lambda: self.save_text("html"))

        self.setStyleSheet("""
            QTextEdit { background:#FAFFFA; }
            QPushButton { padding:6px 12px; }
        """)

        self._last_markdown = ""
        self._last_txt = ""

    # ---- actions ----
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "회의 자료 파일 선택", "",
            "Audio/Video (*.wav *.mp3 *.m4a *.flac *.mp4 *.mkv *.aac);;All Files (*)"
        )
        if not path: return
        self.progress.setVisible(True)
        self.progress.setRange(0,0)
        self.txt.clear()

        self.thread = QThread(self)
        self.worker = _SummWorker(path, self._settings_cache)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.sig_done.connect(self.on_done)
        self.worker.sig_error.connect(self.on_err)
        self.worker.sig_done.connect(self.thread.quit)
        self.worker.sig_error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_done(self, result: dict):
        self.progress.setVisible(False)
        md = result.get("markdown","")
        title = result.get("title","회의록")
        self._last_markdown = md
        # TXT 버전(마크다운 제거 간이)
        self._last_txt = md.replace("#","").replace("**","").replace("`","")
        if not self.edit_title.text().strip():
            self.edit_title.setText(title + " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.txt.setPlainText(md)
        QMessageBox.information(self, "완료", "회의록을 생성했습니다.")

    def on_err(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "오류", msg)

    def copy_to_clip(self):
        text = self.txt.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "복사", "복사할 내용이 없습니다.")
            return
        self.clipboard().setText(text)
        QMessageBox.information(self, "복사", "클립보드로 복사했습니다.")

    def clipboard(self):
        from PySide6.QtWidgets import QApplication
        return QApplication.instance().clipboard()

    def save_text(self, kind: str):
        text = self.txt.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "저장", "저장할 내용이 없습니다.")
            return
        default = (self.edit_title.text().strip() or "회의록").replace(" ","_")
        if kind == "txt":
            path, _ = QFileDialog.getSaveFileName(self, "TXT 저장", f"{default}.txt", "Text (*.txt)")
            data = self._last_txt or text
        elif kind == "md":
            path, _ = QFileDialog.getSaveFileName(self, "Markdown 저장", f"{default}.md", "Markdown (*.md)")
            data = self._last_markdown or text
        else:  # html
            path, _ = QFileDialog.getSaveFileName(self, "HTML 저장", f"{default}.html", "HTML (*.html)")
            data = "<html><body><pre style='white-space:pre-wrap'>" + \
                   (self._last_markdown or text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") + \
                   "</pre></body></html>"
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            QMessageBox.information(self, "저장", f"파일로 저장했습니다:\n{path}")

    # 외부에서 회의록 설정을 주입하고 싶을 때 사용
    def set_settings(self, settings: dict):
        self._settings_cache = settings or {}
