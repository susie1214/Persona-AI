# ui/meeting_notes.py
# -*- coding: utf-8 -*-
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QLineEdit,
    QCheckBox, QComboBox
)
import os, datetime

from core.offline_meeting import process_audio_file

from core.summarizer import actions_from_segments

class _SummWorker(QObject):

    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, path, settings, use_llm_summary=True, llm_backend=None, speaker_manager=None):
        super().__init__()
        self.path = path
        self.settings = settings or {}
        self.use_llm_summary = use_llm_summary
        self.llm_backend = llm_backend
        self.speaker_manager = speaker_manager

    def run(self):
        try:
            # mp3/wav/mp4/m4a 등 ffmpeg로 처리됨
            res = process_audio_file(
                self.path,
                asr_model="medium",
                use_gpu=(os.getenv("FORCE_CPU","0")!="1"),
                diarize=True,
                use_llm_summary=self.use_llm_summary,
                llm_backend=self.llm_backend,
                settings=self.settings,
                speaker_manager=self.speaker_manager
            )
            
            # print(f"[DEBUG - meeting_notes] res : {res}")
            
            self.sig_done.emit(res)
            
        except Exception as e:
            self.sig_error.emit(str(e))

class MeetingNotesView(QWidget):
    """업로드 → 요약/회의록 → TXT/MD/HTML 저장 & 클립보드 복사"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_console = parent
        self._settings_cache = {}  # meeting_settings에서 가져다 넣어도 됨
        L = QVBoxLayout(self)

        # 헤더
        head = QHBoxLayout()
        head.addWidget(QLabel("회의 자료 업로드(.wav/.mp3/.m4a/.mp4 등)"))
        head.addStretch(1)
        self.btn_upload = QPushButton("파일 선택")
        head.addWidget(self.btn_upload)
        L.addLayout(head)

        # LLM 요약 옵션
        llm_options = QHBoxLayout()
        self.chk_llm_summary = QCheckBox("🤖 AI 요약 사용 (LLM)")
        self.chk_llm_summary.setChecked(True)  # 기본값: LLM 사용
        self.chk_llm_summary.setToolTip("LLM을 사용하여 회의록을 지능적으로 요약합니다 (OpenAI API 키 필요)")
        llm_options.addWidget(self.chk_llm_summary)

        self.combo_llm_backend = QComboBox()
        self.combo_llm_backend.addItems([
            "openai:gpt-4o-mini",
            "openai:gpt-4o",
        ])
        self.combo_llm_backend.setToolTip("사용할 LLM 모델 선택")
        llm_options.addWidget(self.combo_llm_backend)
        llm_options.addStretch()

        self.chk_llm_summary.toggled.connect(lambda checked: self.combo_llm_backend.setEnabled(checked))

        L.addLayout(llm_options)

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

        # 결과 표시 (요약/전사 분리)
        L.addWidget(QLabel("회의 요약"))
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(False)
        L.addWidget(self.txt_summary, 1)

        L.addWidget(QLabel("회의 전체 전사"))
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True) # 전사는 읽기 전용
        L.addWidget(self.txt_transcript, 1)

        # 액션 버튼
        actions = QHBoxLayout()
        self.btn_copy = QPushButton("복사")
        self.chk_save_summary = QCheckBox("요약 포함")
        self.chk_save_summary.setChecked(True)
        self.chk_save_transcript = QCheckBox("전사 포함")
        self.chk_save_transcript.setChecked(True)

        self.btn_save_txt = QPushButton("TXT 저장")
        self.btn_save_md = QPushButton("Markdown 저장")
        self.btn_save_html = QPushButton("HTML 저장")
        actions.addWidget(self.btn_copy)
        actions.addStretch(1)
        actions.addWidget(self.chk_save_summary)
        actions.addWidget(self.chk_save_transcript)
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
        self.edit_title.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0,0)
        self.txt_summary.clear()
        self.txt_transcript.clear()

        # LLM 요약 옵션 가져오기
        use_llm = self.chk_llm_summary.isChecked()
        llm_backend = self.combo_llm_backend.currentText() if use_llm else None

        # speaker_manager 가져오기 (main_console에서)
        speaker_manager = None
        if self.main_console and hasattr(self.main_console, 'speaker_manager'):
            speaker_manager = self.main_console.speaker_manager

        self.thread = QThread(self)
        self.worker = _SummWorker(
            path,
            self._settings_cache,
            use_llm_summary=use_llm,
            llm_backend=llm_backend,
            speaker_manager=speaker_manager
        )
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
        summary = result.get("summary", "")
        transcript = result.get("markdown", "")
        title = result.get("title", "회의록")
        segments = result.get("segments", [])

        self.txt_summary.setPlainText(summary)
        self.txt_transcript.setPlainText(transcript)

        self._last_markdown = transcript
        self._last_txt = transcript.replace("#", "").replace("**", "").replace("`", "")

        if not self.edit_title.text().strip():
            self.edit_title.setText(title + " 회의록")

        # RAG 저장을 위해 main_console의 메서드 호출
        if self.main_console and hasattr(self.main_console, '_save_summary_to_rag'):
            action_items = actions_from_segments(segments)
            self.main_console._save_summary_to_rag(summary, action_items)
            # 사용자에게 RAG 저장 사실 알림 (옵션)
            self.main_console.on_status("✓ 파일 요약본이 RAG에 저장되었습니다.")

        # 화자 매핑 탭 새로고침 (파일 처리 중 새로운 화자가 식별되었을 수 있음)
        if self.main_console and hasattr(self.main_console, 'meeting_settings'):
            if hasattr(self.main_console.meeting_settings, 'speaker_tab'):
                self.main_console.meeting_settings.speaker_tab.load_speakers()
                self.main_console.on_status("✓ 화자 매핑 정보가 업데이트되었습니다.")

        QMessageBox.information(self, "완료", "회의록을 생성했습니다.")

    def on_err(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "오류", msg)

    def copy_to_clip(self):
        content = []
        if self.chk_save_summary.isChecked():
            content.append("--- 요약 ---\n" + self.txt_summary.toPlainText())
        if self.chk_save_transcript.isChecked():
            content.append("--- 전사 ---\n" + self.txt_transcript.toPlainText())

        text_to_copy = "\n\n".join(content)

        if not text_to_copy.strip():
            QMessageBox.warning(self, "복사", "복사할 내용이 없습니다.")
            return
        self.clipboard().setText(text_to_copy)
        QMessageBox.information(self, "복사", "클립보드로 복사했습니다.")

    def clipboard(self):
        # from PySide6.QtWidgets import QApplication
        # return QApplication.instance().clipboard()
        from PySide6.QtGui import QGuiApplication
        return QGuiApplication.clipboard()

    def save_text(self, kind: str):
        content = []
        summary_text = self.txt_summary.toPlainText()
        transcript_text = self.txt_transcript.toPlainText()

        if self.chk_save_summary.isChecked():
            content.append("--- 요약 ---\n" + summary_text)
        if self.chk_save_transcript.isChecked():
            content.append("--- 전사 ---\n" + transcript_text)

        text_to_save = "\n\n".join(content)

        if not text_to_save.strip():
            QMessageBox.warning(self, "저장", "저장할 내용이 없습니다.")
            return

        default = (self.edit_title.text().strip() or "회의록").replace(" ", "_")
        
        if kind == "txt":
            path, _ = QFileDialog.getSaveFileName(self, "TXT 저장", f"{default}.txt", "Text (*.txt)")
            data = text_to_save
        elif kind == "md":
            path, _ = QFileDialog.getSaveFileName(self, "Markdown 저장", f"{default}.md", "Markdown (*.md)")
            data = text_to_save # For now, just save as plain text. Can be improved to be more markdown-like.
        else:  # html
            path, _ = QFileDialog.getSaveFileName(self, "HTML 저장", f"{default}.html", "HTML (*.html)")
            summary_html = self.txt_summary.toHtml()
            transcript_html = self.txt_transcript.toHtml()
            html_content = []
            if self.chk_save_summary.isChecked():
                html_content.append(f"<h1>요약</h1>{summary_html}")
            if self.chk_save_transcript.isChecked():
                html_content.append(f"<h1>전사</h1>{transcript_html}")
            data = "<html><body>" + "<br>".join(html_content) + "</body></html>"

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            QMessageBox.information(self, "저장", f"파일로 저장했습니다:\n{path}")

    # 외부에서 회의록 설정을 주입하고 싶을 때 사용
    def set_settings(self, settings: dict):
        self._settings_cache = settings or {}

    def update_notes(self, summary_html: str, transcript_text: str):
        """외부에서 요약 및 전사 내용을 직접 업데이트"""
        self.txt_summary.setHtml(summary_html)
        self.txt_transcript.setPlainText(transcript_text)
        self.edit_title.setText("실시간 회의록 - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

