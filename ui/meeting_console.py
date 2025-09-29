# -*- coding: utf-8 -*-
# ui/meeting_console.py
import os, datetime
from PySide6.QtCore import Qt, QTimer, Signal, QDateTime
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QLabel,
    QTabWidget,
    QSplitter,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QDateTimeEdit,
    QTextEdit,
    QDockWidget,
)

from ui.survey_wizard import PersonaSurveyWizard
from ui.chat_dock import ChatDock
from ui.meeting_notes import MeetingNotesView

from core.audio import AudioWorker, Segment, MeetingState, fmt_time, now_str
from core.diarization import DiarizationWorker
from core.summarizer import simple_summarize, extract_actions
from core.rag_store import RagStore
from core.adapter import AdapterManager

THEME = {
    "bg": "#e6f5e6",
    "pane": "#99cc99",
    "light_bg": "#fafffa",
    "btn": "#ffe066",
    "btn_hover": "#ffdb4d",
    "btn_border": "#cccc99",
}
HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_MODEL = "medium"


# ---------------- Participant dialog ----------------
class ParticipantDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("회의 참가자 등록")
        self.resize(320, 150)
        layout = QVBoxLayout(self)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("참가자 이름 입력 (예: 신현택)")
        layout.addWidget(self.edit_name)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_name(self) -> str:
        return self.edit_name.text().strip()


# ---------------- Main window ----------------
class MeetingConsole(QMainWindow):
    sig_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 보조 서비스")
        self.resize(1280, 860)

        # state / workers
        self.state = MeetingState()
        self.audio_worker = AudioWorker(self.state)
        self.audio_worker.sig_transcript.connect(self.on_segment)
        self.audio_worker.sig_status.connect(self.on_status)

        self.diar_worker = DiarizationWorker(self.state)
        self.diar_worker.sig_status.connect(self.on_status)
        self.diar_worker.sig_diar_done.connect(self.on_diar_done)

        self.rag = RagStore()
        self.adapter = AdapterManager()

        # tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_live_tab()
        # self._build_timeline_tab()
        # self._build_qa_tab()
        self._build_action_tab()
        self._build_settings_tab()

        # 회의록 탭 (업로드 → 요약/회의록 저장/복사)
        self.meeting_notes = MeetingNotesView(self)
        self.tabs.addTab(self.meeting_notes, "회의록")

        self._apply_theme()

        # 우측 개인 챗봇 도크
        self.chat_dock = QDockWidget("Personal Chatbot", self)
        self.chat_panel = ChatDock()
        self.chat_dock.setWidget(self.chat_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)

        # live 미리보기 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(1000)

        # 설문 마법사(최초 1회)
        self.survey = PersonaSurveyWizard(parent=self)
        self.survey.show()

    # ---------------- UI builders ----------------
    def _build_live_tab(self):
        self.live_root = QWidget()
        L = QVBoxLayout(self.live_root)

        # top bar
        bar = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_sum = QPushButton("Summarize")
        self.btn_add2rag = QPushButton("Index to RAG")
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addStretch(1)
        bar.addWidget(self.btn_sum)
        bar.addWidget(self.btn_add2rag)
        L.addLayout(bar)

        # mid bar
        mid = QHBoxLayout()
        mid.addWidget(QLabel("Forced Speaker:"))
        self.cmb_forced = QComboBox()
        self.cmb_forced.addItem("None")
        mid.addWidget(self.cmb_forced)

        self.chk_diar = QCheckBox("Auto Diarization (pyannote)")
        self.chk_diar.setChecked(self.state.diarization_enabled)
        mid.addWidget(self.chk_diar)
        L.addLayout(mid)

        # split
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        Lv = QVBoxLayout(left)
        self.list_chat = QListWidget()
        Lv.addWidget(self.list_chat)
        splitter.addWidget(left)

        right = QWidget()
        Rv = QVBoxLayout(right)
        Rv.addWidget(QLabel("Status"))
        self.txt_status = QPlainTextEdit()
        self.txt_status.setReadOnly(True)
        Rv.addWidget(self.txt_status)
        Rv.addWidget(QLabel("Preview (Summary)"))
        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        Rv.addWidget(self.txt_preview)
        splitter.addWidget(right)
        splitter.setSizes([900, 380])
        L.addWidget(splitter)

        self.tabs.addTab(self.live_root, "Live")

        # events
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_sum.clicked.connect(self.on_summarize)
        self.btn_add2rag.clicked.connect(self.on_index_to_rag)
        self.cmb_forced.currentTextChanged.connect(self.on_forced_changed)
        self.chk_diar.stateChanged.connect(self.on_diar_toggle)

    def _build_timeline_tab(self):
        self.timeline_root = QWidget()
        L = QVBoxLayout(self.timeline_root)
        self.timeline = QListWidget()
        L.addWidget(self.timeline)
        self.tabs.addTab(self.timeline_root, "Timeline")

    def _build_qa_tab(self):
        root = QWidget()
        L = QVBoxLayout(root)

        top = QHBoxLayout()
        self.edit_q = QLineEdit()
        self.edit_q.setPlaceholderText("질문 입력 (RAG + Tone)")
        self.btn_ans = QPushButton("Answer")
        top.addWidget(self.edit_q)
        top.addWidget(self.btn_ans)
        L.addLayout(top)

        self.cmb_adapter = QComboBox()
        self.cmb_adapter.addItem("None")
        self.btn_load_base = QPushButton("Load Base (QLoRA)")
        self.btn_add_adapter = QPushButton("Add Adapter…")
        tool = QHBoxLayout()
        tool.addWidget(QLabel("Tone Adapter:"))
        tool.addWidget(self.cmb_adapter)
        tool.addWidget(self.btn_load_base)
        tool.addWidget(self.btn_add_adapter)
        L.addLayout(tool)

        self.txt_ans = QTextEdit()
        self.txt_ans.setReadOnly(True)
        L.addWidget(self.txt_ans)
        self.tabs.addTab(root, "Q&A")

        self.btn_ans.clicked.connect(self.on_answer)
        self.btn_load_base.clicked.connect(self.on_load_base)
        self.btn_add_adapter.clicked.connect(self.on_add_adapter)
        self.cmb_adapter.currentTextChanged.connect(self.on_adapter_changed)

    def _build_action_tab(self):
        root = QWidget()
        L = QVBoxLayout(root)

        L.addWidget(QLabel("회의 전체요약"))
        self.txt_summary = QTextEdit()
        L.addWidget(self.txt_summary)

        L.addWidget(QLabel("Action Items"))
        self.txt_actions = QTextEdit()
        L.addWidget(self.txt_actions)

        row = QHBoxLayout()
        row.addWidget(QLabel("다음 회의 시작"))
        self.dt_start = QDateTimeEdit()
        self.dt_start.setDateTime(QDateTime.currentDateTime().addDays(7))
        self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm")
        row.addWidget(self.dt_start)

        row.addWidget(QLabel("종료"))
        self.dt_end = QDateTimeEdit()
        self.dt_end.setDateTime(QDateTime.currentDateTime().addDays(7).addSecs(3600))
        self.dt_end.setDisplayFormat("yyyy-MM-dd HH:mm")
        row.addWidget(self.dt_end)

        self.btn_sched_memo = QPushButton("Make Schedule Memo")
        row.addWidget(self.btn_sched_memo)
        L.addLayout(row)

        L.addWidget(QLabel("다음 회의 메모"))
        self.txt_sched = QTextEdit()
        L.addWidget(self.txt_sched)

        self.tabs.addTab(root, "Action & Schedule")
        self.btn_sched_memo.clicked.connect(self.on_make_schedule)

    def _build_settings_tab(self):
        root = QWidget()
        F = QFormLayout(root)

        self.cmb_asr = QComboBox()
        for m in ["tiny", "base", "small", "medium", "large-v3"]:
            self.cmb_asr.addItem(m)
        self.cmb_asr.setCurrentText(DEFAULT_MODEL)

        self.chk_gpu = QCheckBox("Use GPU if available")
        self.chk_gpu.setChecked(True)

        self.chk_diar2 = QCheckBox("Auto Diarization")
        self.chk_diar2.setChecked(False)

        self.edit_hf = QLineEdit()
        self.edit_hf.setPlaceholderText(f"{HF_TOKEN_ENV} (HuggingFace token)")

        self.btn_add_participant = QPushButton("참가자 추가")
        self.cmb_map_id = QComboBox()
        self.cmb_map_name = QComboBox()

        # preload names
        for n in ["신현택", "박길실", "조진경"]:
            self.cmb_map_name.addItem(n)
            # live 탭 스피커 강제에도 반영
            if self.cmb_forced.findText(n) < 0:
                self.cmb_forced.addItem(n)

        F.addRow("Whisper Model", self.cmb_asr)
        F.addRow("", self.chk_gpu)
        F.addRow("Auto Diarization", self.chk_diar2)
        F.addRow("HF Token", self.edit_hf)
        F.addRow("", self.btn_add_participant)
        F.addRow(QLabel("pyannote SpeakerID"), self.cmb_map_id)
        F.addRow(QLabel("Participant Name"), self.cmb_map_name)

        self.tabs.addTab(root, "Settings")

        self.btn_add_participant.clicked.connect(self.on_add_participant)
        self.chk_diar2.stateChanged.connect(self.on_diar_toggle_settings)

    def _apply_theme(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{ background-color: {THEME['bg']}; }}
            QTabWidget::pane {{ border: 2px solid {THEME['pane']}; }}
            QPushButton {{
                background-color: {THEME['btn']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {THEME['btn_hover']}; }}
            QListWidget, QTextEdit, QPlainTextEdit {{
                background-color: {THEME['light_bg']};
                border: 1px solid {THEME['pane']};
            }}
            QLineEdit, QComboBox, QDateTimeEdit {{
                background-color: #ffffff;
                border: 1px solid {THEME['pane']};
                border-radius: 6px;
                padding: 4px 6px;
            }}
            """
        )

    # ---------------- Events ----------------
    def on_start(self):
        # apply settings
        self.state.use_gpu = self.chk_gpu.isChecked()
        self.state.asr_model = self.cmb_asr.currentText()
        self.state.diarization_enabled = (
            self.chk_diar.isChecked() or self.chk_diar2.isChecked()
        )
        tok = self.edit_hf.text().strip()
        if tok:
            os.environ[HF_TOKEN_ENV] = tok

        try:
            self.audio_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Start failed: {e}")
            return

        if self.state.diarization_enabled:
            self.diar_worker.start()

        self.on_status("Started.")

    def on_stop(self):
        try:
            self.audio_worker.stop()
            self.diar_worker.stop()
        except Exception:
            pass
        self.on_status("Stopped.")

    def on_summarize(self):
        self.state.summary = simple_summarize(self.state.live_segments, max_len=12)
        self.state.actions = extract_actions(self.state.live_segments)
        self.txt_summary.setText(self.state.summary)
        self.txt_actions.setText(
            "\n".join(self.state.actions) if self.state.actions else "(액션아이템 없음)"
        )
        QMessageBox.information(self, "Done", "요약/액션아이템 생성 완료")

    def on_index_to_rag(self):
        if not self.rag.ok:
            QMessageBox.warning(self, "RAG", "Qdrant 사용 불가(미설치/연결 실패).")
            return
        # 최근 50줄만 인덱싱 (데모)
        self.rag.upsert_segments(self.state.live_segments[-50:])
        QMessageBox.information(self, "RAG", "최근 발언을 RAG 인덱싱했습니다.")

    def on_forced_changed(self, text):
        self.state.forced_speaker_name = None if (text == "None") else text

    def on_diar_toggle(self):
        self.state.diarization_enabled = self.chk_diar.isChecked()
        self.chk_diar2.setChecked(self.state.diarization_enabled)

    def on_diar_toggle_settings(self):
        self.state.diarization_enabled = self.chk_diar2.isChecked()
        self.chk_diar.setChecked(self.state.diarization_enabled)

    def on_add_participant(self):
        dlg = ParticipantDialog(self)
        if dlg.exec():
            name = dlg.get_name()
            if not name:
                return
            spk_id = f"SPEAKER_{len(self.state.speaker_map):02d}"
            self.state.speaker_map[spk_id] = name
            self.cmb_map_id.addItem(spk_id)
            if self.cmb_forced.findText(name) < 0:
                self.cmb_forced.addItem(name)
            if self.cmb_map_name.findText(name) < 0:
                self.cmb_map_name.addItem(name)
            QMessageBox.information(self, "등록 완료", f"{spk_id} → {name}")

    def on_make_schedule(self):
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("HH:mm")
        memo = (
            f"다음 회의: {s} ~ {e}\n"
            f"참석자: {', '.join(sorted(set([seg.speaker_name for seg in self.state.live_segments if seg.speaker_name!='Unknown'])))}\n"
            f"안건: 액션아이템 점검"
        )
        self.state.schedule_note = memo
        self.txt_sched.setText(memo)
        QMessageBox.information(self, "메모 생성", "다음 회의 메모를 작성했습니다.")

    def on_answer(self):
        q = self.edit_q.text().strip()
        if not q:
            return
        ctx = self.rag.search(q, topk=5) if self.rag.ok else []
        lines = [f"- [{c.get('speaker_name','?')}] {c.get('text','')}" for c in ctx]
        ctx_block = "\n".join(lines) if lines else "(근거 없음)"
        ans = self.adapter.respond(f"Q: {q}")
        self.txt_ans.setText(f"{ans}\n\n[근거]\n{ctx_block}")

    def on_load_base(self):
        if not self.adapter.available:
            QMessageBox.warning(self, "QLoRA", "transformers/peft 미설치로 비활성.")
            return
        ok = self.adapter.load_base()
        QMessageBox.information(
            self, "QLoRA", "Base loaded." if ok else "Base load 실패."
        )

    def on_add_adapter(self):
        if not self.adapter.available or self.adapter.base_model is None:
            QMessageBox.warning(self, "QLoRA", "Base 모델 먼저 로드하세요.")
            return
        name = f"adapter_{self.cmb_adapter.count()}"
        adapter_path = os.getenv("QLORA_ADAPTER_PATH", "")
        ok = self.adapter.load_adapter(name, adapter_path) if adapter_path else False
        self.cmb_adapter.addItem(name)
        QMessageBox.information(
            self, "QLoRA", f"Adapter '{name}' 추가 {'성공' if ok else '(더미 등록)'}."
        )

    def on_adapter_changed(self, name):
        self.adapter.set_active(None if name == "None" else name)

    # ---------------- Signals ----------------
    def on_status(self, msg: str):
        self.txt_status.appendPlainText(f"{now_str()}  {msg}")

    def on_segment(self, seg: Segment):
        self.state.live_segments.append(seg)
        self.list_chat.addItem(QListWidgetItem(f"[{seg.speaker_name}] {seg.text}"))
        self.list_chat.scrollToBottom()
        self.timeline.addItem(
            QListWidgetItem(
                f"{fmt_time(seg.start)}~{fmt_time(seg.end)} | {seg.speaker_name}: {seg.text}"
            )
        )
        self.timeline.scrollToBottom()
        # 🌟 새로 추가: 화자=페르소나 자동 전환
        if getattr(self, "chat_panel", None):
            self.chat_panel.set_active_persona(seg.speaker_name)

    def on_diar_done(self, results):
        self.state.diar_segments = results
        existing = set(self._combo_items(self.cmb_map_id))
        for _, _, spk in results:
            if spk not in existing:
                self.cmb_map_id.addItem(spk)
                existing.add(spk)
        self.on_status(f"Diarization updated ({len(results)} segments).")

    def _combo_items(self, combo: QComboBox) -> list[str]:
        return [combo.itemText(i) for i in range(combo.count())]

    def _refresh_preview(self):
        self.txt_preview.setPlainText(simple_summarize(self.state.live_segments))
