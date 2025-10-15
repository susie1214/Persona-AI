# -*- coding: utf-8 -*-
# ui/meeting_console.py
import os, datetime, json, time
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
    QCalendarWidget,
    QDateEdit,
)

from ui.survey_wizard import PersonaSurveyWizard
from ui.chat_dock import ChatDock
from ui.meeting_notes import MeetingNotesView
from ui.meeting_settings import MeetingSettingsWidget

from core.audio import AudioWorker, Segment, MeetingState, fmt_time, now_str
from core.diarization import DiarizationWorker
# ✅ 요약/액션/HTML/안건 추출 유틸 불러오기
from core.summarizer import (
    render_summary_html_from_segments,
    actions_from_segments,
    render_actions_table_html,
    extract_agenda,
    llm_summarize,
)
from core.rag_store import RagStore
from core.adapter import AdapterManager
from core.speaker import SpeakerManager
import numpy as np

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
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_name(self) -> str:
        return self.edit_name.text().strip()


class EnrollSpeakerDialog(QDialog):
    def __init__(self, unnamed_speakers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("새 화자 등록")
        layout = QFormLayout(self)

        self.cmb_speaker_id = QComboBox()
        self.cmb_speaker_id.addItems(unnamed_speakers)
        layout.addRow("등록할 화자 ID:", self.cmb_speaker_id)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("화자 이름 입력")
        layout.addRow("이름:", self.edit_name)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_data(self):
        return self.cmb_speaker_id.currentText(), self.edit_name.text().strip()


# ---------------- Main window ----------------
class MeetingConsole(QMainWindow):
    sig_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 보조 서비스")
        self.resize(1280, 860)

        # state / workers
        self.state = MeetingState()

        # SpeakerManager를 먼저 생성 (모든 컴포넌트가 공유)
        self.speaker_manager = SpeakerManager()

        # AudioWorker에 SpeakerManager 전달
        self.audio_worker = AudioWorker(self.state, speaker_manager=self.speaker_manager)
        self.audio_worker.sig_transcript.connect(self.on_segment)
        self.audio_worker.sig_status.connect(self.on_status)
        self.audio_worker.sig_new_speaker_detected.connect(self.on_new_speaker_auto_assigned)

        # DiarizationWorker도 같은 SpeakerManager 공유
        self.diar_worker = DiarizationWorker(self.state, speaker_manager=self.speaker_manager)
        self.diar_worker.sig_status.connect(self.on_status)
        self.diar_worker.sig_diar_done.connect(self.on_diar_done)
        self.diar_worker.sig_new_speaker.connect(self.on_new_speaker)

        self.adapter = AdapterManager()
        self.unnamed_speakers = {}

        # 녹음 상태
        self.recording = False
        self.recording_start_time = None

        # tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_live_tab()
        
        # 회의록 탭 (업로드 → 요약/회의록 저장/복사)
        # self.meeting_notes = MeetingNotesView(self)
        # self.tabs.addTab(self.meeting_notes, "Minutes")
        self._build_minutes_tab()
        self._build_action_tab()
        self._build_settings_tab()
        self._apply_theme()
        
        # RAG Store 초기화 (영구 저장소 경로 지정)
        os.makedirs("data/qdrant_db", exist_ok=True)
        self.rag = RagStore(persist_path="data/qdrant_db")
        if self.rag.ok:
            self.on_status("✓ RAG Store 초기화 완료 (data/qdrant_db)")
        else:
            self.on_status("⚠ RAG Store 사용 불가 - qdrant-client 또는 sentence-transformers 미설치")

        # 우측 개인 챗봇 도크
        self.chat_dock = QDockWidget("Persona Chatbot", self)
        self.chat_panel = ChatDock(rag_store = self.rag)
        self.chat_dock.setWidget(self.chat_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chat_dock)

        # live 미리보기 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(1000)

        # 설문 마법사(최초 1회)
        # self.survey = PersonaSurveyWizard(parent=self)
        # self.survey.show()
        

    # ---------------- UI builders ----------------
    def _build_minutes_tab(self):
        """Minutes 탭: 회의 전체요약 + Action Items를 이 탭에서 표시"""
        # 통합된 회의록 뷰 사용
        self.meeting_notes = MeetingNotesView(self)
        self.tabs.addTab(self.meeting_notes, "Minutes")

    def _build_live_tab(self):
        self.live_root = QWidget()
        L = QVBoxLayout(self.live_root)

        # top bar
        bar = QHBoxLayout()
        self.btn_start = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop Recording")
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
        self.chk_diar = QCheckBox("Auto Diarization (pyannote)")
        self.chk_diar.setChecked(self.state.diarization_enabled)
        mid.addWidget(self.chk_diar)
        L.addLayout(mid)

        # split
        splitter = QSplitter(Qt.Orientation.Horizontal)
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

        # 녹음 상태 표시
        self.lbl_record_status = QLabel("녹음 중지됨")
        self.lbl_record_status.setStyleSheet("color: gray; font-weight: bold;")
        Rv.addWidget(self.lbl_record_status)

        splitter.addWidget(right)
        splitter.setSizes([900, 380])
        L.addWidget(splitter)

        self.tabs.addTab(self.live_root, "Live")

        # events
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_sum.clicked.connect(self.on_summarize)
        self.btn_add2rag.clicked.connect(self.on_index_to_rag)

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
        H = QHBoxLayout(root)

        # ================== LEFT: Calendar + Form ==================
        left = QWidget()
        L = QVBoxLayout(left)

        # Calendar header: Year/Month selectors (민트/화이트/옐로 테마)
        header = QHBoxLayout()
        self.cmb_year = QComboBox()
        self.cmb_month = QComboBox()
        y0 = datetime.datetime.now().year
        for y in range(y0 - 2, y0 + 4):
            self.cmb_year.addItem(str(y))
        for m in range(1, 13):
            self.cmb_month.addItem(f"{m:02d}")

        header.addWidget(QLabel("Year"))
        header.addWidget(self.cmb_year)
        header.addSpacing(8)
        header.addWidget(QLabel("Month"))
        header.addWidget(self.cmb_month)
        header.addStretch(1)
        L.addLayout(header)

        # Big calendar
        from PySide6.QtWidgets import QCalendarWidget
        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        # mint-ish style hints
        self.calendar.setStyleSheet(f"""
            QCalendarWidget QToolButton {{
                background-color: {THEME['btn']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 6px;
                padding: 4px 8px;
                font-weight: 600;
            }}
            QCalendarWidget QToolButton:hover {{ background-color: {THEME['btn_hover']}; }}
            QCalendarWidget QWidget {{ alternate-background-color: {THEME['light_bg']}; }}
            QCalendarWidget QAbstractItemView:enabled {{
                color: #2f6;
                selection-background-color: {THEME['pane']};
                selection-color: #000;
            }}
        """)
        L.addWidget(self.calendar, stretch=1)

        # Form: 회의/프로젝트/장소 등
        form = QFormLayout()
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("회의 주제 / 프로젝트명")
        form.addRow("제목", self.edit_title)

        self.edit_location = QLineEdit()
        self.edit_location.setPlaceholderText("장소(선택)")
        form.addRow("장소", self.edit_location)

        # 회의 시작/종료 (달력 날짜와 동기화되는 시간)
        self.dt_start = QDateTimeEdit()
        self.dt_start.setCalendarPopup(True)
        self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm")

        self.dt_end = QDateTimeEdit()
        self.dt_end.setCalendarPopup(True)
        self.dt_end.setDisplayFormat("yyyy-MM-dd HH:mm")

        today = QDateTime.currentDateTime()
        self.dt_start.setDateTime(today.addDays(7))
        self.dt_end.setDateTime(today.addDays(7).addSecs(3600))

        form.addRow("회의 시작", self.dt_start)
        form.addRow("회의 종료", self.dt_end)

        # 프로젝트 시작/마감, 결제일
        self.d_project_start = QDateEdit()
        self.d_project_start.setCalendarPopup(True)
        self.d_project_start.setDisplayFormat("yyyy-MM-dd")
        self.d_project_start.setDate(self.dt_start.date())

        self.d_project_due = QDateEdit()
        self.d_project_due.setCalendarPopup(True)
        self.d_project_due.setDisplayFormat("yyyy-MM-dd")
        self.d_project_due.setDate(self.dt_start.date().addDays(30))

        self.d_payment_due = QDateEdit()
        self.d_payment_due.setCalendarPopup(True)
        self.d_payment_due.setDisplayFormat("yyyy-MM-dd")
        self.d_payment_due.setDate(self.dt_start.date().addDays(14))

        form.addRow("프로젝트 시작", self.d_project_start)
        form.addRow("프로젝트 마감", self.d_project_due)
        form.addRow("결제일", self.d_payment_due)

        L.addLayout(form)

        H.addWidget(left, stretch=3)

        # ================== RIGHT: Schedule Memo + To-do ==================
        right = QWidget()
        R = QVBoxLayout(right)

        # Schedule memo
        R.addWidget(QLabel("Schedule Memo"))
        self.txt_sched = QTextEdit()
        self.txt_sched.setPlaceholderText("자동 생성되며, 직접 수정도 가능해요.")
        R.addWidget(self.txt_sched, stretch=1)

        # To-do list (간단 추가/삭제)
        todo_row = QHBoxLayout()
        todo_row.addWidget(QLabel("To-do"))
        self.edit_todo = QLineEdit()
        self.edit_todo.setPlaceholderText("할 일을 입력하고 +를 누르세요")
        self.btn_todo_add = QPushButton("+")
        self.btn_todo_del = QPushButton("−")
        todo_row.addWidget(self.edit_todo, stretch=1)
        todo_row.addWidget(self.btn_todo_add)
        todo_row.addWidget(self.btn_todo_del)
        R.addLayout(todo_row)

        self.list_todo = QListWidget()
        R.addWidget(self.list_todo, stretch=1)

        # Generate button
        gen = QHBoxLayout()
        self.btn_sched_memo = QPushButton("Make Schedule Memo")
        gen.addStretch(1)
        gen.addWidget(self.btn_sched_memo)
        R.addLayout(gen)

        H.addWidget(right, stretch=2)

        self.tabs.addTab(root, "Schedule")

        # ---------- signals ----------
        # 연/월 콤보 → 달력 페이지 변경
        self.cmb_year.currentTextChanged.connect(self._on_year_month_changed)
        self.cmb_month.currentTextChanged.connect(self._on_year_month_changed)

        # 달력 날짜 선택 → 시작/종료 날짜 부분만 해당 날짜로 갱신
        self.calendar.selectionChanged.connect(self._on_calendar_selected)

        # 시간/제목/장소 바뀌면 미리보기 즉시 갱신
        self.dt_start.dateTimeChanged.connect(self._refresh_schedule_preview)
        self.dt_end.dateTimeChanged.connect(self._refresh_schedule_preview)
        self.edit_title.textChanged.connect(self._refresh_schedule_preview)
        self.edit_location.textChanged.connect(self._refresh_schedule_preview)
        self.d_project_start.dateChanged.connect(self._refresh_schedule_preview)
        self.d_project_due.dateChanged.connect(self._refresh_schedule_preview)
        self.d_payment_due.dateChanged.connect(self._refresh_schedule_preview)

        # todo
        self.btn_todo_add.clicked.connect(self._on_todo_add)
        self.btn_todo_del.clicked.connect(self._on_todo_del)

        # 메모 생성
        self.btn_sched_memo.clicked.connect(self.on_make_schedule)

        # 초기 달력/콤보 동기화
        d = self.dt_start.date()
        self.calendar.setSelectedDate(d)
        self.cmb_year.setCurrentText(str(d.year()))
        self.cmb_month.setCurrentText(f"{d.month():02d}")

        # 초기 미리보기
        self._refresh_schedule_preview()

    def _on_year_month_changed(self):
        """연/월 콤보 변경 → 달력 페이지 이동"""
        try:
            y = int(self.cmb_year.currentText())
            m = int(self.cmb_month.currentText())
            self.calendar.setCurrentPage(y, m)
        except Exception:
            pass

    def _on_calendar_selected(self):
        """달력에서 날짜 선택 → 시작/종료 날짜의 '날짜'만 바꾸고 시간은 유지"""
        d = self.calendar.selectedDate()
        start = self.dt_start.dateTime()
        end = self.dt_end.dateTime()
        self.dt_start.setDateTime(QDateTime(d, start.time()))
        self.dt_end.setDateTime(QDateTime(d, end.time()))
        # 프로젝트 시작 기본값도 동기
        if not self.edit_title.text().strip():
            self.d_project_start.setDate(d)
        self._refresh_schedule_preview()

    def _on_todo_add(self):
        txt = self.edit_todo.text().strip()
        if not txt:
            return
        self.list_todo.addItem(txt)
        self.edit_todo.clear()
        self._refresh_schedule_preview()

    def _on_todo_del(self):
        for it in self.list_todo.selectedItems():
            self.list_todo.takeItem(self.list_todo.row(it))
        self._refresh_schedule_preview()

    def _refresh_schedule_preview(self):
        """우측 Schedule Memo 영역 자동 갱신(읽기/쓰기 가능하므로 기본 템플릿만 갱신)"""
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("yyyy-MM-dd HH:mm")
        title = self.edit_title.text().strip() or "(제목 미정)"
        loc = self.edit_location.text().strip()
        pj_s = self.d_project_start.date().toString("yyyy-MM-dd")
        pj_d = self.d_project_due.date().toString("yyyy-MM-dd")
        pay = self.d_payment_due.date().toString("yyyy-MM-dd")

        todos = [self.list_todo.item(i).text() for i in range(self.list_todo.count())]
        todo_block = "\n".join([f"• {t}" for t in todos]) if todos else "• (등록된 To-do 없음)"

        memo = (
            f"[일정]\n"
            f"- 회의: {s} ~ {e}\n"
            f"- 제목: {title}\n"
            f"- 장소: {loc or '-'}\n\n"
            f"[프로젝트]\n"
            f"- 시작: {pj_s}\n"
            f"- 마감: {pj_d}\n"
            f"- 결제일: {pay}\n\n"
            f"[To-do]\n{todo_block}\n"
        )
        # 사용자가 수동 편집했더라도 기본 베이스를 항상 다시 깔아주고 싶다면 setPlainText,
        # 수동 편집을 보존하고 싶다면 현재 텍스트가 비어 있을 때만 세팅하세요.
        self.txt_sched.setPlainText(memo)


    # def _build_action_tab(self):
    #     root = QWidget()
    #     L = QVBoxLayout(root)

    #     L.addWidget(QLabel("회의 전체요약"))
    #     self.txt_summary = QTextEdit()
    #     L.addWidget(self.txt_summary)

    #     L.addWidget(QLabel("Action Items"))
    #     self.txt_actions = QTextEdit()
    #     L.addWidget(self.txt_actions)

    #     row = QHBoxLayout()
    #     row.addWidget(QLabel("다음 회의 시작"))
    #     self.dt_start = QDateTimeEdit()
    #     self.dt_start.setCalendarPopup(True)
    #     self.dt_start.setKeyboardTracking(True)
    #     self.dt_start.setDateTime(QDateTime.currentDateTime().addDays(7))
    #     self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm")
    #     row.addWidget(self.dt_start)

    #     row.addWidget(QLabel("종료"))
    #     self.dt_end = QDateTimeEdit()
    #     self.dt_end.setCalendarPopup(True)
    #     self.dt_end.setKeyboardTracking(True)
    #     self.dt_end.setDateTime(QDateTime.currentDateTime().addDays(7).addSecs(3600))
    #     self.dt_end.setDisplayFormat("yyyy-MM-dd HH:mm")
    #     row.addWidget(self.dt_end)

    #     self.btn_sched_memo = QPushButton("Make Schedule Memo")
    #     row.addWidget(self.btn_sched_memo)
    #     L.addLayout(row)

    #     L.addWidget(QLabel("다음 회의 메모"))
    #     self.txt_sched = QTextEdit()
    #     L.addWidget(self.txt_sched)

    #     self.tabs.addTab(root, "Schedule")
    #     self.btn_sched_memo.clicked.connect(self.on_make_schedule)

    def _build_settings_tab(self):
        # 새로운 통합 설정 위젯 생성
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        # 기존 오디오/시스템 설정
        system_group = QWidget()
        F = QFormLayout(system_group)

        self.cmb_asr = QComboBox()
        for m in ["small", "medium", "large-v3"]: # "base", 
            self.cmb_asr.addItem(m)
        self.cmb_asr.setCurrentText(DEFAULT_MODEL)

        self.chk_gpu = QCheckBox("Use GPU if available")
        self.chk_gpu.setChecked(True)

        self.chk_diar2 = QCheckBox("Auto Diarization")
        self.chk_diar2.setChecked(False)

        self.edit_hf = QLineEdit()
        self.edit_hf.setPlaceholderText(f"{HF_TOKEN_ENV} (HuggingFace token)")
        # .env 파일에서 로드된 토큰이 있으면 표시
        existing_token = os.getenv(HF_TOKEN_ENV, "")
        if existing_token:
            self.edit_hf.setText(f"{existing_token}")
            self.edit_hf.setEchoMode(QLineEdit.EchoMode.Password)
            self.on_status(f"✓ .env에서 HF_TOKEN 로드됨: {existing_token[:10]}...")

        self.btn_add_participant = QPushButton("참가자 추가")
        self.btn_save_speakers = QPushButton("화자 정보 저장")
        self.btn_load_speakers = QPushButton("화자 정보 로드")
        self.btn_clear_db = QPushButton("Vector DB 초기화")

        F.addRow("Whisper Model", self.cmb_asr)
        F.addRow("", self.chk_gpu)
        F.addRow("Auto Diarization", self.chk_diar2)
        F.addRow("HF Token", self.edit_hf)
        F.addRow("", self.btn_add_participant)

        # 화자 관리 버튼들을 가로로 배치
        speaker_buttons = QHBoxLayout()
        speaker_buttons.addWidget(self.btn_save_speakers)
        speaker_buttons.addWidget(self.btn_load_speakers)
        F.addRow("화자 관리:", speaker_buttons)

        # DB 관리 버튼
        db_buttons = QHBoxLayout()
        db_buttons.addWidget(self.btn_clear_db)
        F.addRow("DB 관리:", db_buttons)

        layout.addWidget(QLabel("🔧 시스템 설정"))
        layout.addWidget(system_group)

        # 회의 설정 및 화자 매핑 위젯 (speaker_manager 공유)
        self.meeting_settings = MeetingSettingsWidget(speaker_manager=self.speaker_manager)
        self.meeting_settings.speaker_mapping_changed.connect(self.on_speaker_mapping_changed)
        layout.addWidget(self.meeting_settings)

        self.tabs.addTab(main_widget, "Settings")

        self.btn_add_participant.clicked.connect(self.on_add_participant)
        self.btn_save_speakers.clicked.connect(self.save_speaker_mapping)
        self.btn_load_speakers.clicked.connect(self.load_speaker_mapping)
        self.btn_clear_db.clicked.connect(self.on_clear_vector_db)
        self.chk_diar2.stateChanged.connect(self.on_diar_toggle_settings)

    def on_clear_vector_db(self):
        """Vector DB를 초기화"""
        reply = QMessageBox.question(
            self,
            "Vector DB 초기화",
            "정말로 Vector DB의 모든 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.rag and self.rag.ok:
                if self.rag.clear_collection():
                    self.on_status("✓ Vector DB가 성공적으로 초기화되었습니다.")
                    QMessageBox.information(self, "완료", "Vector DB가 초기화되었습니다.")
                else:
                    self.on_status("⚠ Vector DB 초기화에 실패했습니다.")
                    QMessageBox.warning(self, "오류", "Vector DB 초기화 중 오류가 발생했습니다.")
            else:
                self.on_status("⚠ RAG Store가 초기화되지 않아 DB를 초기화할 수 없습니다.")
                QMessageBox.warning(self, "오류", "RAG Store가 유효하지 않습니다.")

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
            self.on_status("화자 분리(Diarization) 활성화 - 대화 겹침 자동 감지")

        # 녹음 자동 시작
        os.makedirs("output/recordings", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y년%m월%d일_%H시%M분")
        recording_path = f"output/recordings/meeting_{timestamp}.wav"

        self.audio_worker.start_recording(recording_path)
        self.recording = True
        self.recording_start_time = time.time()

        # UI 업데이트
        self.lbl_record_status.setText(f"🔴 녹음 중: {recording_path.split("/")[-1]}")
        self.lbl_record_status.setStyleSheet("color: red; font-weight: bold;")

        self.on_status(f"Started. 녹음 시작: {recording_path}")

    def on_stop(self):
        # 녹음 중지 및 파일 저장
        saved_path = None
        if self.recording:
            saved_path = self.audio_worker.stop_recording()
            self.recording = False

            # 녹음 시간 계산
            if self.recording_start_time:
                duration = time.time() - self.recording_start_time
                duration_str = fmt_time(duration)
            else:
                duration_str = "00:00"

            # UI 업데이트
            self.lbl_record_status.setText(f"녹음 완료 (시간: {duration_str})")
            self.lbl_record_status.setStyleSheet("color: green; font-weight: bold;")

        # 오디오 캡처 중지
        try:
            self.audio_worker.stop()
            self.diar_worker.stop()
        except Exception:
            pass

        # 녹음 결과 메시지
        if saved_path:
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            duration_str = fmt_time(duration)
            self.on_status(f"Stopped. 녹음 저장 완료: {saved_path} (시간: {duration_str})")
            QMessageBox.information(self, "녹음 완료",
                f"녹음이 저장되었습니다.\n\n파일: {saved_path}\n시간: {duration_str}")
        else:
            self.on_status("Stopped.")

    def on_summarize(self):
        # 1) LLM을 이용한 AI 요약 생성
        summary_text = llm_summarize(self.state.live_segments)
        
        print(f"[DEBUG - meeting_console] summary_text : {summary_text}")
        
        self.state.summary = summary_text  # state에 텍스트 요약 저장

        # 2) Action Items 추출
        items = actions_from_segments(self.state.live_segments)
        self.state.actions = items
        actions_html = render_actions_table_html(items)

        # 3) Transcript 텍스트 생성
        transcript_lines = []
        for seg in self.state.live_segments:
            transcript_lines.append(f"[{seg.speaker_name}] {seg.text}")
        transcript_text = "\n".join(transcript_lines)

        # 4) 표시용 HTML 생성 (요약 + 액션 아이템)
        # QTextEdit은 기본적인 마크다운(줄바꿈)을 지원하므로 pre 태그로 감싸기
        summary_html = f"<pre>{summary_text}</pre>"
        html_for_display = summary_html + actions_html

        # 5) meeting_notes 뷰에 업데이트
        self.meeting_notes.update_notes(html_for_display, transcript_text)

        # 6) 🎯 AI 요약문을 RAG에 저장
        self._save_summary_to_rag(summary_text, items)

        QMessageBox.information(self, "Done", "AI 요약 및 액션 아이템 생성 완료\n요약 문서가 RAG에 저장되었습니다.")

    def _save_summary_to_rag(self, summary_text: str, action_items: list):
        """
        요약 문서를 RAG에 저장 (원본 대화는 저장하지 않음)

        Args:
            summary_text: 텍스트 형식의 요약 문서
            action_items: 액션 아이템 리스트
        """
        if not self.rag.ok:
            self.on_status("⚠ RAG Store 사용 불가 - 요약 문서 저장 생략")
            QMessageBox.warning(self, "RAG 저장 실패", 
                              "RAG Store가 초기화되지 않았습니다.\n" 
                              "콘솔 로그에서 Qdrant 또는 SentenceTransformer 관련 오류를 확인하세요.")
            return

        # 회의 메타데이터
        meeting_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        participants = sorted(set(
            seg.speaker_name if seg.speaker_name != "Unknown" else "speaker_00"
            for seg in self.state.live_segments
            if getattr(seg, "text", "").strip()
        ))

        # 요약 문서를 하나의 세그먼트로 저장
        summary_segment = {
            "speaker_id": "SYSTEM",
            "speaker_name": "회의 요약",
            "text": f"[{meeting_date}] 회의 요약 - 참석자: {', '.join(participants)}\n\n{summary_text}",
            "start": 0.0,
            "end": 0.0,
        }

        # 각 액션 아이템을 개별 세그먼트로 저장
        action_segments = []
        for item in action_items:
            action_text = f"[액션아이템] {item.get('title', '')} (담당: {item.get('owner', '')}, 기한: {item.get('due', '')})"
            action_segments.append({
                "speaker_id": item.get('owner', 'SYSTEM'),
                "speaker_name": item.get('owner', '미지정'),
                "text": action_text,
                "start": 0.0,
                "end": 0.0,
            })

        # RAG 저장 전 내용 출력
        print("-" * 50)
        print("[DEBUG] Documents being sent to RAG store:")
        print("--- 1. Summary Document ---")
        print(summary_segment['text'])
        print("--- 2. Action Item Documents ---")
        for i, act_seg in enumerate(action_segments, 1):
            print(f"{i}. {act_seg['text']}")
        print("-" * 50)

        # RAG에 저장
        try:
            self.rag.upsert_segments([summary_segment] + action_segments)
            self.on_status(f"✓ 요약 문서 RAG 저장 완료: 요약 1개 + 액션아이템 {len(action_segments)}개")
        except Exception as e:
            self.on_status(f"⚠ 요약 문서 RAG 저장 실패: {e}")

    def on_index_to_rag(self):
        """
        ⚠️ 주의: 이 기능은 개발/테스트 용도입니다.
        실제 운영에서는 요약 생성 시 자동으로 RAG에 저장됩니다.
        원본 대화는 QLoRA 학습에 사용되며 RAG에 저장되지 않습니다.
        """
        if not self.rag.ok:
            QMessageBox.warning(self, "RAG", "Qdrant 사용 불가(미설치/연결 실패).")
            return

        # 경고 메시지
        reply = QMessageBox.question(
            self,
            "RAG 인덱싱",
            "⚠️ 이 기능은 테스트 용도입니다.\n\n"
            "실제 운영에서는:\n"
            "• 요약 문서만 RAG에 저장됩니다 (Summarize 버튼 클릭 시 자동)\n"
            "• 원본 대화는 QLoRA 학습 데이터로 사용됩니다\n\n"
            "테스트를 위해 최근 대화를 RAG에 저장하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # 최근 50줄만 인덱싱 (테스트용)
        self.rag.upsert_segments(self.state.live_segments[-50:])
        QMessageBox.information(self, "RAG", "테스트용으로 최근 발언을 RAG에 저장했습니다.")

    def on_diar_toggle(self):
        self.state.diarization_enabled = self.chk_diar.isChecked()
        self.chk_diar2.setChecked(self.state.diarization_enabled)

    def on_diar_toggle_settings(self):
        self.state.diarization_enabled = self.chk_diar2.isChecked()
        self.chk_diar.setChecked(self.state.diarization_enabled)

    def on_add_participant(self):
        if not self.unnamed_speakers:
            QMessageBox.information(self, "화자 등록", "새로 감지된 화자가 없습니다.")
            return

        dlg = EnrollSpeakerDialog(list(self.unnamed_speakers.keys()), self)
        if dlg.exec():
            speaker_id, name = dlg.get_data()
            if not name:
                QMessageBox.warning(self, "오류", "이름을 입력해야 합니다.")
                return

            embeddings = self.unnamed_speakers.pop(speaker_id, [])
            if not embeddings:
                return

            for emb in embeddings:
                self.speaker_manager.add_speaker_embedding(name, emb)
            
            self.state.speaker_map[speaker_id] = name
            self.on_status(f"New speaker enrolled: {speaker_id} -> {name}")
            QMessageBox.information(self, "등록 완료", f"{name} 님의 목소리를 등록했습니다.")

    def on_make_schedule(self):
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("HH:mm")
        title = self.edit_title.text().strip()
        loc = self.edit_location.text().strip()

        # 1) 자동 안건 추출
        agenda_list = extract_agenda(self.state.live_segments, max_items=5)
        agenda_line = " · ".join(agenda_list) if agenda_list else "-"

        # 2) 기한 있는 Action Item 정리(있으면 덧붙임)
        lines = []
        for ai in (self.state.actions or []):
            due = ai.get("due")
            if due:
                owner = ai.get("owner", "")
                t = ai.get("title", "")
                lines.append(f"[{due}] {t} — {owner}")
        ai_block = ("\n" + "\n".join(lines)) if lines else ""

        pj_s = self.d_project_start.date().toString("yyyy-MM-dd")
        pj_d = self.d_project_due.date().toString("yyyy-MM-dd")
        pay  = self.d_payment_due.date().toString("yyyy-MM-dd")

        participants = ', '.join(sorted(set(
            seg.speaker_name for seg in self.state.live_segments if seg.speaker_name != "Unknown"
        ))) or "-"

        memo = (
            f"회의: {s} ~ {e}\n"
            f"제목: {title}\n"
            f"장소: {loc or '-'}\n"
            f"참석자: {participants}\n"
            f"안건: {agenda_line}{ai_block}\n\n"
            f"[프로젝트]\n"
            f"- 시작: {pj_s}\n"
            f"- 마감: {pj_d}\n"
            f"- 결제일: {pay}\n"
        )

        self.state.schedule_note = memo
        self.txt_sched.setPlainText(memo)
        QMessageBox.information(self, "메모 생성", "스케줄 메모를 갱신했습니다.")

    def save_speaker_mapping(self):
        """화자 매핑 정보를 JSON 파일로 저장"""
        try:
            speaker_data = {
                "speaker_map": self.state.speaker_map,
                "speaker_counter": self.state.speaker_counter,
                "timestamp": datetime.datetime.now().isoformat()
            }

            with open("speaker_mapping.json", "w", encoding="utf-8") as f:
                json.dump(speaker_data, f, ensure_ascii=False, indent=2)

            self.on_status("화자 매핑 정보 저장 완료: speaker_mapping.json")
            QMessageBox.information(self, "저장 완료", "화자 매핑 정보가 저장되었습니다.")
        except Exception as e:
            self.on_status(f"화자 매핑 저장 실패: {e}")
            QMessageBox.warning(self, "저장 실패", f"화자 매핑 저장 중 오류가 발생했습니다:\n{e}")

    def load_speaker_mapping(self):
        """JSON 파일에서 화자 매핑 정보를 로드"""
        try:
            if not os.path.exists("speaker_mapping.json"):
                QMessageBox.information(self, "파일 없음", "저장된 화자 매핑 파일이 없습니다.")
                return

            with open("speaker_mapping.json", "r", encoding="utf-8") as f:
                speaker_data = json.load(f)

            self.state.speaker_map = speaker_data.get("speaker_map", {})
            self.state.speaker_counter = speaker_data.get("speaker_counter", 0)

            self.on_status(f"화자 매핑 정보 로드 완료: {len(self.state.speaker_map)}개 화자")
            QMessageBox.information(self, "로드 완료",
                f"화자 매핑 정보가 로드되었습니다.\n화자 수: {len(self.state.speaker_map)}개")

            # 설정 탭의 화자 매핑 테이블 새로고침
            if hasattr(self, 'meeting_settings') and hasattr(self.meeting_settings, 'refresh_speaker_mapping'):
                self.meeting_settings.refresh_speaker_mapping()

        except Exception as e:
            self.on_status(f"화자 매핑 로드 실패: {e}")
            QMessageBox.warning(self, "로드 실패", f"화자 매핑 로드 중 오류가 발생했습니다:\n{e}")

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

    def on_load_audio_file(self):
        """오디오 파일을 불러와서 전사 및 요약 표시"""
        from PySide6.QtWidgets import QFileDialog, QProgressDialog
        from PySide6.QtCore import QThread, QObject, Signal

        # 파일 선택
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "오디오 파일 선택",
            "",
            "Audio/Video Files (*.wav *.mp3 *.m4a *.mp4 *.aac *.flac);;All Files (*)"
        )

        if not file_path:
            return

        # 프로그레스 다이얼로그
        progress = QProgressDialog("파일 처리 중...", "취소", 0, 0, self)
        progress.setWindowTitle("오디오 파일 처리")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        # Worker 스레드로 처리
        from core.offline_meeting import process_audio_file

        class FileProcessWorker(QObject):
            finished = Signal(dict)
            error = Signal(str)

            def __init__(self, path):
                super().__init__()
                self.path = path

            def run(self):
                try:
                    result = process_audio_file(
                        self.path,
                        asr_model="medium",
                        use_gpu=True,
                        diarize=True,
                        use_llm_summary=True
                    )
                    self.finished.emit(result)
                except Exception as e:
                    self.error.emit(str(e))

        thread = QThread()
        worker = FileProcessWorker(file_path)
        worker.moveToThread(thread)

        def on_finished(result):
            progress.close()
            self._display_file_result(result)
            thread.quit()
            thread.deleteLater()

        def on_error(msg):
            progress.close()
            QMessageBox.critical(self, "오류", f"파일 처리 중 오류 발생:\n{msg}")
            thread.quit()
            thread.deleteLater()

        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        thread.start()

    def _display_file_result(self, result: dict):
        """
        파일 처리 결과를 UI에 표시

        Args:
            result: process_audio_file()의 반환값
                - segments: 전사 세그먼트 리스트
                - markdown: 마크다운 회의록
                - summary: AI 요약 텍스트
                - actions: 액션 아이템 리스트
        """
        segments = result.get("segments", [])
        transcript_text = result.get("markdown", "")
        summary_text = result.get("summary", "") # AI 요약

        # 1. Segment 객체로 변환하여 state에 저장 (QLoRA 학습용)
        from core.audio import Segment
        self.state.live_segments = []
        for seg in segments:
            self.state.live_segments.append(Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
                speaker_id=seg.get("speaker", "Unknown"),
                speaker_name=seg.get("speaker", "Unknown")
            ))

        # 2. 액션 아이템 추출 및 HTML 생성
        action_items = actions_from_segments(self.state.live_segments)
        actions_html = render_actions_table_html(action_items) if action_items else "<p>액션 없음</p>"

        # 3. 표시용 HTML 생성 (요약 + 액션 아이템)
        summary_html = f"<pre>{summary_text}</pre>"
        html_for_display = summary_html + actions_html

        # 4. meeting_notes 뷰에 업데이트
        self.meeting_notes.update_notes(html_for_display, transcript_text)

        # 5. AI 요약 문서를 RAG에 저장
        self._save_summary_to_rag(summary_text, action_items)

        # 상태 메시지
        self.on_status(f"✓ 파일 처리 완료: {len(segments)}개 세그먼트, {len(action_items)}개 액션아이템")

        # Minutes 탭으로 전환
        self.tabs.setCurrentWidget(self.meeting_notes)

        QMessageBox.information(
            self,
            "파일 처리 완료",
            f"전사 완료: {len(segments)}개 발언\n"
            f"액션 아이템: {len(action_items) if action_items else 0}개\n\n"
            f"AI 요약 및 전사 내용이 RAG에 저장되었습니다."
        )

    # ---------------- Signals ----------------
    def on_status(self, msg: str):
        self.txt_status.appendPlainText(f"{now_str()}  {msg}")

    def on_segment(self, seg: Segment):
        # seg가 dict인 경우 Segment 객체로 변환
        if isinstance(seg, dict):
            seg = Segment(
                start=seg.get('start', 0.0),
                end=seg.get('end', 0.0),
                text=seg.get('text', ''),
                speaker_id=seg.get('speaker_id', 'Unknown'),
                speaker_name=seg.get('speaker_name', 'Unknown')
            )

        # live_segments가 dict로 잘못 변경된 경우 복구
        if isinstance(self.state.live_segments, dict):
            self.on_status("ERROR: live_segments가 dict로 변경됨. list로 복구합니다.")
            self.state.live_segments = []

        self.state.live_segments.append(seg)
        self.list_chat.addItem(QListWidgetItem(f"[{seg.speaker_name}] {seg.text}"))
        self.list_chat.scrollToBottom()

    def on_diar_done(self, results):
        """화자 분리 결과 처리 (새로운 speaker_xx 형태 ID로 처리)"""
        self.state.diar_segments = results

        for start, end, speaker_id, confidence in results:
            # speaker_map에 speaker_id -> display_name 매핑 업데이트
            display_name = self.diar_worker.get_speaker_manager().get_speaker_display_name(speaker_id)
            if speaker_id not in self.state.speaker_map:
                self.state.speaker_map[speaker_id] = display_name

        self.on_status(f"화자 분리 완료: {len(results)}개 구간 처리")

    def on_new_speaker(self, speaker_id: str, display_name: str):
        """새로운 화자 감지 시 처리"""
        self.state.speaker_map[speaker_id] = display_name
        self.on_status(f"새로운 화자 감지: {speaker_id} ({display_name})")

        # 설정 탭의 화자 매핑 테이블 새로고침
        if hasattr(self, 'meeting_settings') and hasattr(self.meeting_settings, 'refresh_speaker_mapping'):
            self.meeting_settings.refresh_speaker_mapping()

    def on_new_speaker_auto_assigned(self, speaker_name: str):
        """새로운 화자가 자동으로 할당되었을 때 처리"""
        self.on_status(f"새 화자 자동 할당: {speaker_name}")

        # SpeakerManager의 speakers가 dict인 경우 복구
        if isinstance(self.speaker_manager.speakers, dict):
            self.on_status("ERROR: speaker_manager.speakers가 dict로 변경됨. list로 복구합니다.")
            self.speaker_manager.speakers = list(self.speaker_manager.speakers.values()) if self.speaker_manager.speakers else []

        # SpeakerManager에 화자 추가 (임베딩 없이 ID만 등록)
        if speaker_name not in self.speaker_manager.speaker_mapping:
            from core.speaker import Speaker
            new_speaker = Speaker(
                speaker_id=speaker_name,
                display_name=speaker_name,
                embeddings=[],
                confidence_scores=[]
            )
            self.speaker_manager.speakers.append(new_speaker)
            self.speaker_manager.speaker_mapping[speaker_name] = speaker_name

            # 다음 ID 업데이트 (speaker_XX 형태에서 숫자 추출)
            try:
                if speaker_name.startswith("speaker_"):
                    speaker_num = int(speaker_name.split("_")[1])
                    if speaker_num >= self.speaker_manager.next_speaker_id:
                        self.speaker_manager.next_speaker_id = speaker_num + 1
            except Exception:
                pass

            self.speaker_manager.save_speakers()
            self.speaker_manager.save_speaker_mapping()

        # 설정 탭의 화자 매핑 테이블 새로고침
        if hasattr(self, 'meeting_settings') and hasattr(self.meeting_settings, 'refresh_speaker_mapping'):
            self.meeting_settings.refresh_speaker_mapping()

    def on_speaker_mapping_changed(self, mapping: dict):
        """화자 매핑이 변경되었을 때 처리"""
        # state의 speaker_map 업데이트
        self.state.speaker_map.update(mapping)

        # 리셋된 경우 (빈 딕셔너리)
        if not mapping:
            self.state.speaker_map = {}
            self.state.speaker_counter = 0
            self.on_status("화자 매핑이 초기화되었습니다.")
        else:
            self.on_status(f"화자 매핑 업데이트: {len(mapping)}개")

    def _combo_items(self, combo: QComboBox) -> list[str]:
        return [combo.itemText(i) for i in range(combo.count())]

    def _refresh_preview(self):
        # 화자 정보가 포함된 미리보기 생성
        if not self.state.live_segments:
            self.txt_preview.setPlainText("실시간 대화 내용이 여기에 표시됩니다.")
            return

        # 최근 10개 발언만 표시하되 화자 정보 확실히 포함
        recent_segments = self.state.live_segments[-10:]
        preview_lines = []

        for seg in recent_segments:
            if getattr(seg, "text", "").strip():
                speaker_display = seg.speaker_name
                if speaker_display == "Unknown":
                    speaker_display = "speaker_00"
                preview_lines.append(f"[{speaker_display}] {seg.text}")

        if preview_lines:
            self.txt_preview.setPlainText("\n".join(preview_lines))
        else:
            self.txt_preview.setPlainText("대화 내용을 분석 중입니다...")
