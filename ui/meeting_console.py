# -*- coding: utf-8 -*-
# ui/meeting_console.py
import os, datetime, json, time
from PySide6.QtCore import Qt, QTimer, Signal, QDateTime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QPlainTextEdit, QLabel, QTabWidget, QSplitter, QComboBox,
    QCheckBox, QFormLayout, QLineEdit, QMessageBox, QDialog, QDialogButtonBox,
    QDateTimeEdit, QTextEdit, QDockWidget, QCalendarWidget, QDateEdit,
)

from ui.survey_wizard import PersonaSurveyWizard
from ui.chat_dock import ChatDock
from ui.meeting_notes import MeetingNotesView
from ui.meeting_settings import MeetingSettingsWidget
from ui.documents_tab_qt6 import DocumentsTab
from core.audio import AudioWorker, Segment, MeetingState, fmt_time, now_str
from core.diarization import DiarizationWorker
from core.summarizer import (
    render_summary_html_from_segments, actions_from_segments,
    render_actions_table_html, extract_agenda, llm_summarize,
)
from core.rag_store import RagStore
from core.adapter import AdapterManager
from core.speaker import SpeakerManager
import numpy as np
from core.schedule_store import Schedule as JSONSchedule, save_schedule as json_save, list_month as json_list_month, new_id as json_new_id


THEME = {
    "bg": "#e6f5e6", "pane": "#99cc99", "light_bg": "#fafffa",
    "btn": "#ffe066", "btn_hover": "#ffdb4d", "btn_border": "#cccc99",
}
HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_MODEL = "medium"


class MeetingConsole(QMainWindow):
    sig_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 보조 서비스")
        self.resize(1280, 860)

        self.state = MeetingState()
        self.speaker_manager = SpeakerManager()
        self.audio_worker = AudioWorker(self.state, speaker_manager=self.speaker_manager)
        self.diar_worker = DiarizationWorker(self.state, speaker_manager=self.speaker_manager)
        self.adapter = AdapterManager()
        self.recording = False
        self.recording_start_time = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_live_tab()
        self._build_minutes_tab()
        self._build_schedule_tab()
        self.documents_tab = DocumentsTab(self)
        self.tabs.addTab(self.documents_tab, "Documents")
        self._build_settings_tab()
        self._apply_theme()
        self._connect_signals()



        os.makedirs("data/qdrant_db", exist_ok=True)
        self.rag = RagStore(persist_path="data/qdrant_db")
        self.on_status("✓ RAG Store 초기화 완료" if self.rag.ok else "⚠ RAG Store 사용 불가")

        self.chat_dock = QDockWidget("Persona Chatbot", self)
        self.chat_panel = ChatDock(rag_store=self.rag)
        self.chat_dock.setWidget(self.chat_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chat_dock)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(1000)

        self._calendar_cache = {}  # {day: [items]}  로딩 캐시
        self._reload_calendar()    # 현재 연/월 일정 로드 및 표시

    def _current_schedule_payload(self) -> dict:
        # ISO 문자열로 변환
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm").replace(" ", "T") + ":00"
        e = self.dt_end.dateTime().toString("yyyy-MM-dd HH:mm").replace(" ", "T") + ":00"
        pj_s = self.d_project_start.date().toString("yyyy-MM-dd") or None
        pj_d = self.d_project_due.date().toString("yyyy-MM-dd") or None
        pay  = self.d_payment_due.date().toString("yyyy-MM-dd") or None
        todos = [self.list_todo.item(i).text() for i in range(self.list_todo.count())]

        return {
            "title": self.edit_title.text().strip(),
            "location": self.edit_location.text().strip() or None,
            "meeting_start": s,
            "meeting_end": e,
            "project_start": pj_s,
            "project_due": pj_d,
            "settlement_at": pay,
            "todos": todos,
        }

    def _save_schedule_json(self):
        data = self._current_schedule_payload()
        # 업서트 키: (title + meeting_start)
        # 새로 저장할 때마다 새로운 id 생성 (업서트 내부에서 기존건 갱신됨)
        row = JSONSchedule(
            id=json_new_id(),
            **data
        )
        json_save(row)  # ← 파일 schedules.json에 원자적으로 저장
        # 저장 후 현재 달 다시 로드
        self._reload_calendar()

    def _reload_calendar(self):
        try:
            y = int(self.cmb_year.currentText())
            m = int(self.cmb_month.currentText())
        except Exception:
            # 초기 진입 시 combobox가 아직 준비 안 되었을 수도 있음
            d = self.dt_start.date()
            y, m = d.year(), d.month()
        self._calendar_cache = json_list_month(y, m)  # {day : [items]}
        # 날짜별 툴팁 표시(디자인 안 바꾸고 가볍게 정보만)
        from PySide6.QtGui import QTextCharFormat
        fmt_default = QTextCharFormat()
        self.calendar.setDateTextFormat(self.calendar.selectedDate(), fmt_default)  # 리셋용

        # 간단 툴팁: 같은 달의 각 날짜 셀에 일정 요약
        for day, items in self._calendar_cache.items():
            date_obj = self.calendar.selectedDate()
            qdate = date_obj  # 임시
            qdate.setDate(int(self.cmb_year.currentText()), int(self.cmb_month.currentText()), day)
            tips = []
            for it in items:
                t = it.get("title", "-")
                st = it.get("meeting_start", "")[11:16]  # HH:MM
                tips.append(f"{st} {t}")
            self.calendar.setDateTextFormat(qdate, QTextCharFormat())  # 형식은 유지
            self.calendar.setToolTip("\n".join(tips) if tips else "")


    def _compose_schedule_doc(self) -> str:
        """현재 폼 값을 기반으로 RAG에 넣을 문서 문자열을 만든다."""
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("yyyy-MM-dd HH:mm")
        title = self.edit_title.text().strip() or "(제목 미정)"
        loc = self.edit_location.text().strip() or "-"
        pj_s = self.d_project_start.date().toString("yyyy-MM-dd")
        pj_d = self.d_project_due.date().toString("yyyy-MM-dd")
        pay  = self.d_payment_due.date().toString("yyyy-MM-dd")

        todos = [self.list_todo.item(i).text() for i in range(self.list_todo.count())]
        todo_block = "\n".join([f"- {t}" for t in todos]) if todos else "- (없음)"

        # 🔎 검색에 잘 잡히도록 키워드/태그 형식 포함
        # type:schedule, title:, when:, where:, project: 등 명시
        doc = (
            "[SCHEDULE DOC]\n"
            f"type: schedule\n"
            f"title: {title}\n"
            f"when: {s} ~ {e}\n"
            f"where: {loc}\n"
            f"project_start: {pj_s}\n"
            f"project_due: {pj_d}\n"
            f"settlement_due: {pay}\n"
            f"todos:\n{todo_block}\n"
        )
        return doc
    
    def _save_schedule_to_rag(self):
        """현재 스케줄을 RAG에 Segment로 저장(업서트)"""
        if not (self.rag and self.rag.ok):
            return
        from core.audio import Segment

        text = self._compose_schedule_doc()
        seg = Segment(
            text=text,
            start=0.0,                    # 시간 축 사용 안 함
            end=0.0,
            speaker_name="SCHEDULE"       # 검색 시 필터링에 유용
        )
        # 기존 요약 저장과 동일한 방식으로 업서트
        self.rag.upsert_segments([seg])

    def _connect_signals(self):
        self.audio_worker.sig_transcript.connect(self.on_segment)
        self.audio_worker.sig_status.connect(self.on_status)
        self.diar_worker.sig_status.connect(self.on_status)
        self.diar_worker.sig_diar_done.connect(self.on_diar_done)

        # Reverted to original connection
        self.diar_worker.sig_new_speaker.connect(self.on_new_speaker)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_sum.clicked.connect(self.on_summarize)
        self.btn_add2rag.clicked.connect(self.on_index_to_rag)
        self.btn_sched_memo.clicked.connect(self.on_make_schedule)
        self.chk_diar2.stateChanged.connect(self.on_diar_toggle_settings)
        # self.btn_clear_db.clicked.connect(self.on_clear_vector_db)

    def _build_live_tab(self):
        self.live_root = QWidget()
        L = QVBoxLayout(self.live_root)
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
        mid = QHBoxLayout()
        self.chk_diar = QCheckBox("Auto Diarization (pyannote)")
        self.chk_diar.setChecked(self.state.diarization_enabled)
        mid.addWidget(self.chk_diar)
        L.addLayout(mid)
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
        self.lbl_record_status = QLabel("녹음 중지됨")
        self.lbl_record_status.setStyleSheet("color: gray; font-weight: bold;")
        Rv.addWidget(self.lbl_record_status)
        splitter.addWidget(right)
        splitter.setSizes([900, 380])
        L.addWidget(splitter)
        self.tabs.addTab(self.live_root, "Live")

    def _build_minutes_tab(self):
        self.meeting_notes = MeetingNotesView(self)
        self.tabs.addTab(self.meeting_notes, "Minutes")

    def _build_schedule_tab(self):
        """스케줄/프로젝트 관리 탭 (커밋 1d68c94 버전)"""
        root = QWidget()
        H = QHBoxLayout(root)

        # ================== LEFT: Calendar + Form ==================
        left = QWidget()
        L = QVBoxLayout(left)

        # Calendar header: Year/Month selectors
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
        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
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

        # 회의 시작/종료
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

        # To-do list
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

        # 달력 날짜 선택 → 시작/종료 날짜 동기화
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

        # 초기 달력/콤보 동기화
        d = self.dt_start.date()
        self.calendar.setSelectedDate(d)
        self.cmb_year.setCurrentText(str(d.year()))
        self.cmb_month.setCurrentText(f"{d.month():02d}")

        # 초기 미리보기
        self._refresh_schedule_preview()

    def _build_settings_tab(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        system_group = QWidget()
        F = QFormLayout(system_group)

        self.cmb_asr = QComboBox()
        for m in ["small", "medium", "large-v3"]:
            self.cmb_asr.addItem(m)
        self.cmb_asr.setCurrentText(DEFAULT_MODEL)

        self.chk_gpu = QCheckBox("Use GPU if available")
        self.chk_gpu.setChecked(True)

        self.chk_diar2 = QCheckBox("Auto Diarization")
        self.chk_diar2.setChecked(False)

        self.edit_hf = QLineEdit()
        self.edit_hf.setPlaceholderText(f"{HF_TOKEN_ENV} (HuggingFace token)")
        existing_token = os.getenv(HF_TOKEN_ENV, "")
        if existing_token:
            self.edit_hf.setText(f"{existing_token}")
            self.edit_hf.setEchoMode(QLineEdit.EchoMode.Password)

        # Vector DB 초기화 버튼
        self.btn_clear_db = QPushButton("Vector DB 초기화")
        self.btn_clear_db.setStyleSheet("background-color: #fee2e2; color: #991b1b;")

        F.addRow("Whisper Model", self.cmb_asr)
        F.addRow("", self.chk_gpu)
        F.addRow("Auto Diarization", self.chk_diar2)
        F.addRow("HF Token", self.edit_hf)

        # DB 관리 버튼
        db_buttons = QHBoxLayout()
        db_buttons.addWidget(self.btn_clear_db)
        db_buttons.addStretch()
        F.addRow("DB 관리:", db_buttons)

        layout.addWidget(QLabel("🔧 시스템 설정"))
        layout.addWidget(system_group)

        self.meeting_settings = MeetingSettingsWidget(speaker_manager=self.speaker_manager)
        self.meeting_settings.speaker_mapping_changed.connect(self.on_speaker_mapping_changed)
        layout.addWidget(self.meeting_settings)

        self.tabs.addTab(main_widget, "Settings")

        # 버튼 연결
        self.btn_clear_db.clicked.connect(self.on_clear_vector_db)

    def _apply_theme(self):
        self.setStyleSheet(
            f"""QMainWindow {{ background-color: {THEME['bg']}; }}
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

    def on_start(self):
        self.state.diarization_enabled = (self.chk_diar.isChecked() or self.chk_diar2.isChecked())
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
        os.makedirs("output/recordings", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y년%m월%d일_%H시%M분")
        recording_path = f"output/recordings/meeting_{timestamp}.wav"
        self.audio_worker.start_recording(recording_path)
        self.recording = True
        self.recording_start_time = time.time()
        self.lbl_record_status.setText(f"🔴 녹음 중: {os.path.basename(recording_path)}")
        self.lbl_record_status.setStyleSheet("color: red; font-weight: bold;")
        self.on_status(f"Started. 녹음 시작: {recording_path}")

    def on_stop(self):
        saved_path = self.audio_worker.stop_recording() if self.recording else None
        self.recording = False
        try:
            self.audio_worker.stop()
            self.diar_worker.stop()
        except Exception:
            pass
        if saved_path:
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            self.on_status(f"Stopped. 녹음 저장 완료: {saved_path} (시간: {fmt_time(duration)})")
            QMessageBox.information(self, "녹음 완료", f"녹음이 저장되었습니다.\n\n파일: {saved_path}")
        else:
            self.on_status("Stopped.")

    def on_summarize(self):
        summary_text = llm_summarize(self.state.live_segments)
        self.state.summary = summary_text
        items = actions_from_segments(self.state.live_segments)
        self.state.actions = items
        actions_html = render_actions_table_html(items)
        transcript_text = "\n".join([f"[{seg.speaker_name}] {seg.text}" for seg in self.state.live_segments])
        summary_html = f"<pre>{summary_text}</pre>"
        html_for_display = summary_html + actions_html

        # Minutes 탭 업데이트
        self.meeting_notes.update_notes(html_for_display, transcript_text)

        # RAG에 요약과 실시간 세그먼트 저장
        self._save_summary_to_rag(summary_text, items, self.state.live_segments)
        QMessageBox.information(self, "Done", "AI 요약 및 액션 아이템 생성 완료\n요약 문서가 RAG에 저장되었습니다.")

    def _save_summary_to_rag(self, summary_text: str, action_items: list, segments=None):
        """요약과 세그먼트를 RAG에 저장"""
        if not self.rag.ok:
            return

        count = 0

        # 1. 세그먼트 저장 (실제 발언 내용)
        if segments:
            count = self.rag.upsert_segments(segments)
            print(f"[INFO] Saved {count} segments to RAG")

        # 2. 요약 텍스트도 하나의 특별한 세그먼트로 저장 (검색 가능하도록)
        if summary_text and summary_text.strip():
            from core.audio import Segment

            summary_segment = Segment(
                text=f"[회의 요약]\n{summary_text}",
                start=0.0,
                end=0.0,
                speaker_name="SUMMARY"
            )
            self.rag.upsert_segments([summary_segment])
            print("[INFO] Saved summary to RAG")

    def on_index_to_rag(self):
        if not self.rag.ok:
            return
        self.rag.upsert_segments(self.state.live_segments[-50:])
        QMessageBox.information(self, "RAG", "테스트용으로 최근 발언을 RAG에 저장했습니다.")

    def on_diar_toggle_settings(self):
        self.state.diarization_enabled = self.chk_diar2.isChecked()
        self.chk_diar.setChecked(self.state.diarization_enabled)

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

    def on_status(self, msg: str):
        self.txt_status.appendPlainText(f"{now_str()}  {msg}")

    def on_segment(self, seg: Segment):
        if isinstance(seg, dict):
            seg = Segment(**seg)
        self.state.live_segments.append(seg)
        self.list_chat.addItem(QListWidgetItem(f"[{seg.speaker_name}] {seg.text}"))
        self.list_chat.scrollToBottom()

    def on_diar_done(self, results):
        self.state.diar_segments = results
        self.on_status(f"화자 분리 완료: {len(results)}개 구간 처리")
        if hasattr(self, 'meeting_settings') and hasattr(self.meeting_settings, 'speaker_tab'):
            self.meeting_settings.speaker_tab.load_speakers()

    def on_new_speaker(self, speaker_id: str, display_name: str):
        self.state.speaker_map[speaker_id] = display_name
        self.on_status(f"새로운 화자 감지: {speaker_id} ({display_name})")
        if hasattr(self, 'meeting_settings') and hasattr(self.meeting_settings, 'speaker_tab'):
            self.meeting_settings.speaker_tab.load_speakers()

    def on_speaker_mapping_changed(self, mapping: dict):
        self.state.speaker_map.update(mapping)
        if not mapping:
            self.state.speaker_map = {}
        self.on_status(f"화자 매핑 업데이트: {len(mapping)}개")

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
        """우측 Schedule Memo 영역 자동 갱신"""
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
        self.txt_sched.setPlainText(memo)

    def on_make_schedule(self):
        """스케줄 메모 생성 (커밋 1d68c94 버전)"""
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("HH:mm")
        title = self.edit_title.text().strip()
        loc = self.edit_location.text().strip()

        # 1) 자동 안건 추출
        agenda_list = extract_agenda(self.state.live_segments, max_items=5)
        agenda_line = " · ".join(agenda_list) if agenda_list else "-"

        # 2) 기한 있는 Action Item 정리
        lines = []
        for ai in (self.state.actions or []):
            due = ai.get("due") if isinstance(ai, dict) else None
            if due:
                owner = ai.get("owner", "") if isinstance(ai, dict) else ""
                t = ai.get("title", "") if isinstance(ai, dict) else str(ai)
                lines.append(f"[{due}] {t} — {owner}")
        ai_block = ("\n" + "\n".join(lines)) if lines else ""

        pj_s = self.d_project_start.date().toString("yyyy-MM-dd")
        pj_d = self.d_project_due.date().toString("yyyy-MM-dd")
        pay = self.d_payment_due.date().toString("yyyy-MM-dd")

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

        self._save_schedule_json()  # (이미 넣으셨다면 그대로 유지)
        self._save_schedule_to_rag()  # ← 이 줄 추가


    def _refresh_preview(self):
        if not self.state.live_segments:
            return
        recent_segments = self.state.live_segments[-10:]
        preview_lines = [f"[{seg.speaker_name}] {seg.text}" for seg in recent_segments if getattr(seg, "text", "").strip()]
        self.txt_preview.setPlainText("\n".join(preview_lines) if preview_lines else "대화 내용을 분석 중입니다...")
