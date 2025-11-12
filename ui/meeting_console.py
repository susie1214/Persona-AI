# -*- coding: utf-8 -*-
# ui/meeting_console.py
import os, datetime, json, time, re, uuid
from typing import List, Dict, Any
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal, QDateTime, QDate, QRect, QObject, QEvent
from PySide6.QtGui import QPainter, QFont, QTextCharFormat, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QPlainTextEdit, QLabel, QTabWidget, QSplitter, QComboBox,
    QCheckBox, QFormLayout, QLineEdit, QMessageBox, QDialog, QDialogButtonBox,
    QDateTimeEdit, QTextEdit, QDockWidget, QCalendarWidget, QDateEdit, QScrollArea,
    QProgressBar,
)

from ui.survey_wizard import PersonaSurveyWizard
from ui.chat_dock import ChatDock
from ui.meeting_notes import MeetingNotesView
from ui.meeting_settings import MeetingSettingsWidget
from ui.documents_tab import DocumentsTab
from core.audio import AudioWorker, Segment, MeetingState, fmt_time, now_str
from core.diarization import DiarizationWorker
from core.summarizer import (
    render_summary_html_from_segments, actions_from_segments,
    render_actions_table_html, extract_agenda, llm_summarize,
    extract_schedules_from_summary,
)
from core.rag_store import RagStore
from core.adapter import AdapterManager
from core.speaker import SpeakerManager
from core.digital_persona import DigitalPersonaManager
from core.persona_store import PersonaStore
from core.voice_store import VoiceStore
from core.persona_training_worker import PersonaTrainingWorker, TrainingProgressWidget
import numpy as np
from core.schedule_store import Schedule as JSONSchedule, save_schedule as json_save, list_month as json_list_month, new_id as json_new_id

# 스케줄 JSON 경로 (삭제/업데이트에 사용)
SCHEDULE_JSON_PATH = Path("schedules.json")

THEME = {
    "bg": "#e6f5e6", "pane": "#99cc99", "light_bg": "#fafffa",
    "btn": "#ffe066", "btn_hover": "#ffdb4d", "btn_border": "#cccc99",
    "btn_ok": "#66cc66", "btn_danger": "#ff6666",
}
HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_MODEL = "medium"


class EmojiCalendar(QCalendarWidget):
    """이모지 마크를 표시할 수 있는 캘린더 (디자인 유지, 덧그리기만)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emoji_marks = {}  # Dict[QDate, str]

    def set_emoji_marks(self, marks: dict):
        """날짜별 이모지 마크 설정"""
        self._emoji_marks = marks
        # QCalendarWidget 전체를 다시 그리도록 요청
        self.updateCells()

    def paintCell(self, painter: QPainter, rect: QRect, date: QDate):
        """각 날짜 셀을 그릴 때 이모지 추가"""
        super().paintCell(painter, rect, date)
        if date in self._emoji_marks:
            painter.save()
            font = painter.font()
            font.setPointSize(font.pointSize() + 2)
            painter.setFont(font)
            painter.drawText(
                rect.adjusted(2, 0, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self._emoji_marks[date]
            )
            painter.restore()


class ScheduleSelectionDialog(QDialog):
    """추출된 일정을 선택하여 달력에 추가하는 대화상자"""

    def __init__(self, schedules: List[Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("회의에서 일정 추출")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.schedules = schedules
        self.selected_schedules = []

        layout = QVBoxLayout(self)

        # 설명 라벨
        info_label = QLabel(f"🎯 회의 요약에서 {len(schedules)}개의 일정을 발견했습니다.\n추가할 일정을 선택하세요:")
        info_label.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(info_label)

        # 일정 목록 (체크박스)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {THEME['light_bg']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 12pt;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            QListWidget::item:hover {{
                background-color: {THEME['pane']};
            }}
        """)

        for idx, sch in enumerate(schedules):
            title = sch.get("title", "제목 없음")
            date = sch.get("date", "날짜 없음")
            time_str = sch.get("time")
            sch_type = sch.get("type", "todo")
            assignee = sch.get("assignee")
            description = sch.get("description", "")

            # 아이콘 선택
            icon_map = {
                "meeting": "🗓️",
                "project": "📁",
                "todo": "✅",
                "deadline": "⏰"
            }
            icon = icon_map.get(sch_type, "📌")

            # 표시 텍스트 구성
            time_part = f" {time_str}" if time_str else ""
            assignee_part = f" ({assignee})" if assignee else ""
            desc_part = f"\n    → {description[:50]}" if description else ""

            display_text = f"{icon} {date}{time_part} - {title}{assignee_part}{desc_part}"

            item = QListWidgetItem(display_text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)  # 기본값: 모두 선택
            item.setData(Qt.ItemDataRole.UserRole, idx)  # 인덱스 저장
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # 전체 선택/해제 버튼
        select_btns = QHBoxLayout()
        btn_select_all = QPushButton("✅ 전체 선택")
        btn_deselect_all = QPushButton("⬜ 전체 해제")
        btn_select_all.clicked.connect(self._select_all)
        btn_deselect_all.clicked.connect(self._deselect_all)
        select_btns.addWidget(btn_select_all)
        select_btns.addWidget(btn_deselect_all)
        select_btns.addStretch()
        layout.addLayout(select_btns)

        # 확인/취소 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def _on_accept(self):
        """선택된 일정만 추출"""
        self.selected_schedules = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                self.selected_schedules.append(self.schedules[idx])
        self.accept()

    def get_selected_schedules(self):
        """선택된 일정 반환"""
        return self.selected_schedules


class _ScheduleListDialog(QDialog):
    """특정 날짜의 일정 목록을 보여주는 대화상자"""

    def __init__(self, date: QDate, schedules: List[Dict], parent=None):
        super().__init__(parent)
        self.date = date
        self.schedules = schedules
        self.selected_schedule = None

        self.setWindowTitle(f"일정 목록 - {date.toString('yyyy년 MM월 dd일')}")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)

        # 헤더
        header = QLabel(f"📅 {date.toString('yyyy년 MM월 dd일')} 일정 ({len(schedules)}개)")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        layout.addWidget(header)

        # 일정 목록
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {THEME['light_bg']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 12pt;
            }}
            QListWidget::item {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            QListWidget::item:hover {{
                background-color: {THEME['pane']};
            }}
        """)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        for sch in schedules:
            title = sch.get("title", "제목 없음")
            meeting_start = sch.get("meeting_start", "")
            location = sch.get("location", "")

            # 시간 추출
            time_str = ""
            if meeting_start:
                try:
                    dt = datetime.fromisoformat(meeting_start)
                    time_str = dt.strftime("%H:%M")
                except:
                    pass

            # 표시 텍스트
            display_text = f"🕐 {time_str} - {title}" if time_str else f"📌 {title}"
            if location:
                display_text += f"\n    📍 {location}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, sch)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_view = QPushButton("📄 상세보기")
        btn_close = QPushButton("닫기")
        btn_view.clicked.connect(self._on_view_clicked)
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_view)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _on_item_double_clicked(self, item):
        """항목 더블클릭시 상세보기"""
        self.selected_schedule = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_view_clicked(self):
        """상세보기 버튼 클릭"""
        current_item = self.list_widget.currentItem()
        if current_item:
            self.selected_schedule = current_item.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def get_selected_schedule(self):
        """선택된 일정 반환"""
        return self.selected_schedule


class _ScheduleDetailDialog(QDialog):
    """일정 상세보기 및 수정/삭제 대화상자"""

    def __init__(self, schedule: Dict, parent=None):
        super().__init__(parent)
        self.schedule = schedule
        self.action = None  # "save", "delete", or None

        self.setWindowTitle("일정 상세")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)

        # 제목
        title_label = QLabel(f"📋 {schedule.get('title', '제목 없음')}")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background-color: {THEME['light_bg']}; border: 1px solid {THEME['btn_border']}; border-radius: 4px;")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # 필드들
        self.edit_title = QLineEdit(schedule.get("title", ""))
        self.edit_location = QLineEdit(schedule.get("location", ""))

        # 회의 시작/종료
        meeting_start = schedule.get("meeting_start", "")
        meeting_end = schedule.get("meeting_end", "")

        self.dt_start = QDateTimeEdit()
        self.dt_end = QDateTimeEdit()

        if meeting_start:
            try:
                dt = datetime.fromisoformat(meeting_start)
                self.dt_start.setDateTime(QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute))
            except:
                self.dt_start.setDateTime(QDateTime.currentDateTime())
        else:
            self.dt_start.setDateTime(QDateTime.currentDateTime())

        if meeting_end:
            try:
                dt = datetime.fromisoformat(meeting_end)
                self.dt_end.setDateTime(QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute))
            except:
                self.dt_end.setDateTime(QDateTime.currentDateTime().addSecs(3600))
        else:
            self.dt_end.setDateTime(QDateTime.currentDateTime().addSecs(3600))

        # 프로젝트 날짜
        self.d_project_start = QDateEdit()
        self.d_project_due = QDateEdit()
        self.d_settlement = QDateEdit()

        project_start = schedule.get("project_start", "")
        if project_start:
            try:
                y, m, d = map(int, project_start.split("-"))
                self.d_project_start.setDate(QDate(y, m, d))
            except:
                self.d_project_start.setDate(QDate.currentDate())
        else:
            self.d_project_start.setDate(QDate.currentDate())

        project_due = schedule.get("project_due", "")
        if project_due:
            try:
                y, m, d = map(int, project_due.split("-"))
                self.d_project_due.setDate(QDate(y, m, d))
            except:
                self.d_project_due.setDate(QDate.currentDate())
        else:
            self.d_project_due.setDate(QDate.currentDate())

        settlement = schedule.get("settlement_at", "")
        if settlement:
            try:
                y, m, d = map(int, settlement.split("-"))
                self.d_settlement.setDate(QDate(y, m, d))
            except:
                self.d_settlement.setDate(QDate.currentDate())
        else:
            self.d_settlement.setDate(QDate.currentDate())

        # TODOs
        self.list_todo = QListWidget()
        todos = schedule.get("todos", []) or []
        for todo in todos:
            self.list_todo.addItem(todo)

        # 폼 레이아웃
        form = QFormLayout()
        form.addRow("제목:", self.edit_title)
        form.addRow("장소:", self.edit_location)
        form.addRow("회의 시작:", self.dt_start)
        form.addRow("회의 종료:", self.dt_end)
        form.addRow("프로젝트 시작:", self.d_project_start)
        form.addRow("프로젝트 마감:", self.d_project_due)
        form.addRow("결제일:", self.d_settlement)
        form.addRow("To-Do 목록:", self.list_todo)

        content_layout.addLayout(form)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 저장")
        btn_delete = QPushButton("🗑️ 삭제")
        btn_cancel = QPushButton("취소")

        btn_save.setStyleSheet(f"background-color: {THEME['btn_ok']}; color: white; font-weight: bold; padding: 8px;")
        btn_delete.setStyleSheet(f"background-color: {THEME['btn_danger']}; color: white; font-weight: bold; padding: 8px;")

        btn_save.clicked.connect(self._on_save)
        btn_delete.clicked.connect(self._on_delete)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def _on_save(self):
        """저장 버튼"""
        self.action = "save"

        # 수정된 값 반영
        self.schedule["title"] = self.edit_title.text()
        self.schedule["location"] = self.edit_location.text()
        self.schedule["meeting_start"] = self.dt_start.dateTime().toString("yyyy-MM-ddTHH:mm:ss")
        self.schedule["meeting_end"] = self.dt_end.dateTime().toString("yyyy-MM-ddTHH:mm:ss")
        self.schedule["project_start"] = self.d_project_start.date().toString("yyyy-MM-dd")
        self.schedule["project_due"] = self.d_project_due.date().toString("yyyy-MM-dd")
        self.schedule["settlement_at"] = self.d_settlement.date().toString("yyyy-MM-dd")

        todos = [self.list_todo.item(i).text() for i in range(self.list_todo.count())]
        self.schedule["todos"] = todos

        self.accept()

    def _on_delete(self):
        """삭제 버튼"""
        reply = QMessageBox.question(
            self,
            "일정 삭제",
            f"정말로 '{self.schedule.get('title', '이 일정')}'을(를) 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.action = "delete"
            self.accept()

    def get_action(self):
        """사용자가 선택한 동작 반환"""
        return self.action

    def get_schedule(self):
        """수정된 일정 반환"""
        return self.schedule


def asdict_schedule(s) -> Dict[str, Any]:
    """Schedule 객체를 dict로 변환하는 헬퍼 함수"""
    if isinstance(s, dict):
        return s
    from dataclasses import asdict
    return asdict(s)


class MeetingConsole(QMainWindow):
    sig_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 보조 서비스")
        self.resize(1280, 860)

        self.state = MeetingState()

        # Core 컴포넌트 초기화
        os.makedirs("data/qdrant_db", exist_ok=True)
        os.makedirs("data/digital_personas", exist_ok=True)

        # RagStore 먼저 초기화 (복잡한 초기화 과정)
        print("[INFO] Initializing RagStore...")
        self.rag = RagStore(persist_path="data/qdrant_db")

        # VoiceStore는 별도 디렉토리 사용 (Qdrant 클라이언트 충돌 방지)
        print("[INFO] Initializing VoiceStore...")
        os.makedirs("data/qdrant_db/voice", exist_ok=True)
        self.voice_store = VoiceStore(persist_path="data/qdrant_db/voice")

        # PersonaStore 초기화
        print("[INFO] Initializing PersonaStore...")
        self.persona_store = PersonaStore()

        # Speaker & Persona 관리자 초기화
        self.speaker_manager = SpeakerManager(voice_store=self.voice_store, persona_manager=None)

        # DigitalPersonaManager 초기화 (항상 시도)
        self.persona_manager = None
        if self.rag.ok and self.voice_store.ok:
            try:
                self.persona_manager = DigitalPersonaManager(
                    voice_store=self.voice_store,
                    rag_store=self.rag,
                    persona_store=self.persona_store,
                    storage_path="data/digital_personas"
                )
                # SpeakerManager에 PersonaManager 연결 (화자 이름 변경 시 페르소나 자동 동기화)
                self.speaker_manager.persona_manager = self.persona_manager
                print("[INFO] DigitalPersonaManager initialized successfully")
            except Exception as e:
                print(f"[WARN] DigitalPersonaManager initialization failed: {e}")
                self.persona_manager = None
        else:
            print("[WARN] Skipping DigitalPersonaManager - RAG or VoiceStore not available")
            print(f"       RagStore.ok: {self.rag.ok}, VoiceStore.ok: {self.voice_store.ok}")

        # Audio & Diarization Workers
        self.audio_worker = AudioWorker(
            self.state,
            speaker_manager=self.speaker_manager,
            persona_manager=self.persona_manager
        )
        self.diar_worker = DiarizationWorker(self.state, speaker_manager=self.speaker_manager)
        self.adapter = AdapterManager()
        self.recording = False
        self.recording_start_time = None

        # QLoRA 학습 관련 초기화
        self.training_workers = {}  # {speaker_id: PersonaTrainingWorker}
        self.auto_training_enabled = True  # 자동 학습 활성화 여부
        self.min_utterances_for_training = 20  # 학습 최소 발언 수

        # LLM 백엔드 설정
        self.default_llm_backend = "kanana:kakaocorp/kanana-1.5-2.1b-instruct"  # 기본 LLM 백엔드

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

        self.on_status("✓ RAG Store 초기화 완료" if self.rag.ok else "⚠ RAG Store 사용 불가")

        self.chat_dock = QDockWidget("Persona Chatbot", self)
        self.chat_panel = ChatDock(
            rag_store=self.rag,
            persona_manager=self.persona_manager,
            default_backend=self.default_llm_backend
        )
        self.chat_dock.setWidget(self.chat_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chat_dock)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(1000)

        self._calendar_cache = {}  # {day: [items]}  로딩 캐시
        self._reload_calendar()    # 현재 연/월 일정 로드 및 표시

        # EmojiCalendar 기능 활성화
        self._promote_calendar_to_emoji()
        self._refresh_calendar_emoji_marks()

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
        # 일정 목록도 업데이트
        self._update_schedule_list()

    def _reload_calendar(self):
        try:
            y = int(self.cmb_year.currentText())
            m = int(self.cmb_month.currentText())
        except Exception:
            # 초기 진입 시 combobox가 아직 준비 안 되었을 수도 있음
            d = self.dt_start.date()
            y, m = d.year(), d.month()
        self._calendar_cache = json_list_month(y, m)  # {day : [items]}

        # 날짜별 강조 표시 및 툴팁
        from PySide6.QtGui import QTextCharFormat, QColor
        from PySide6.QtCore import QDate

        # 모든 날짜 형식 초기화
        fmt_default = QTextCharFormat()

        # 일정이 있는 날짜 강조
        fmt_highlight = QTextCharFormat()
        fmt_highlight.setBackground(QColor("#ffe066"))  # 노란색 배경
        fmt_highlight.setFontWeight(700)  # 볼드체

        # 해당 월의 모든 날짜에 대해 처리
        for day in range(1, 32):
            try:
                qdate = QDate(y, m, day)
                if not qdate.isValid():
                    continue

                if day in self._calendar_cache and self._calendar_cache[day]:
                    # 일정이 있는 날: 강조 표시
                    self.calendar.setDateTextFormat(qdate, fmt_highlight)
                else:
                    # 일정이 없는 날: 기본 형식
                    self.calendar.setDateTextFormat(qdate, fmt_default)
            except Exception:
                pass

        # EmojiCalendar 이모지 마크도 업데이트
        self._refresh_calendar_emoji_marks()


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

        # QLoRA 학습 진행 상황 위젯 추가
        self.training_progress = TrainingProgressWidget()
        self.training_progress.hide()  # 초기에는 숨김
        Rv.addWidget(self.training_progress)

        splitter.addWidget(right)
        splitter.setSizes([900, 380])
        L.addWidget(splitter)
        self.tabs.addTab(self.live_root, "Live")

    def _build_minutes_tab(self):
        self.meeting_notes = MeetingNotesView(
            self,
            speaker_manager=self.speaker_manager,
            persona_manager=self.persona_manager
        )
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

        # Big calendar (with emoji marks)
        self.calendar = EmojiCalendar()
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

        # 선택된 날짜의 일정 목록
        L.addWidget(QLabel("📅 선택된 날짜의 일정:"))
        self.list_schedules = QListWidget()
        self.list_schedules.setMaximumHeight(150)
        self.list_schedules.setStyleSheet(f"""
            QListWidget {{
                background-color: {THEME['light_bg']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {THEME['pane']};
                color: #000;
            }}
        """)
        L.addWidget(self.list_schedules)

        # 일정 관리 버튼
        schedule_btns = QHBoxLayout()
        self.btn_load_schedule = QPushButton("📝 수정")
        self.btn_delete_schedule = QPushButton("🗑️ 삭제")
        self.btn_new_schedule = QPushButton("➕ 새 일정")
        schedule_btns.addWidget(self.btn_load_schedule)
        schedule_btns.addWidget(self.btn_delete_schedule)
        schedule_btns.addWidget(self.btn_new_schedule)
        L.addLayout(schedule_btns)

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

        # 일정 관리 버튼
        self.btn_load_schedule.clicked.connect(self._on_load_schedule)
        self.btn_delete_schedule.clicked.connect(self._on_delete_schedule)
        self.btn_new_schedule.clicked.connect(self._on_new_schedule)

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

        # Vector DB 초기화 버튼들
        self.btn_clear_db = QPushButton("Vector DB 초기화 (회의만)")
        self.btn_clear_db.setStyleSheet("background-color: #fee2e2; color: #991b1b;")

        self.btn_clear_all_db = QPushButton("모든 VectorDB 초기화")
        self.btn_clear_all_db.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold;")

        F.addRow("Whisper Model", self.cmb_asr)
        F.addRow("", self.chk_gpu)
        F.addRow("Auto Diarization", self.chk_diar2)
        F.addRow("HF Token", self.edit_hf)

        # DB 관리 버튼
        db_buttons = QHBoxLayout()
        db_buttons.addWidget(self.btn_clear_db)
        db_buttons.addWidget(self.btn_clear_all_db)
        db_buttons.addStretch()
        F.addRow("DB 관리:", db_buttons)

        layout.addWidget(QLabel("🔧 시스템 설정"))
        layout.addWidget(system_group)

        # QLoRA 학습 설정 추가
        training_group = QWidget()
        T = QFormLayout(training_group)

        self.chk_auto_training = QCheckBox("회의 종료 시 자동 학습")
        self.chk_auto_training.setChecked(self.auto_training_enabled)
        self.chk_auto_training.setToolTip("회의 종료 시 화자별 QLoRA 말투 학습을 자동으로 시작합니다")

        self.spin_min_utterances = QLineEdit()
        self.spin_min_utterances.setText(str(self.min_utterances_for_training))
        self.spin_min_utterances.setPlaceholderText("최소 발언 수")
        self.spin_min_utterances.setToolTip("학습에 필요한 최소 발언 수 (권장: 20개 이상)")

        T.addRow("자동 학습:", self.chk_auto_training)
        T.addRow("최소 발언 수:", self.spin_min_utterances)

        layout.addWidget(QLabel("🧠 QLoRA 페르소나 학습"))
        layout.addWidget(training_group)

        # LLM 백엔드 설정 추가
        llm_group = QWidget()
        L = QFormLayout(llm_group)

        self.cmb_llm_backend = QComboBox()
        # 사용 가능한 LLM 백엔드 목록
        llm_backends = [
            ("OpenAI GPT-4o-mini", "openai:gpt-4o-mini"),
            ("A.X-4.0 (4-bit)", "ax:skt/A.X-4.0"),
            ("Midm-2.0-Mini (4-bit)", "midm:K-intelligence/Midm-2.0-Mini-Instruct"),
            ("Kanana-1.5-2.1b (4-bit)", "kanana:kakaocorp/kanana-1.5-2.1b-instruct"),
            ("Ollama Llama3", "ollama:llama3"),
        ]

        for display_name, backend_id in llm_backends:
            self.cmb_llm_backend.addItem(display_name, backend_id)

        # 기본값 설정
        idx = self.cmb_llm_backend.findData(self.default_llm_backend)
        if idx >= 0:
            self.cmb_llm_backend.setCurrentIndex(idx)

        self.cmb_llm_backend.setToolTip("챗봇 및 요약에 사용할 기본 LLM 백엔드를 선택하세요")

        L.addRow("기본 LLM 백엔드:", self.cmb_llm_backend)

        layout.addWidget(QLabel("🤖 LLM 백엔드 설정"))
        layout.addWidget(llm_group)

        self.meeting_settings = MeetingSettingsWidget(
            speaker_manager=self.speaker_manager,
            persona_manager=self.persona_manager
        )
        self.meeting_settings.speaker_mapping_changed.connect(self.on_speaker_mapping_changed)
        self.meeting_settings.persona_updated.connect(self.on_persona_updated)
        layout.addWidget(self.meeting_settings)

        self.tabs.addTab(main_widget, "Settings")

        # 버튼 연결
        self.btn_clear_db.clicked.connect(self.on_clear_vector_db)
        self.btn_clear_all_db.clicked.connect(self.on_clear_all_vector_db)
        self.chk_auto_training.stateChanged.connect(self._on_auto_training_changed)
        self.spin_min_utterances.textChanged.connect(self._on_min_utterances_changed)
        self.cmb_llm_backend.currentIndexChanged.connect(self._on_llm_backend_changed)

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

        # 회의 종료: 참여한 화자들의 meeting_count 증가 + 자동 학습
        if self.persona_manager and self.state.speaker_map:
            speaker_ids = list(self.state.speaker_map.keys())
            if speaker_ids:
                self.persona_manager.on_meeting_ended(speaker_ids)
                self.on_status(f"회의 종료: {len(speaker_ids)}명 참여자 기록 업데이트")

                # 자동 학습 트리거
                if self.auto_training_enabled:
                    self._trigger_auto_training(speaker_ids)

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

        # 🆕 LLM으로 일정 추출 시도
        extracted_schedules = extract_schedules_from_summary(summary_text, self.state.live_segments)

        if extracted_schedules:
            self._prompt_add_schedules_to_calendar(extracted_schedules)

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
        """Vector DB를 초기화 (회의 컬렉션만)"""
        reply = QMessageBox.question(
            self,
            "Vector DB 초기화",
            "정말로 Vector DB의 모든 데이터를 삭제하시겠습니까?\n(회의 컬렉션만 삭제됩니다)\n이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.rag and self.rag.ok:
                if self.rag.clear_collection():
                    self.on_status("✓ Vector DB가 성공적으로 초기화되었습니다. (회의 컬렉션)")
                    QMessageBox.information(self, "완료", "Vector DB가 초기화되었습니다.\n(회의 컬렉션만 삭제됨)")
                else:
                    self.on_status("⚠ Vector DB 초기화에 실패했습니다.")
                    QMessageBox.warning(self, "오류", "Vector DB 초기화 중 오류가 발생했습니다.")
            else:
                self.on_status("⚠ RAG Store가 초기화되지 않아 DB를 초기화할 수 없습니다.")
                QMessageBox.warning(self, "오류", "RAG Store가 유효하지 않습니다.")

    def on_clear_all_vector_db(self):
        """모든 VectorDB를 초기화 (회의 + 문서)"""
        reply = QMessageBox.warning(
            self,
            "⚠️ 모든 VectorDB 초기화",
            "정말로 모든 VectorDB 데이터를 삭제하시겠습니까?\n\n삭제될 데이터:\n- 회의 컬렉션 (meeting_ctx)\n- 문서 컬렉션 (project_docs)\n\n이 작업은 되돌릴 수 없습니다!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.rag and self.rag.ok:
                if self.rag.clear_all_collections():
                    self.on_status("✓ 모든 VectorDB가 성공적으로 초기화되었습니다.")
                    QMessageBox.information(
                        self,
                        "완료",
                        "모든 VectorDB가 초기화되었습니다.\n\n삭제된 데이터:\n- 회의 컬렉션 (meeting_ctx)\n- 문서 컬렉션 (project_docs)"
                    )
                else:
                    self.on_status("⚠ VectorDB 초기화에 실패했습니다.")
                    QMessageBox.warning(self, "오류", "VectorDB 초기화 중 오류가 발생했습니다.")
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

    def on_persona_updated(self, speaker_id: str):
        """페르소나 업데이트 시 ChatDock 드롭다운 갱신"""
        self.chat_panel.refresh_personas()
        self.on_status(f"페르소나 업데이트: {speaker_id}")

    def _on_year_month_changed(self):
        """연/월 콤보 변경 → 달력 페이지 이동"""
        try:
            y = int(self.cmb_year.currentText())
            m = int(self.cmb_month.currentText())
            self.calendar.setCurrentPage(y, m)
        except Exception:
            pass

    def _on_calendar_selected(self):
        """달력에서 날짜 선택 → 시작/종료 날짜의 '날짜'만 바꾸고 시간은 유지 + 일정 목록 표시"""
        d = self.calendar.selectedDate()
        start = self.dt_start.dateTime()
        end = self.dt_end.dateTime()
        self.dt_start.setDateTime(QDateTime(d, start.time()))
        self.dt_end.setDateTime(QDateTime(d, end.time()))
        # 프로젝트 시작 기본값도 동기
        if not self.edit_title.text().strip():
            self.d_project_start.setDate(d)

        # 선택된 날짜의 일정 목록 표시
        self._update_schedule_list()
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

    def _update_schedule_list(self):
        """선택된 날짜의 일정 목록 업데이트"""
        self.list_schedules.clear()
        d = self.calendar.selectedDate()
        date_str = d.toString("yyyy-MM-dd")

        # 해당 날짜의 일정 가져오기
        from core.schedule_store import list_day
        schedules = list_day(date_str)

        if not schedules:
            return

        for sch in schedules:
            schedule_id = sch.get("id")
            title = sch.get("title", "제목 없음")
            meeting_start = sch.get("meeting_start", "")
            time_str = meeting_start[11:16] if len(meeting_start) > 11 else ""

            # TODO 개수 표시
            todos = sch.get("todos", [])
            todo_count = len(todos) if todos else 0
            todo_str = f" [TODO: {todo_count}]" if todo_count > 0 else ""

            display_text = f"{time_str} {title}{todo_str}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, schedule_id)  # ID 저장
            self.list_schedules.addItem(item)

    def _on_load_schedule(self):
        """선택된 일정을 폼에 로드하여 수정"""
        current_item = self.list_schedules.currentItem()
        if not current_item:
            QMessageBox.warning(self, "일정 선택", "수정할 일정을 먼저 선택해주세요.")
            return

        schedule_id = current_item.data(Qt.ItemDataRole.UserRole)
        from core.schedule_store import get_by_id
        sch = get_by_id(schedule_id)

        if not sch:
            QMessageBox.warning(self, "오류", "일정을 찾을 수 없습니다.")
            return

        # 폼에 데이터 로드
        self.edit_title.setText(sch.get("title", ""))
        self.edit_location.setText(sch.get("location", ""))

        # 날짜/시간 파싱
        meeting_start = sch.get("meeting_start", "")
        meeting_end = sch.get("meeting_end", "")

        if meeting_start:
            dt_start = QDateTime.fromString(meeting_start, "yyyy-MM-ddTHH:mm:ss")
            if dt_start.isValid():
                self.dt_start.setDateTime(dt_start)

        if meeting_end:
            dt_end = QDateTime.fromString(meeting_end, "yyyy-MM-ddTHH:mm:ss")
            if dt_end.isValid():
                self.dt_end.setDateTime(dt_end)

        # 프로젝트 날짜
        if sch.get("project_start"):
            pj_start = QDate.fromString(sch.get("project_start"), "yyyy-MM-dd")
            if pj_start.isValid():
                self.d_project_start.setDate(pj_start)

        if sch.get("project_due"):
            pj_due = QDate.fromString(sch.get("project_due"), "yyyy-MM-dd")
            if pj_due.isValid():
                self.d_project_due.setDate(pj_due)

        if sch.get("settlement_at"):
            settlement = QDate.fromString(sch.get("settlement_at"), "yyyy-MM-dd")
            if settlement.isValid():
                self.d_payment_due.setDate(settlement)

        # TODO 리스트 로드
        self.list_todo.clear()
        todos = sch.get("todos", [])
        if todos:
            for todo in todos:
                self.list_todo.addItem(todo)

        self._refresh_schedule_preview()
        QMessageBox.information(self, "로드 완료", f"'{sch.get('title')}'을(를) 수정할 수 있습니다.\n저장 버튼을 눌러 업데이트하세요.")

    def _on_delete_schedule(self):
        """선택된 일정 삭제"""
        current_item = self.list_schedules.currentItem()
        if not current_item:
            QMessageBox.warning(self, "일정 선택", "삭제할 일정을 먼저 선택해주세요.")
            return

        schedule_id = current_item.data(Qt.ItemDataRole.UserRole)
        from core.schedule_store import get_by_id, delete_schedule

        sch = get_by_id(schedule_id)
        if not sch:
            QMessageBox.warning(self, "오류", "일정을 찾을 수 없습니다.")
            return

        # 확인 대화상자
        reply = QMessageBox.question(
            self,
            "일정 삭제",
            f"'{sch.get('title')}'을(를) 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            delete_schedule(schedule_id)
            self._reload_calendar()
            self._update_schedule_list()
            QMessageBox.information(self, "삭제 완료", "일정이 삭제되었습니다.")

    def _on_new_schedule(self):
        """새 일정 입력 모드 (폼 초기화)"""
        self.edit_title.clear()
        self.edit_location.clear()
        self.list_todo.clear()

        # 현재 선택된 날짜로 기본값 설정
        d = self.calendar.selectedDate()
        today = QDateTime.currentDateTime()

        self.dt_start.setDateTime(QDateTime(d, today.time()))
        self.dt_end.setDateTime(QDateTime(d, today.addSecs(3600).time()))
        self.d_project_start.setDate(d)
        self.d_project_due.setDate(d.addDays(30))
        self.d_payment_due.setDate(d.addDays(14))

        self._refresh_schedule_preview()
        QMessageBox.information(self, "새 일정", "새 일정을 입력하세요.")

    def _promote_calendar_to_emoji(self):
        """EmojiCalendar에 더블클릭 이벤트 필터 설치"""
        class DoubleClickFilter(QObject):
            def __init__(self, parent_console):
                super().__init__()
                self.parent_console = parent_console

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.MouseButtonDblClick:
                    # 더블클릭한 날짜 가져오기
                    date = self.parent_console.calendar.selectedDate()
                    self.parent_console._open_schedule_list_dialog_for(date)
                    return True
                return super().eventFilter(obj, event)

        self._double_click_filter = DoubleClickFilter(self)

        # QCalendarWidget의 내부 QTableView 찾기
        from PySide6.QtWidgets import QTableView
        table_view = self.calendar.findChild(QTableView)
        if table_view:
            table_view.viewport().installEventFilter(self._double_click_filter)
        else:
            # QTableView를 못 찾으면 calendar 자체에 설치
            self.calendar.installEventFilter(self._double_click_filter)

    def _refresh_calendar_emoji_marks(self):
        """현재 달력 캐시를 기반으로 이모지 마크 갱신"""
        from PySide6.QtCore import QDate
        marks = {}

        try:
            y = int(self.cmb_year.currentText())
            m = int(self.cmb_month.currentText())
        except:
            d = self.dt_start.date()
            y, m = d.year(), d.month()

        # _calendar_cache: {day: [items]}
        for day, items in self._calendar_cache.items():
            if items:
                qdate = QDate(y, m, day)
                if qdate.isValid():
                    # 일정 개수에 따라 이모지 선택
                    count = len(items)
                    if count == 1:
                        marks[qdate] = "📌"
                    elif count == 2:
                        marks[qdate] = "📌📌"
                    else:
                        marks[qdate] = f"📌×{count}"

        self.calendar.set_emoji_marks(marks)

    def _open_schedule_list_dialog_for(self, date: QDate):
        """특정 날짜의 일정 목록 대화상자 열기"""
        date_str = date.toString("yyyy-MM-dd")
        from core.schedule_store import list_day
        schedules = list_day(date_str)

        if not schedules:
            QMessageBox.information(self, "일정 없음", f"{date.toString('yyyy년 MM월 dd일')}에는 등록된 일정이 없습니다.")
            return

        # 일정 목록 대화상자
        dialog = _ScheduleListDialog(date, schedules, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_schedule = dialog.get_selected_schedule()
            if selected_schedule:
                self._edit_schedule_dialog(selected_schedule)

    def _edit_schedule_dialog(self, schedule: Dict):
        """일정 상세보기 및 수정/삭제 대화상자"""
        dialog = _ScheduleDetailDialog(schedule, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            action = dialog.get_action()
            updated_schedule = dialog.get_schedule()

            if action == "save":
                self._upsert_schedule_item(updated_schedule)
                self._save_specific_schedule_to_rag(updated_schedule)
                QMessageBox.information(self, "저장 완료", "일정이 업데이트되었습니다.")
            elif action == "delete":
                self._delete_schedule_by_id(updated_schedule.get("id"))
                QMessageBox.information(self, "삭제 완료", "일정이 삭제되었습니다.")

            # 달력 갱신
            self._reload_calendar()
            self._refresh_calendar_emoji_marks()
            self._update_schedule_list()

    def _save_specific_schedule_to_rag(self, schedule: Dict):
        """특정 스케줄을 RAG에 저장"""
        if not (self.rag and self.rag.ok):
            return
        from core.audio import Segment

        # 문서 생성
        title = schedule.get("title", "제목 없음")
        meeting_start = schedule.get("meeting_start", "")
        meeting_end = schedule.get("meeting_end", "")
        location = schedule.get("location", "") or "-"
        project_start = schedule.get("project_start", "")
        project_due = schedule.get("project_due", "")
        settlement = schedule.get("settlement_at", "")
        todos = schedule.get("todos", []) or []

        todo_block = "\n".join([f"- {t}" for t in todos]) if todos else "- (없음)"

        doc = (
            "[SCHEDULE DOC]\n"
            f"type: schedule\n"
            f"title: {title}\n"
            f"when: {meeting_start} ~ {meeting_end}\n"
            f"where: {location}\n"
            f"project_start: {project_start}\n"
            f"project_due: {project_due}\n"
            f"settlement_due: {settlement}\n"
            f"todos:\n{todo_block}\n"
        )

        seg = Segment(
            text=doc,
            start=0.0,
            end=0.0,
            speaker_name="SCHEDULE"
        )

        doc_id = f"schedule_{schedule.get('id', uuid.uuid4())}"
        self.rag.add_segments([seg], doc_id=doc_id)
        print(f"[INFO] Schedule saved to RAG: {doc_id}")

    def _upsert_schedule_item(self, schedule: Dict):
        """일정 업데이트 (schedule_store.py 사용)"""
        from core.schedule_store import save_schedule

        sch = JSONSchedule(
            id=schedule.get("id"),
            title=schedule.get("title", ""),
            location=schedule.get("location"),
            meeting_start=schedule.get("meeting_start", ""),
            meeting_end=schedule.get("meeting_end", ""),
            project_start=schedule.get("project_start"),
            project_due=schedule.get("project_due"),
            settlement_at=schedule.get("settlement_at"),
            todos=schedule.get("todos", [])
        )
        save_schedule(sch)

    def _delete_schedule_by_id(self, schedule_id: int):
        """ID로 일정 삭제"""
        from core.schedule_store import delete_schedule
        delete_schedule(schedule_id)

        # RAG에서도 삭제
        if self.rag and self.rag.ok:
            doc_id = f"schedule_{schedule_id}"
            # Qdrant는 delete_by_id 미지원이므로 필터링으로 삭제 (또는 직접 구현 필요)
            # 여기서는 로그만 남김
            print(f"[INFO] Schedule deleted from store: {doc_id}")

    def _prompt_add_schedules_to_calendar(self, schedules: List[Dict]):
        """추출된 일정을 사용자에게 확인받고 달력에 추가"""
        # 대화상자 표시
        dialog = ScheduleSelectionDialog(schedules, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = dialog.get_selected_schedules()
        if not selected:
            return

        # 선택된 일정을 달력에 추가
        added_count = 0
        for sch in selected:
            try:
                # Schedule 객체 생성
                title = sch.get("title", "제목 없음")
                date_str = sch.get("date")
                time_str = sch.get("time")
                description = sch.get("description", "")
                assignee = sch.get("assignee")

                # 시간 처리 (없으면 기본값 09:00 ~ 10:00)
                if time_str:
                    meeting_start = f"{date_str}T{time_str}:00"
                    # 종료 시간은 1시간 후
                    start_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    end_dt = start_dt + datetime.timedelta(hours=1)
                    meeting_end = end_dt.strftime("%Y-%m-%dT%H:%M:00")
                else:
                    meeting_start = f"{date_str}T09:00:00"
                    meeting_end = f"{date_str}T10:00:00"

                # 담당자를 장소나 설명에 추가
                location = None
                if assignee:
                    location = f"담당: {assignee}"
                    if description:
                        description = f"[{assignee}] {description}"

                # TODO 리스트 (description을 TODO로 변환)
                todos = []
                if description:
                    todos = [description]

                # 스케줄 저장
                schedule = JSONSchedule(
                    id=json_new_id(),
                    title=title,
                    location=location,
                    meeting_start=meeting_start,
                    meeting_end=meeting_end,
                    project_start=date_str,
                    project_due=date_str,
                    settlement_at=None,
                    todos=todos
                )

                json_save(schedule)
                added_count += 1

            except Exception as e:
                print(f"[ERROR] 일정 저장 실패: {e}")
                continue

        # 달력 갱신
        if added_count > 0:
            self._reload_calendar()
            self._update_schedule_list()
            QMessageBox.information(
                self,
                "일정 추가 완료",
                f"{added_count}개의 일정이 달력에 추가되었습니다."
            )

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

    # ========== QLoRA 자동 학습 관련 메서드 ==========

    def _on_auto_training_changed(self, state):
        """자동 학습 설정 변경"""
        self.auto_training_enabled = (state == Qt.CheckState.Checked.value)
        self.on_status(f"자동 학습: {'활성화' if self.auto_training_enabled else '비활성화'}")

    def _on_min_utterances_changed(self, text):
        """최소 발언 수 설정 변경"""
        try:
            value = int(text)
            if value > 0:
                self.min_utterances_for_training = value
        except ValueError:
            pass

    def _on_llm_backend_changed(self, index):
        """LLM 백엔드 설정 변경"""
        backend_id = self.cmb_llm_backend.itemData(index)
        if backend_id:
            self.default_llm_backend = backend_id
            self.on_status(f"기본 LLM 백엔드 변경: {backend_id}")

            # ChatDock에도 반영
            if hasattr(self, 'chat_panel'):
                self.chat_panel.set_default_backend(backend_id)

    def _trigger_auto_training(self, speaker_ids: List[str]):
        """
        회의 종료 시 화자별 자동 학습 트리거 (순차 학습)

        Args:
            speaker_ids: 참여한 화자 ID 리스트
        """
        if not self.rag or not self.rag.ok:
            self.on_status("⚠ RAG Store 없음 - 학습 불가")
            return

        # 필터링: 발언 수 충분한 화자만 추출
        speakers_to_train = []
        for speaker_id in speaker_ids:
            try:
                results = self.rag.search_by_speaker(speaker_id, query="", topk=1000)

                # 짧은 발언 필터링 (3단어 이상만 학습 대상)
                valid_utterances = [
                    utt for utt in results
                    if utt.get("text") and len(utt.get("text", "").strip().split()) >= 3
                ]
                utterance_count = len(valid_utterances)

                if utterance_count < self.min_utterances_for_training:
                    self.on_status(
                        f"⏭ {speaker_id}: 유효한 발언 수 부족 ({utterance_count}/{self.min_utterances_for_training}) - 학습 건너뜀 "
                        f"(전체: {len(results)}개, 필터링됨: {len(results) - utterance_count}개)"
                    )
                    continue

                # 화자 이름 가져오기
                speaker_name = self.state.speaker_map.get(speaker_id, speaker_id)
                speakers_to_train.append((speaker_id, speaker_name, utterance_count))

            except Exception as e:
                self.on_status(f"❌ {speaker_id} 학습 체크 실패: {e}")

        # 순차 학습: 한 명씩 완료 후 다음 사람 진행
        if speakers_to_train:
            self.on_status(f"📋 총 {len(speakers_to_train)}명의 화자 순차 학습 시작")
            self._train_speakers_sequentially(speakers_to_train, index=0)

    def _train_speakers_sequentially(self, speakers_to_train: List[tuple], index: int):
        """
        화자들을 순차적으로 학습 (재귀함수)

        Args:
            speakers_to_train: [(speaker_id, speaker_name, utterance_count), ...] 리스트
            index: 현재 학습할 화자의 인덱스
        """
        if index >= len(speakers_to_train):
            # 모든 화자 학습 완료
            self.on_status(f"✅ 모든 화자 학습 완료!")
            return

        speaker_id, speaker_name, utterance_count = speakers_to_train[index]
        self.on_status(f"🔄 [{index + 1}/{len(speakers_to_train)}] {speaker_name} 학습 시작...")

        # 다음 화자 학습을 위한 콜백 등록
        def on_next_speaker():
            self.on_status(f"✅ {speaker_name} 학습 완료! 다음 화자 준비 중...")
            self._train_speakers_sequentially(speakers_to_train, index + 1)

        # 현재 화자 학습 시작 (완료 시 on_next_speaker 호출)
        self._start_training_with_callback(speaker_id, speaker_name, utterance_count, on_next_speaker)

    def _start_training_with_callback(self, speaker_id: str, speaker_name: str, utterance_count: int, on_complete_callback):
        """
        특정 화자의 QLoRA 학습 시작 (완료 콜백 포함)

        Args:
            speaker_id: 화자 ID
            speaker_name: 화자 이름
            utterance_count: 발언 수
            on_complete_callback: 학습 완료 시 호출할 콜백 함수
        """
        # 이미 학습 중인지 체크
        if speaker_id in self.training_workers:
            existing_worker = self.training_workers[speaker_id]
            if existing_worker.isRunning():
                self.on_status(f"⚠ {speaker_name} 이미 학습 중")
                return

        # Worker 생성
        worker = PersonaTrainingWorker(
            rag_store=self.rag,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            min_utterances=self.min_utterances_for_training,
            num_epochs=3,          # 원래 설정
            batch_size=4,          # 원래 설정
        )

        # 시그널 연결
        worker.sig_status.connect(self._on_training_status)
        worker.sig_progress.connect(self._on_training_progress)
        # 완료 시 콜백 함수 먼저 호출 후 기본 처리
        worker.sig_finished.connect(lambda sid, path: (
            on_complete_callback(),
            self._on_training_finished(sid, path)
        ))
        worker.sig_error.connect(self._on_training_error)

        # 진행 위젯 표시
        self.training_progress.reset()
        self.training_progress.show()
        self.training_progress.update_status(f"🚀 {speaker_name} 학습 준비 중...")

        # 학습 시작
        self.training_workers[speaker_id] = worker
        worker.start()

        self.on_status(f"🧠 {speaker_name} QLoRA 학습 시작 (발언: {utterance_count}개)")

    def _start_training(self, speaker_id: str, speaker_name: str, utterance_count: int):
        """
        특정 화자의 QLoRA 학습 시작

        Args:
            speaker_id: 화자 ID
            speaker_name: 화자 이름
            utterance_count: 발언 수
        """
        # 이미 학습 중인지 체크
        if speaker_id in self.training_workers:
            existing_worker = self.training_workers[speaker_id]
            if existing_worker.isRunning():
                self.on_status(f"⚠ {speaker_name} 이미 학습 중")
                return

        # Worker 생성
        worker = PersonaTrainingWorker(
            rag_store=self.rag,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            min_utterances=self.min_utterances_for_training,
            num_epochs=1,          # 원래 설정
            batch_size=2,          # 원래 설정
        )

        # 시그널 연결
        worker.sig_status.connect(self._on_training_status)
        worker.sig_progress.connect(self._on_training_progress)
        worker.sig_finished.connect(self._on_training_finished)
        worker.sig_error.connect(self._on_training_error)

        # 진행 위젯 표시
        self.training_progress.reset()
        self.training_progress.show()
        self.training_progress.update_status(f"🚀 {speaker_name} 학습 준비 중...")

        # 학습 시작
        self.training_workers[speaker_id] = worker
        worker.start()

        self.on_status(f"🧠 {speaker_name} QLoRA 학습 시작 (발언: {utterance_count}개)")

    def _on_training_status(self, message: str):
        """학습 상태 메시지 업데이트"""
        self.training_progress.update_status(message)
        self.on_status(message)

    def _on_training_progress(self, progress: int):
        """학습 진행률 업데이트"""
        self.training_progress.update_progress(progress)

    def _on_training_finished(self, speaker_id: str, adapter_path: str):
        """학습 완료 처리"""
        speaker_name = self.state.speaker_map.get(speaker_id, speaker_id)

        self.training_progress.set_success()
        self.on_status(f"✅ {speaker_name} 학습 완료!")
        self.on_status(f"   어댑터 저장 위치: {adapter_path}")

        # 3초 후 진행 위젯 숨김
        QTimer.singleShot(3000, self.training_progress.hide)

        # DigitalPersona에 어댑터 경로 저장
        if self.persona_manager:
            try:
                persona = self.persona_manager.get_persona(speaker_id)
                if persona:
                    persona.qlora_adapter_path = adapter_path
                    self.persona_manager.save_persona(speaker_id)
                    self.on_status(f"   페르소나에 어댑터 경로 저장됨")
            except Exception as e:
                self.on_status(f"⚠ 페르소나 업데이트 실패: {e}")

        # Worker 정리
        if speaker_id in self.training_workers:
            del self.training_workers[speaker_id]

        # 완료 알림
        QMessageBox.information(
            self,
            "학습 완료",
            f"{speaker_name}님의 말투 학습이 완료되었습니다!\n\n"
            f"어댑터: {adapter_path}\n\n"
            f"이제 챗봇에서 {speaker_name}님의 말투로 대화할 수 있습니다."
        )

    def _on_training_error(self, error_msg: str):
        """학습 에러 처리"""
        self.training_progress.set_error(error_msg)
        self.on_status(f"❌ 학습 실패: {error_msg}")

        # 5초 후 진행 위젯 숨김
        QTimer.singleShot(5000, self.training_progress.hide)

        # 에러 다이얼로그
        QMessageBox.warning(
            self,
            "학습 실패",
            f"페르소나 학습 중 오류가 발생했습니다.\n\n{error_msg[:200]}\n\n"
            f"PEFT 라이브러리가 설치되어 있는지 확인하세요:\n"
            f"pip install peft transformers accelerate bitsandbytes"
        )
