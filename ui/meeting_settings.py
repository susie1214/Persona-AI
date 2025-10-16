# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QHBoxLayout, QCheckBox,
    QPushButton, QGridLayout, QButtonGroup, QScrollArea, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from core.speaker import SpeakerManager

TEMPLATES = [
    ("일반 회의", "general"),
    ("데일리 스탠드업", "daily"),
    ("스프린트 플래닝", "sprint_planning"),
    ("아키텍처 결정 회의", "adr"),
    ("긴급 장애 대응", "incident"),
    ("성과 리뷰(1:1)", "review"),
    ("브레인스토밍 세션", "brainstorm"),
    ("사용자 정의", "custom"),
]

GLOSSARY_PRESETS = [
    ("카카오 용어집", "kakao"),
    ("인프라 용어집", "infra"),
]

class SpeakerMappingWidget(QWidget):
    """화자 매핑 관리 위젯"""
    mapping_changed = Signal(dict)

    def __init__(self, speaker_manager=None, parent=None):
        super().__init__(parent)
        self.speaker_manager = speaker_manager if speaker_manager else SpeakerManager()
        self.init_ui()
        self.load_speakers()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 제목
        layout.addWidget(QLabel("🎤 화자 매핑 관리"))

        # 테이블
        self.speaker_table = QTableWidget()
        self.speaker_table.setColumnCount(4)
        self.speaker_table.setHorizontalHeaderLabels(["화자 ID", "표시 이름", "임베딩 수", "액션"])

        # 테이블 헤더 설정
        header = self.speaker_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.speaker_table)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 새로고침")
        self.btn_refresh.clicked.connect(self.load_speakers)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_reset = QPushButton("🗑️ 화자 전체 삭제")
        self.btn_reset.setStyleSheet("background-color: #fee2e2; color: #991b1b;")
        self.btn_reset.clicked.connect(self.reset_speakers)
        btn_layout.addWidget(self.btn_reset)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def load_speakers(self):
        """화자 목록 로드 및 테이블 업데이트"""
        # VoiceStore(DB)에서 최신 화자 정보를 강제로 다시 로드하여 메모리 캐시를 갱신
        self.speaker_manager.reload()

        speakers = self.speaker_manager.get_all_speakers()
        print(f"[DEBUG] on setting loading speakers : {speakers}")
        self.speaker_table.setRowCount(len(speakers))

        for row, (speaker_id, display_name, embedding_count) in enumerate(speakers):
            # 화자 ID (읽기 전용)
            id_item = QTableWidgetItem(speaker_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.speaker_table.setItem(row, 0, id_item)

            # 표시 이름 (편집 가능)
            name_item = QTableWidgetItem(display_name)
            self.speaker_table.setItem(row, 1, name_item)

            # 임베딩 수 (읽기 전용)
            count_item = QTableWidgetItem(str(embedding_count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.speaker_table.setItem(row, 2, count_item)

            # 저장 버튼
            save_btn = QPushButton("저장")
            save_btn.clicked.connect(lambda checked, r=row: self.save_speaker_name(r))
            self.speaker_table.setCellWidget(row, 3, save_btn)

    def save_speaker_name(self, row):
        """화자 이름 저장"""
        speaker_id = self.speaker_table.item(row, 0).text()
        new_name = self.speaker_table.item(row, 1).text().strip()

        if not new_name:
            QMessageBox.warning(self, "경고", "화자 이름을 입력해주세요.")
            return

        if self.speaker_manager.update_speaker_name(speaker_id, new_name):
            QMessageBox.information(self, "성공", f"화자 '{speaker_id}'의 이름이 '{new_name}'으로 변경되었습니다.")
            # 화자 매핑 딕셔너리 생성 {speaker_id: display_name}
            mapping = {s.speaker_id: s.display_name for s in self.speaker_manager.speakers.values()}
            self.mapping_changed.emit(mapping)
        else:
            QMessageBox.warning(self, "오류", f"화자 '{speaker_id}' 업데이트에 실패했습니다.")

    def get_speaker_mapping(self) -> dict:
        """현재 화자 매핑 반환"""
        return {s.speaker_id: s.display_name for s in self.speaker_manager.speakers.values()}

    def reset_speakers(self):
        """모든 화자 정보 초기화"""
        # 확인 다이얼로그
        reply = QMessageBox.question(
            self,
            "화자 전체 삭제",
            "모든 화자 정보를 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.speaker_manager.reset_all_speakers():
                self.load_speakers()
                self.mapping_changed.emit({})
                QMessageBox.information(self, "완료", "모든 화자 정보가 삭제되었습니다.")
            else:
                QMessageBox.warning(self, "오류", "화자 정보 삭제 중 오류가 발생했습니다.")

class MeetingSettingsWidget(QWidget):
    """
    회의 설정 및 화자 매핑 관리를 위한 탭 기반 위젯
    - 회의 설정: 참석자, 컨텍스트, 템플릿, 용어집
    - 화자 매핑: 화자 ID와 실제 이름 매핑 관리
    """
    settings_changed = Signal(dict)
    speaker_mapping_changed = Signal(dict)

    def __init__(self, speaker_manager=None, parent=None):
        super().__init__(parent)
        self._settings = {
            "attendees": [],
            "context": "",
            "template": "general",
            "glossaries": [],
            "custom_glossary": "",
        }
        self._speaker_manager = speaker_manager

        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)

        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        root.addWidget(self.tab_widget)

        # 회의 설정 탭
        self.meeting_tab = self.create_meeting_settings_tab()
        self.tab_widget.addTab(self.meeting_tab, "회의 설정")

        # 화자 매핑 탭 (외부에서 주입받은 speaker_manager 사용)
        self.speaker_tab = SpeakerMappingWidget(speaker_manager=self._speaker_manager)
        self.speaker_tab.mapping_changed.connect(self.speaker_mapping_changed.emit)
        self.tab_widget.addTab(self.speaker_tab, "화자 매핑")

        # 심플 스타일
        self.setStyleSheet("""
            QLabel { font-weight: 600; }
            QLineEdit, QTextEdit { border:1px solid #E5E7EB; border-radius:8px; padding:6px; }
            QPushButton[checkable="true"] { border:1px solid #E5E7EB; border-radius:10px; padding:8px; }
            QPushButton[checkable="true"]:checked { background:#EEF2FF; border-color:#6366F1; }
            QTableWidget { border:1px solid #E5E7EB; border-radius:8px; }
            QTabWidget::pane { border:1px solid #E5E7EB; border-radius:8px; }
        """)

    def create_meeting_settings_tab(self):
        """회의 설정 탭 생성"""
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setSpacing(12)

        # 참석자
        root.addWidget(QLabel("👥 참석자 (쉼표로 구분)"))
        self.attendees = QLineEdit()
        self.attendees.setPlaceholderText("예: 신○택, 조○경")
        root.addWidget(self.attendees)

        # 추가 컨텍스트
        root.addWidget(QLabel("🧩 추가 컨텍스트"))
        self.context = QTextEdit()
        self.context.setPlaceholderText("회의 주제, 안건, 목표 등 추가 정보를 입력하세요.")
        self.context.setFixedHeight(90)
        root.addWidget(self.context)

        # 용어집
        root.addWidget(QLabel("📚 용어집"))
        grow = QHBoxLayout()
        self.gloss_checks = []
        for title, key in GLOSSARY_PRESETS:
            cb = QCheckBox(title)
            cb.setProperty("glossary_key", key)
            grow.addWidget(cb)
            self.gloss_checks.append(cb)
        grow.addStretch(1)
        root.addLayout(grow)

        self.custom_gloss = QTextEdit()
        self.custom_gloss.setPlaceholderText("커스텀 용어집 (예: 약어=풀네임; 약어2=설명 ...)")
        self.custom_gloss.setFixedHeight(70)
        root.addWidget(self.custom_gloss)

        # 템플릿 선택
        root.addWidget(QLabel("🧱 회의록 템플릿 선택"))
        grid = QGridLayout()
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        for i, (label, key) in enumerate(TEMPLATES):
            b = QPushButton(label)
            b.setCheckable(True)
            if i == 0:
                b.setChecked(True)
            b.setProperty("tpl_key", key)
            self.btn_group.addButton(b, i)
            grid.addWidget(b, i // 4, i % 4)
        root.addLayout(grid)

        # 적용 버튼
        action = QHBoxLayout()
        action.addStretch(1)
        self.btn_apply = QPushButton("설정 적용")
        action.addWidget(self.btn_apply)
        root.addLayout(action)

        self.btn_apply.clicked.connect(self._on_apply)

        return widget

    def _on_apply(self):
        tpl_key = None
        for b in self.btn_group.buttons():
            if b.isChecked():
                tpl_key = b.property("tpl_key")
                break

        glossaries = [cb.property("glossary_key") for cb in self.gloss_checks if cb.isChecked()]
        s = {
            "attendees": self._parse_attendees(self.attendees.text()),
            "context": self.context.toPlainText().strip(),
            "template": tpl_key or "general",
            "glossaries": glossaries,
            "custom_glossary": self.custom_gloss.toPlainText().strip(),
        }
        self._settings.update(s)
        self.settings_changed.emit(self._settings)

    @staticmethod
    def _parse_attendees(text: str):
        return [x.strip() for x in text.split(",") if x.strip()]

    def get_settings(self) -> dict:
        self._on_apply()
        return self._settings

    def set_settings(self, s: dict):
        s = s or {}
        self.attendees.setText(", ".join(s.get("attendees", [])))
        self.context.setText(s.get("context", ""))
        for b in self.btn_group.buttons():
            b.setChecked(b.property("tpl_key") == s.get("template", "general"))
        for cb in self.gloss_checks:
            cb.setChecked(cb.property("glossary_key") in set(s.get("glossaries", [])))
        self.custom_gloss.setText(s.get("custom_glossary", ""))
        self._settings.update(self.get_settings())

    def refresh_speaker_mapping(self):
        """화자 매핑 탭 새로고침"""
        self.speaker_tab.load_speakers()

    def get_speaker_mapping(self) -> dict:
        """현재 화자 매핑 딕셔너리 반환"""
        return self.speaker_tab.get_speaker_mapping()
