# -*- coding: utf-8 -*-
"""
디지털 페르소나 관리 위젯
- 등록된 페르소나 목록 표시
- 사전 지식 입력 마법사 실행
- 페르소나 상세 정보 조회
- 시스템 프롬프트 미리보기
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QTextEdit, QDialog, QDialogButtonBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal

from core.persona import DigitalPersonaManager
from core.speaker import SpeakerManager
from ui.survey_wizard import DigitalPersonaPriorKnowledgeWizard


class PersonaDetailDialog(QDialog):
    """페르소나 상세 정보 다이얼로그"""

    def __init__(self, persona, parent=None):
        super().__init__(parent)
        self.persona = persona
        self.setWindowTitle(f"페르소나 상세 - {persona.display_name}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 기본 정보
        basic_group = QGroupBox("기본 정보")
        basic_layout = QFormLayout()
        basic_layout.addRow("Speaker ID:", QLabel(self.persona.speaker_id))
        basic_layout.addRow("표시 이름:", QLabel(self.persona.display_name))
        basic_layout.addRow("역할:", QLabel(self.persona.role or "-"))
        basic_layout.addRow("부서:", QLabel(self.persona.department or "-"))
        basic_layout.addRow("발언 수:", QLabel(str(self.persona.utterance_count)))
        basic_layout.addRow("임베딩 품질:", QLabel(f"{self.persona.embedding_quality:.2%}"))
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # 전문성 및 성격
        expertise_group = QGroupBox("전문성 및 성격")
        expertise_layout = QVBoxLayout()
        expertise_layout.addWidget(QLabel(f"전문 분야: {', '.join(self.persona.expertise) or '-'}"))
        expertise_layout.addWidget(QLabel(f"성격 키워드: {', '.join(self.persona.personality_keywords) or '-'}"))
        expertise_group.setLayout(expertise_layout)
        layout.addWidget(expertise_group)

        # 시스템 프롬프트 미리보기
        prompt_group = QGroupBox("시스템 프롬프트 미리보기")
        prompt_layout = QVBoxLayout()
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setPlainText(self.persona.generate_system_prompt())
        self.prompt_text.setFixedHeight(150)
        prompt_layout.addWidget(self.prompt_text)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # 닫기 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class PersonaManagementWidget(QWidget):
    """디지털 페르소나 관리 위젯"""

    persona_updated = Signal(str)  # speaker_id 전달

    def __init__(
        self,
        persona_manager: Optional[DigitalPersonaManager] = None,
        speaker_manager: Optional[SpeakerManager] = None,
        parent=None
    ):
        super().__init__(parent)
        self.persona_manager = persona_manager
        self.speaker_manager = speaker_manager
        self.init_ui()
        self.load_personas()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 제목 및 설명
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("🎭 디지털 페르소나 관리"))
        title_layout.addStretch()
        layout.addLayout(title_layout)

        desc = QLabel(
            "화자의 음성 데이터와 사전 지식을 결합하여 디지털 페르소나를 생성합니다.\n"
            "페르소나는 대화 생성, 회의록 작성, 챗봇 응답에 활용됩니다."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6B7280; font-size: 12px;")
        layout.addWidget(desc)

        # 페르소나 테이블
        self.persona_table = QTableWidget()
        self.persona_table.setColumnCount(6)
        self.persona_table.setHorizontalHeaderLabels([
            "Speaker ID", "이름", "역할", "부서", "발언 수", "액션"
        ])

        # 테이블 헤더 설정
        header = self.persona_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.persona_table)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()

        self.btn_refresh = QPushButton("🔄 새로고침")
        self.btn_refresh.clicked.connect(self.load_personas)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_add = QPushButton("➕ 페르소나 추가")
        self.btn_add.clicked.connect(self.add_persona)
        btn_layout.addWidget(self.btn_add)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 스타일
        self.setStyleSheet("""
            QTableWidget {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                background: white;
            }
            QPushButton:hover {
                background: #F3F4F6;
            }
        """)

    def load_personas(self):
        """페르소나 목록 로드"""
        if not self.persona_manager:
            return

        personas = self.persona_manager.get_all_personas()
        self.persona_table.setRowCount(len(personas))

        for row, persona in enumerate(personas):
            # Speaker ID
            id_item = QTableWidgetItem(persona.speaker_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.persona_table.setItem(row, 0, id_item)

            # 이름
            name_item = QTableWidgetItem(persona.display_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.persona_table.setItem(row, 1, name_item)

            # 역할
            role_item = QTableWidgetItem(persona.role or "-")
            role_item.setFlags(role_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.persona_table.setItem(row, 2, role_item)

            # 부서
            dept_item = QTableWidgetItem(persona.department or "-")
            dept_item.setFlags(dept_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.persona_table.setItem(row, 3, dept_item)

            # 발언 수
            count_item = QTableWidgetItem(str(persona.utterance_count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.persona_table.setItem(row, 4, count_item)

            # 액션 버튼
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(4)

            # 수정 버튼
            edit_btn = QPushButton("수정")
            edit_btn.clicked.connect(lambda checked, sid=persona.speaker_id: self.edit_persona(sid))
            action_layout.addWidget(edit_btn)

            # 상세 버튼
            detail_btn = QPushButton("상세")
            detail_btn.clicked.connect(lambda checked, p=persona: self.show_persona_detail(p))
            action_layout.addWidget(detail_btn)

            # 삭제 버튼
            delete_btn = QPushButton("삭제")
            delete_btn.setStyleSheet("QPushButton { color: #DC2626; }")
            delete_btn.clicked.connect(lambda checked, sid=persona.speaker_id, name=persona.display_name: self.delete_persona(sid, name))
            action_layout.addWidget(delete_btn)

            self.persona_table.setCellWidget(row, 5, action_widget)

    def add_persona(self):
        """새 페르소나 추가 (화자 선택 또는 직접 생성)"""
        from PySide6.QtWidgets import QInputDialog

        # 화자 목록 가져오기 (있을 경우)
        speakers = []
        if self.speaker_manager:
            speakers = self.speaker_manager.get_all_speakers()

        if speakers:
            # 옵션 1: 기존 화자에서 선택
            speaker_items = ["[새 페르소나 직접 생성]"] + [f"{sid} ({name})" for sid, name, _ in speakers]
            item, ok = QInputDialog.getItem(
                self,
                "페르소나 추가",
                "화자를 선택하거나 새로 생성하세요:",
                speaker_items,
                0,
                False
            )

            if not ok or not item:
                return

            if item == "[새 페르소나 직접 생성]":
                # 직접 생성
                speaker_id, display_name = self._create_new_persona_dialog()
                if not speaker_id:
                    return
            else:
                # 기존 화자 선택
                speaker_id = item.split(" (")[0]
                display_name = self.speaker_manager.get_speaker_display_name(speaker_id)
        else:
            # 옵션 2: 화자가 없으면 직접 생성
            speaker_id, display_name = self._create_new_persona_dialog()
            if not speaker_id:
                return

        # 사전 지식 입력 마법사 실행
        wizard = DigitalPersonaPriorKnowledgeWizard(
            speaker_id=speaker_id,
                display_name=display_name,
                persona_manager=self.persona_manager,
                parent=self
            )
        wizard.persona_updated.connect(self.on_persona_updated)
        wizard.exec()

    def edit_persona(self, speaker_id: str):
        """페르소나 수정"""
        if not self.persona_manager:
            return

        persona = self.persona_manager.get_persona(speaker_id)
        if not persona:
            QMessageBox.warning(
                self,
                "페르소나 없음",
                f"페르소나를 찾을 수 없습니다: {speaker_id}"
            )
            return

        # 사전 지식 입력 마법사 실행 (기존 정보 로드는 추후 구현 가능)
        wizard = DigitalPersonaPriorKnowledgeWizard(
            speaker_id=speaker_id,
            display_name=persona.display_name,
            persona_manager=self.persona_manager,
            parent=self
        )
        wizard.persona_updated.connect(self.on_persona_updated)
        wizard.exec()

    def show_persona_detail(self, persona):
        """페르소나 상세 정보 표시"""
        dialog = PersonaDetailDialog(persona, self)
        dialog.exec()

    def delete_persona(self, speaker_id: str, display_name: str):
        """페르소나 삭제"""
        if not self.persona_manager:
            return

        # 확인 다이얼로그
        reply = QMessageBox.question(
            self,
            "페르소나 삭제",
            f"'{display_name}' ({speaker_id}) 페르소나를 삭제하시겠습니까?\n\n"
            f"이 작업은 되돌릴 수 없습니다.\n"
            f"페르소나 정보만 삭제되며, 화자 음성 데이터와 발언 기록은 유지됩니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.persona_manager.delete_persona(speaker_id)
                QMessageBox.information(
                    self,
                    "삭제 완료",
                    f"'{display_name}' 페르소나가 삭제되었습니다."
                )
                self.load_personas()
                self.persona_updated.emit(speaker_id)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "삭제 실패",
                    f"페르소나 삭제 중 오류가 발생했습니다:\n{str(e)}"
                )

    def _create_new_persona_dialog(self):
        """새 페르소나 생성 다이얼로그"""
        from PySide6.QtWidgets import QInputDialog

        # Speaker ID 입력
        speaker_id, ok = QInputDialog.getText(
            self,
            "새 페르소나 생성",
            "페르소나 ID를 입력하세요 (예: jkj, user_01):"
        )

        if not ok or not speaker_id.strip():
            return None, None

        speaker_id = speaker_id.strip()

        # Display Name 입력
        display_name, ok = QInputDialog.getText(
            self,
            "새 페르소나 생성",
            "표시 이름을 입력하세요 (예: 조진경):",
            text=speaker_id
        )

        if not ok or not display_name.strip():
            return None, None

        display_name = display_name.strip()

        return speaker_id, display_name

    def on_persona_updated(self, speaker_id: str):
        """페르소나 업데이트 시 호출"""
        self.load_personas()
        self.persona_updated.emit(speaker_id)
