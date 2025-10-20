# -*- coding: utf-8 -*-
"""
화자 & 페르소나 통합 관리 위젯
- 화자 정보 (ID, 이름, 임베딩) + 페르소나 정보 (역할, 부서, 발언)를 하나의 테이블에서 관리
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from core.speaker import SpeakerManager
from core.digital_persona import DigitalPersonaManager
from ui.survey_wizard import DigitalPersonaPriorKnowledgeWizard
from ui.persona_management import PersonaDetailDialog


class SpeakerPersonaWidget(QWidget):
    """화자 & 페르소나 통합 관리 위젯"""

    mapping_changed = Signal(dict)  # 화자 매핑 변경 시그널
    persona_updated = Signal(str)   # 페르소나 업데이트 시그널 (speaker_id)

    def __init__(
        self,
        speaker_manager: Optional[SpeakerManager] = None,
        persona_manager: Optional[DigitalPersonaManager] = None,
        parent=None
    ):
        super().__init__(parent)
        self.speaker_manager = speaker_manager if speaker_manager else SpeakerManager()
        self.persona_manager = persona_manager
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 제목 및 설명
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("🎭 화자 & 페르소나 관리"))
        title_layout.addStretch()
        layout.addLayout(title_layout)

        desc = QLabel(
            "화자의 음성 데이터와 페르소나 정보를 통합 관리합니다.\n"
            "이름 편집 후 '저장'을 클릭하고, '페르소나 설정'으로 상세 정보를 입력하세요."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6B7280; font-size: 12px;")
        layout.addWidget(desc)

        # 통합 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "화자 ID", "표시 이름", "역할", "부서", "발언 수", "임베딩 수", "액션"
        ])

        # 테이블 헤더 설정
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(6, 320)  # 액션 열 너비 고정

        layout.addWidget(self.table)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()

        self.btn_refresh = QPushButton("🔄 새로고침")
        self.btn_refresh.clicked.connect(self.load_data)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_reset = QPushButton("🗑️ 화자 전체 삭제")
        self.btn_reset.setStyleSheet("background-color: #fee2e2; color: #991b1b;")
        self.btn_reset.clicked.connect(self.reset_all_speakers)
        btn_layout.addWidget(self.btn_reset)

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

    def load_data(self):
        """화자 및 페르소나 데이터 로드"""
        # VoiceStore에서 최신 화자 정보 로드
        self.speaker_manager.reload()

        # 모든 화자 가져오기
        speakers = self.speaker_manager.get_all_speakers()
        self.table.setRowCount(len(speakers))

        for row, (speaker_id, display_name, embedding_count) in enumerate(speakers):
            # 페르소나 정보 가져오기 (있으면)
            persona = None
            if self.persona_manager:
                persona = self.persona_manager.get_persona(speaker_id)

            # 화자 ID (읽기 전용)
            id_item = QTableWidgetItem(speaker_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, id_item)

            # 표시 이름 (편집 가능)
            name_item = QTableWidgetItem(display_name)
            self.table.setItem(row, 1, name_item)

            # 역할 (편집 가능, 페르소나가 있으면 표시)
            role = persona.role if persona else ""
            role_item = QTableWidgetItem(role)
            self.table.setItem(row, 2, role_item)

            # 부서 (편집 가능, 페르소나가 있으면 표시)
            dept = persona.department if persona else ""
            dept_item = QTableWidgetItem(dept)
            self.table.setItem(row, 3, dept_item)

            # 발언 수 (읽기 전용)
            utterance_count = persona.utterance_count if persona else 0
            count_item = QTableWidgetItem(str(utterance_count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 4, count_item)

            # 임베딩 수 (읽기 전용)
            emb_item = QTableWidgetItem(str(embedding_count))
            emb_item.setFlags(emb_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 5, emb_item)

            # 액션 버튼들
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(4)

            # 저장 버튼
            save_btn = QPushButton("저장")
            save_btn.setToolTip("이름/역할/부서 저장")
            save_btn.clicked.connect(lambda checked, r=row: self.save_row_data(r))
            action_layout.addWidget(save_btn)

            # 페르소나 설정 버튼
            persona_btn = QPushButton("페르소나")
            persona_btn.setToolTip("상세 페르소나 설정 (설문조사)")
            persona_btn.clicked.connect(lambda checked, sid=speaker_id: self.setup_persona(sid))
            action_layout.addWidget(persona_btn)

            # 상세 버튼 (페르소나가 있을 때만 활성화)
            detail_btn = QPushButton("상세")
            detail_btn.setEnabled(persona is not None)
            detail_btn.setToolTip("페르소나 상세 정보 보기")
            if persona:
                detail_btn.clicked.connect(lambda checked, p=persona: self.show_detail(p))
            action_layout.addWidget(detail_btn)

            # 삭제 버튼
            delete_btn = QPushButton("삭제")
            delete_btn.setStyleSheet("QPushButton { color: #DC2626; }")
            delete_btn.setToolTip("화자 및 페르소나 삭제")
            delete_btn.clicked.connect(
                lambda checked, sid=speaker_id, name=display_name: self.delete_speaker(sid, name)
            )
            action_layout.addWidget(delete_btn)

            self.table.setCellWidget(row, 6, action_widget)

    def save_row_data(self, row):
        """테이블 행 데이터 저장 (이름, 역할, 부서)"""
        speaker_id = self.table.item(row, 0).text()
        new_name = self.table.item(row, 1).text().strip()
        new_role = self.table.item(row, 2).text().strip()
        new_dept = self.table.item(row, 3).text().strip()

        if not new_name:
            QMessageBox.warning(self, "경고", "표시 이름을 입력해주세요.")
            return

        # 1. 화자 이름 업데이트
        if self.speaker_manager.update_speaker_name(speaker_id, new_name):
            # 2. 페르소나가 있으면 역할/부서 업데이트
            if self.persona_manager:
                persona = self.persona_manager.get_persona(speaker_id)
                if persona:
                    self.persona_manager.update_persona(
                        speaker_id,
                        role=new_role,
                        department=new_dept
                    )

            QMessageBox.information(
                self,
                "성공",
                f"'{speaker_id}'의 정보가 업데이트되었습니다."
            )
            self.load_data()
            self.mapping_changed.emit(self.get_speaker_mapping())
            self.persona_updated.emit(speaker_id)
        else:
            QMessageBox.warning(self, "오류", f"화자 '{speaker_id}' 업데이트에 실패했습니다.")

    def setup_persona(self, speaker_id: str):
        """페르소나 설정 마법사 실행"""
        if not self.persona_manager:
            QMessageBox.warning(
                self,
                "페르소나 관리자 없음",
                "페르소나 관리자가 초기화되지 않았습니다."
            )
            return

        display_name = self.speaker_manager.get_speaker_display_name(speaker_id)

        # 설문조사 마법사 실행
        wizard = DigitalPersonaPriorKnowledgeWizard(
            speaker_id=speaker_id,
            display_name=display_name,
            persona_manager=self.persona_manager,
            parent=self
        )
        wizard.persona_updated.connect(self.on_persona_updated)
        wizard.exec()

    def show_detail(self, persona):
        """페르소나 상세 정보 다이얼로그"""
        dialog = PersonaDetailDialog(persona, self)
        dialog.exec()

    def delete_speaker(self, speaker_id: str, display_name: str):
        """화자 및 페르소나 삭제"""
        reply = QMessageBox.question(
            self,
            "화자 삭제",
            f"'{display_name}' ({speaker_id}) 화자를 삭제하시겠습니까?\n\n"
            f"이 작업은 되돌릴 수 없습니다.\n"
            f"화자 음성 데이터와 관련된 페르소나도 함께 삭제됩니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.speaker_manager.delete_speaker(speaker_id):
                    QMessageBox.information(
                        self,
                        "삭제 완료",
                        f"'{display_name}' 화자가 삭제되었습니다."
                    )
                    self.load_data()
                    self.mapping_changed.emit(self.get_speaker_mapping())
                    self.persona_updated.emit(speaker_id)
                else:
                    QMessageBox.warning(
                        self,
                        "삭제 실패",
                        f"화자 '{speaker_id}' 삭제 중 오류가 발생했습니다."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "삭제 실패",
                    f"화자 삭제 중 오류가 발생했습니다:\n{str(e)}"
                )

    def reset_all_speakers(self):
        """모든 화자 정보 초기화"""
        reply = QMessageBox.question(
            self,
            "화자 전체 삭제",
            "모든 화자 정보를 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.speaker_manager.reset_all_speakers():
                self.load_data()
                self.mapping_changed.emit({})
                QMessageBox.information(self, "완료", "모든 화자 정보가 삭제되었습니다.")
            else:
                QMessageBox.warning(self, "오류", "화자 정보 삭제 중 오류가 발생했습니다.")

    def on_persona_updated(self, speaker_id: str):
        """페르소나 업데이트 시 호출"""
        self.load_data()
        self.persona_updated.emit(speaker_id)

    def get_speaker_mapping(self) -> dict:
        """현재 화자 매핑 반환"""
        return {s.speaker_id: s.display_name for s in self.speaker_manager.speakers.values()}
