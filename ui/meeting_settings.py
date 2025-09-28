# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QHBoxLayout, QCheckBox,
    QPushButton, QGridLayout, QButtonGroup, QScrollArea, QMessageBox
)
from PySide6.QtCore import Qt, Signal

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

class MeetingSettingsWidget(QWidget):
    """
    - 참석자(쉼표로 구분)
    - 추가 컨텍스트(회의 목적/안건 등)
    - 템플릿 선택(라디오형 버튼)
    - 용어집(사전) 선택 + 커스텀 용어
    """
    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = {
            "attendees": [],
            "context": "",
            "template": "general",
            "glossaries": [],
            "custom_glossary": "",
        }

        root = QVBoxLayout(self)
        root.setSpacing(12)

        # 참석자
        root.addWidget(QLabel("👥 참석자 (쉼표로 구분)"))
        self.attendees = QLineEdit()
        self.attendees.setPlaceholderText("예: talysa.geist, tj.kim, 조진경")
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

        # 심플 스타일
        self.setStyleSheet("""
            QLabel { font-weight: 600; }
            QLineEdit, QTextEdit { border:1px solid #E5E7EB; border-radius:8px; padding:6px; }
            QPushButton[checkable="true"] { border:1px solid #E5E7EB; border-radius:10px; padding:8px; }
            QPushButton[checkable="true"]:checked { background:#EEF2FF; border-color:#6366F1; }
        """)

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
