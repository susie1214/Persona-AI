# -*- coding: utf-8 -*-
# ui/chat_dock.py
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QComboBox,
)
from PySide6.QtCore import Qt
from core.llm_router import LLMRouter
from core.persona_store import PersonaStore


class ChatDock(QWidget):
    """
    - 상단에서 '활성 페르소나'를 표시/선택(자동 주입도 가능)
    - 페르소나 system 프롬프트 + 사용자 메시지로 LLMRouter 통해 호출
    """

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.store = PersonaStore()
        self.router = LLMRouter()
        self.active_persona = None  # ex) "조진경"
        self.setMinimumWidth(360)

        L = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Persona"))
        self.cmb_persona = QComboBox()
        names = [k for k in self.store.data.keys() if k != "default_style"]
        if not names:
            names = ["(없음)"]
        for n in names:
            self.cmb_persona.addItem(n)
        self.cmb_persona.currentTextChanged.connect(self.on_persona_changed)
        row.addWidget(self.cmb_persona)

        row.addWidget(QLabel("Backend"))
        self.cmb_backend = QComboBox()
        # 선택지 예시
        for b in [
            "openai:gpt-4o-mini",
            "ollama:llama3",
            "ax:A.X-4.0",
            "midm:Midm-2.0-Mini-Instruct",
        ]:
            self.cmb_backend.addItem(b)
        row.addWidget(self.cmb_backend)
        L.addLayout(row)

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        L.addWidget(self.view, 1)

        sub = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("메시지 입력…")
        self.btn = QPushButton("Send")
        sub.addWidget(self.edit, 1)
        sub.addWidget(self.btn)
        L.addLayout(sub)

        self.btn.clicked.connect(self.on_send)

        # # 초기 상태
        # if names and names[0] != "(없음)":
        #     self.set_active_persona(names[0])
        # else:
        #     self.set_active_persona(None)

    # 외부(회의 Live 탭)에서 발화자에 맞게 호출
    def set_active_persona(self, name: str | None):
        self.active_persona = name
        sys = self.store.build_system_prompt(name)
        be = self.store.choose_backend(name)
        self._system_prompt = sys
        # 콤보박스 표시 동기화
        if name and self.cmb_persona.findText(name) >= 0:
            self.cmb_persona.setCurrentText(name)
        if self.cmb_backend.findText(be) >= 0:
            self.cmb_backend.setCurrentText(be)
        self.view.append(f"🧭 Active persona: {name or '(없음)'}  |  backend: {be}")

    def on_persona_changed(self, name: str):
        if name == "(없음)":
            name = "미지정"
            
        self.set_active_persona(name)

    def on_send(self):
        q = self.edit.text().strip()
        if not q:
            return
        self.edit.clear()
        sys_prompt = self._system_prompt or "You are a helpful assistant."
        backend = self.cmb_backend.currentText() or self.store.choose_backend(
            self.active_persona
        )

        # 간단한 system+user 프롬프트 결합
        prompt = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{q}"
        try:
            ans = self.router.complete(backend, prompt, temperature=0.2)
        except Exception as e:
            ans = f"(오류: {e})"
        self.view.append(f"👤 {self.active_persona or 'User'}: {q}")
        self.view.append(f"🤖 {backend}: {ans}\n")
