# # -*- coding: utf-8 -*-
# # ui/chat_dock.py
# from PySide6.QtWidgets import (
#     QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton,
#     QLabel, QHBoxLayout, QComboBox
# )
# from PySide6.QtGui import QIcon, QPixmap
# from PySide6.QtCore import QSize
# from core.llm_router import LLMRouter

# from core.persona_store import PersonaStore
# # 백엔드 이름 → 디스플레이명/아이콘 경로 매핑
# AVATAR_PATHS = {
#     "user":          ("You",              "data/user.png"),
#     # 백엔드 키를 너희 코드에서 쓰는 정확한 이름으로 맞추세요
#     "openai:gpt-4o-mini": ("ChatGPT",     "data/chatgpt.png"),
#     "llama3":        ("Llama 3",          "data/llama.png"),      
#     "A_X-4.0":       ("A.(에이닷)",       "data/aidot.png"),
#     "Midm-2.0-Mini-Instruct": ("믿:음K 2.0","data/mideumk.png"),
# }

# def _icon_from(path: str) -> QIcon:
#     pm = QPixmap(path)
#     if pm.isNull():
#         return QIcon()  # fallback
#     return QIcon(pm)


# class ChatDock(QWidget):
#     """
#     - 초기에는 일반 챗봇으로 동작
#     - 회의 중 자동 페르소나가 생성되면 set_active_persona()로 교체 가능
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.store = PersonaStore()
#         self.router = LLMRouter()
#         self.active_persona = None
#         self._system_prompt = "You are a helpful assistant."  # 기본값
#         self.setMinimumWidth(360)
#         layout = QVBoxLayout(self)
#         # 상단 Persona / Backend 선택
#         row = QHBoxLayout()
#         row.addWidget(QLabel("Persona"))
#         self.cmb_persona = QComboBox()
#         self.cmb_persona.addItem("(없음)")  # 기본 챗봇 모드
#         for k in self.store.data.keys():
#             if k != "default_style":
#                 self.cmb_persona.addItem(k)
#         self.cmb_persona.currentTextChanged.connect(self.on_persona_changed)
#         row.addWidget(self.cmb_persona)
#         row.addWidget(QLabel("Backend"))
#         self.cmb_backend = QComboBox()
#         for b in [
#             "openai:gpt-4o-mini",
#             "ollama:llama3",
#             "ax:A.X-4.0",
#             "midm:Midm-2.0-Mini-Instruct",
#         ]:
#             self.cmb_backend.addItem(b)
#         row.addWidget(self.cmb_backend)
#         layout.addLayout(row)
#         # 대화 뷰
#         self.view = QTextEdit()
#         self.view.setReadOnly(True)
#         layout.addWidget(self.view, 1)
#         # 입력창
#         sub = QHBoxLayout()
#         self.edit = QLineEdit()
#         self.edit.setPlaceholderText("메시지를 입력하세요…")
#         self.btn = QPushButton("Send")
#         sub.addWidget(self.edit, 1)
#         sub.addWidget(self.btn)
#         layout.addLayout(sub)
#         self.btn.clicked.connect(self.on_send)
#         self.edit.returnPressed.connect(self.on_send)
#         # :흰색_확인_표시: 초기 상태: 기본 챗봇 모드
#         default_be = "openai:gpt-4o-mini"
#         self.cmb_backend.setCurrentText(default_be)
#         self.view.append(f":나침반: Active persona: (없음) | backend: {default_be}")
#     def on_persona_changed(self, name: str):
#         if name == "(없음)":
#             self.active_persona = None
#             self._system_prompt = "You are a helpful assistant."
#             return
#         self.set_active_persona(name)
#     def set_active_persona(self, name: str | None):
#         """외부에서 자동 페르소나 주입 시 호출"""
#         self.active_persona = name
#         sys = self.store.build_system_prompt(name)
#         self._system_prompt = sys or "You are a helpful assistant."
#         be = self.store.choose_backend(name)
#         if self.cmb_backend.findText(be) >= 0:
#             self.cmb_backend.setCurrentText(be)
#         self.view.append(f":나침반: Active persona: {name or '(없음)'} | backend: {be}")
#     def on_send(self):
#         q = self.edit.text().strip()
#         if not q:
#             return
#         self.edit.clear()
#         sys_prompt = self._system_prompt
#         backend = self.cmb_backend.currentText()
#         prompt = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{q}"
#         try:
#             ans = self.router.complete(backend, prompt, temperature=0.3)
#         except Exception as e:
#             ans = f"(오류: {e})"
#         # 출력
#         self.view.append(f"{self.active_persona or 'User'}: {q}")
#         self.view.append(f"{backend}: {ans}\n")


# # # -*- coding: utf-8 -*-
# # # ui/chat_dock.py
# # from PySide6.QtWidgets import (
# #     QWidget,
# #     QVBoxLayout,
# #     QTextEdit,
# #     QLineEdit,
# #     QPushButton,
# #     QLabel,
# #     QHBoxLayout,
# #     QComboBox,
# # )
# # from PySide6.QtCore import Qt
# # from core.llm_router import LLMRouter
# # from core.persona_store import PersonaStore


# # class ChatDock(QWidget):
# #     """
# #     - 상단에서 '활성 페르소나'를 표시/선택(자동 주입도 가능)
# #     - 페르소나 system 프롬프트 + 사용자 메시지로 LLMRouter 통해 호출
# #     """

# #     def __init__(self, model=None, parent=None):
# #         super().__init__(parent)
# #         self.store = PersonaStore()
# #         self.router = LLMRouter()
# #         self.active_persona = None  # ex) "조진경"
# #         self.setMinimumWidth(360)

# #         L = QVBoxLayout(self)

# #         row = QHBoxLayout()
# #         row.addWidget(QLabel("Persona"))
# #         self.cmb_persona = QComboBox()
# #         names = [k for k in self.store.data.keys() if k != "default_style"]
# #         if not names:
# #             names = ["(없음)"]
# #         for n in names:
# #             self.cmb_persona.addItem(n)
# #         self.cmb_persona.currentTextChanged.connect(self.on_persona_changed)
# #         row.addWidget(self.cmb_persona)

# #         row.addWidget(QLabel("Backend"))
# #         self.cmb_backend = QComboBox()
# #         # 선택지 예시
# #         for b in [
# #             "openai:gpt-4o-mini",
# #             "ollama:llama3",
# #             "ax:A.X-4.0",
# #             "midm:Midm-2.0-Mini-Instruct",
# #         ]:
# #             self.cmb_backend.addItem(b)
# #         row.addWidget(self.cmb_backend)
# #         L.addLayout(row)

# #         self.view = QTextEdit()
# #         self.view.setReadOnly(True)
# #         L.addWidget(self.view, 1)

# #         sub = QHBoxLayout()
# #         self.edit = QLineEdit()
# #         self.edit.setPlaceholderText("메시지 입력…")
# #         self.btn = QPushButton("Send")
# #         sub.addWidget(self.edit, 1)
# #         sub.addWidget(self.btn)
# #         L.addLayout(sub)

# #         self.btn.clicked.connect(self.on_send)

# #         # # 초기 상태
# #         # if names and names[0] != "(없음)":
# #         #     self.set_active_persona(names[0])
# #         # else:
# #         #     self.set_active_persona(None)

# #     # 외부(회의 Live 탭)에서 발화자에 맞게 호출
# #     def set_active_persona(self, name: str | None):
# #         self.active_persona = name
# #         sys = self.store.build_system_prompt(name)
# #         be = self.store.choose_backend(name)
# #         self._system_prompt = sys
# #         # 콤보박스 표시 동기화
# #         if name and self.cmb_persona.findText(name) >= 0:
# #             self.cmb_persona.setCurrentText(name)
# #         if self.cmb_backend.findText(be) >= 0:
# #             self.cmb_backend.setCurrentText(be)
# #         self.view.append(f"🧭 Active persona: {name or '(없음)'}  |  backend: {be}")

# #     def on_persona_changed(self, name: str):
# #         if name == "(없음)":
# #             name = "미지정"
            
# #         self.set_active_persona(name)

# #     def on_send(self):
# #         q = self.edit.text().strip()
# #         if not q:
# #             return
# #         self.edit.clear()
# #         sys_prompt = self._system_prompt or "You are a helpful assistant."
# #         backend = self.cmb_backend.currentText() or self.store.choose_backend(
# #             self.active_persona
# #         )

# #         # 간단한 system+user 프롬프트 결합
# #         prompt = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{q}"
# #         try:
# #             ans = self.router.complete(backend, prompt, temperature=0.2)
# #         except Exception as e:
# #             ans = f"(오류: {e})"
# #         self.view.append(f"👤 {self.active_persona or 'User'}: {q}")
# #         self.view.append(f"🤖 {backend}: {ans}\n")


# -*- coding: utf-8 -*-
# ui/chat_dock.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QComboBox, QListWidget, QListWidgetItem, QListView
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Qt

from core.llm_router import LLMRouter
from core.persona_store import PersonaStore


# 백엔드 이름 → 디스플레이명/아이콘 경로 매핑 (요청 경로 사용)
AVATAR_PATHS = {
    "user": ("You", "data/user.png"),
    "openai:gpt-4o-mini": ("ChatGPT", "data/chatgpt.png"),
    "llama3": ("Llama 3", "data/llama.png"),
    "A_X-4.0": ("A.(에이닷)", "data/aidot.png"),
    "Midm-2.0-Mini-Instruct": ("믿:음K 2.0", "data/mideumk.png"),
}

def _icon_from(path: str) -> QIcon:
    pm = QPixmap(path)
    if pm.isNull():
        return QIcon()
    return QIcon(pm)

def _norm_backend_key(text: str) -> str:
    """
    콤보박스에 들어가는 표시 문자열을 AVATAR_PATHS 키로 정규화.
    - 'ollama:llama3' -> 'llama3'
    - 'ax:A.X-4.0'    -> 'A_X-4.0'
    - 'midm:Midm-2.0-Mini-Instruct' -> 'Midm-2.0-Mini-Instruct'
    - 나머지는 그대로 사용
    """
    if ":" in text:
        left, right = text.split(":", 1)
        # 특수 케이스 맵핑
        if left == "ollama":
            return right
        if left == "ax":
            return "A_X-4.0"
        if left == "midm":
            return right
    return text


class ChatDock(QWidget):
    """
    Persona Chatbot 패널
    - 상단: Persona/Backend 선택(Backend 아이콘 표시)
    - 중앙: 메시지 리스트(QListWidget, 아이콘 포함)
    - 하단: 입력창 + Send (Enter로도 전송)
    """
    def __init__(self, rag_store=None, parent=None):
        super().__init__(parent)
        self.rag_store = rag_store
        self.store = PersonaStore()
        self.router = LLMRouter()
        self.active_persona = None
        self._system_prompt = "You are a helpful assistant."
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        # === 상단 Persona / Backend ===
        row = QHBoxLayout()
        row.addWidget(QLabel("Persona"))
        self.cmb_persona = QComboBox()
        self.cmb_persona.addItem("(없음)")
        for k in self.store.data.keys():
            if k != "default_style":
                self.cmb_persona.addItem(k)
        self.cmb_persona.currentTextChanged.connect(self.on_persona_changed)
        row.addWidget(self.cmb_persona)

        row.addWidget(QLabel("Backend"))
        self.cmb_backend = QComboBox()
        # 표시 문자열(라벨)을 그대로 넣되, 아이콘은 정규화된 키 기준으로 세팅
        backends = [
            "openai:gpt-4o-mini",
            "ollama:llama3",
            "ax:A.X-4.0",
            "midm:Midm-2.0-Mini-Instruct",
        ]
        for b in backends:
            self.cmb_backend.addItem(b)
        # 아이콘 부착
        for i in range(self.cmb_backend.count()):
            label = self.cmb_backend.itemText(i)
            key = _norm_backend_key(label)
            disp, icon_path = AVATAR_PATHS.get(key, (label, ""))
            if icon_path:
                self.cmb_backend.setItemIcon(i, _icon_from(icon_path))
        row.addWidget(self.cmb_backend)

        layout.addLayout(row)

        # === 중앙: 대화 뷰 (아이콘 포함 리스트) ===
        self.view = QListWidget()
        self.view.setIconSize(QSize(40, 40))
        self.view.setUniformItemSizes(False)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setWordWrap(True)
        layout.addWidget(self.view, 1)

        # 초기 상태 안내
        default_be = "openai:gpt-4o-mini"
        self.cmb_backend.setCurrentText(default_be)
        self._append_status(f"🧭 Active persona: (없음) | backend: {default_be}")

        # === 하단: 입력 ===
        sub = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("메시지를 입력하세요…")
        self.btn = QPushButton("Send")
        sub.addWidget(self.edit, 1)
        sub.addWidget(self.btn)
        layout.addLayout(sub)

        self.btn.clicked.connect(self.on_send)
        self.edit.returnPressed.connect(self.on_send)

    # ---------- 내부 유틸 ----------
    def _current_backend_key(self) -> str:
        label = self.cmb_backend.currentText()
        return _norm_backend_key(label)

    def _append_status(self, text: str):
        it = QListWidgetItem(text)
        it.setFlags(it.flags()) #  & ~Qt.ItemFlag.ItemIsSelectable
        self.view.addItem(it)
        self.view.scrollToBottom()

    def _append_message(self, role: str, text: str, backend_key: str | None = None):
        """
        role: 'user' | 'assistant' (assistant일 때 backend_key 사용)
        """
        if role == "user":
            disp, icon_path = AVATAR_PATHS.get("user", ("You", ""))
            label = disp
            icon = _icon_from(icon_path)
        else:
            key = backend_key or self._current_backend_key()
            disp, icon_path = AVATAR_PATHS.get(key, (key, ""))
            label = disp
            icon = _icon_from(icon_path)

        # 라벨 + 본문(두 줄)
        text_block = f"{label}\n{text}"
        it = QListWidgetItem(icon, text_block)
        # 대충 높이 가늠(본문 길이에 따라 늘려줌)
        # approx_lines = max(1, len(text) // 38 + 1)
        # it.setSizeHint(QSize(0, 26 + approx_lines * 18))
        self.view.addItem(it)
        self.view.scrollToBottom()

    # ---------- 이벤트 ----------
    def on_persona_changed(self, name: str):
        if name == "(없음)":
            self.active_persona = None
            self._system_prompt = "You are a helpful assistant."
            return
        self.set_active_persona(name)

    def set_active_persona(self, name: str | None):
        """외부에서 자동 페르소나 주입 시 호출"""
        self.active_persona = name
        sys = self.store.build_system_prompt(name)
        self._system_prompt = sys or "You are a helpful assistant."
        be = self.store.choose_backend(name)
        if self.cmb_backend.findText(be) >= 0:
            self.cmb_backend.setCurrentText(be)
        self._append_status(f"🧭 Active persona: {name or '(없음)'} | backend: {be}")

    def on_send(self):
        q = self.edit.text().strip()
        if not q:
            return
        self.edit.clear()

        print(f"[DEBUG] User Query: {q}") # 사용자 쿼리 출력

        # 사용자 메시지 렌더
        self._append_message("user", q)

        # RAG 컨텍스트 검색
        context_block = ""
        if self.rag_store and self.rag_store.ok:
            ctx = self.rag_store.search(q, topk=3)
            
            print(f"[DEBUG - chat_dock] searched context : {ctx}")
            if ctx:
                context_lines = ["[관련 회의 내용]", "-" * 20]
                for c in ctx:
                    context_lines.append(f"- {c.get('text', '')}")
                context_block = "\n".join(context_lines)
        
        print(f"[DEBUG] RAG Context:\n{context_block}") # RAG 컨텍스트 출력

        # 백엔드 호출
        sys_prompt = self._system_prompt
        backend_label = self.cmb_backend.currentText()
        backend_key = self._current_backend_key()
        
        # 프롬프트에 컨텍스트 추가
        prompt = f"[SYSTEM]\n{sys_prompt}\n\n"
        if context_block:
            prompt += f"[CONTEXT]\n{context_block}\n\n"
        prompt += f"[USER]\n{q}"

        try:
            ans = self.router.complete(backend_label, prompt, temperature=0.3)
        except Exception as e:
            ans = f"(오류: {e})"

        # 모델 응답 렌더(아이콘은 backend_key 기준)
        final_ans = ans
        if context_block:
            final_ans += f"\n\n---\n{context_block}"
        self._append_message("assistant", final_ans, backend_key=backend_key)
