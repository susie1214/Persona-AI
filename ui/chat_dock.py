# ui/chat_dock.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QComboBox, QListWidget, QListWidgetItem, QListView
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Qt, QThread, Signal, QObject

from core.llm_router import LLMRouter
from core.persona_store import PersonaStore


# ========== LLM 비동기 Worker ==========
class LLMWorker(QObject):
    """LLM 호출을 백그라운드 스레드에서 처리하는 Worker"""
    sig_done = Signal(str)  # 성공 시 응답 텍스트
    sig_error = Signal(str)  # 오류 시 에러 메시지

    def __init__(self, router, backend, prompt, temperature):
        super().__init__()
        self.router = router
        self.backend = backend
        self.prompt = prompt
        self.temperature = temperature

    def run(self):
        """LLM 호출 실행 (별도 스레드에서)"""
        try:
            answer = self.router.complete(self.backend, self.prompt, temperature=self.temperature)
            self.sig_done.emit(answer)
        except Exception as e:
            import traceback
            error_msg = f"LLM 오류: {str(e)}\n{traceback.format_exc()}"
            self.sig_error.emit(error_msg)


# 백엔드 이름 → 디스플레이명/아이콘 경로 매핑 (요청 경로 사용)
AVATAR_PATHS = {
    "user": ("You", "resources/user.png"),
    "openai:gpt-4o-mini": ("ChatGPT", "resources/chatgpt.png"),
    "llama3": ("Llama 3", "resources/llama.png"),
    "A_X-4.0": ("A.(에이닷)", "resources/aidot.png"),
    "Midm-2.0-Mini-Instruct": ("믿:음K 2.0", "resources/mideumk.png"),
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

        # LLM 비동기 처리용
        self.llm_thread = None
        self.llm_worker = None
        self._current_context = ""  # RAG 컨텍스트 임시 저장

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
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
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

        # 이미 LLM 처리 중이면 무시
        if self.llm_thread and self.llm_thread.isRunning():
            self._append_status("⚠️ 이전 요청 처리 중입니다. 잠시만 기다려주세요...")
            return

        self.edit.clear()
        print(f"[DEBUG] User Query: {q}")

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

        print(f"[DEBUG] RAG Context:\n{context_block}")
        self._current_context = context_block  # 나중에 응답에 추가하기 위해 저장

        # 프롬프트 생성
        sys_prompt = self._system_prompt
        backend_label = self.cmb_backend.currentText()

        prompt = f"[SYSTEM]\n{sys_prompt}\n\n"
        if context_block:
            prompt += f"[CONTEXT]\n{context_block}\n\n"
        prompt += f"[USER]\n{q}"

        # "생각 중..." 메시지 표시
        self._append_status("🤔 답변 생성 중...")

        # UI 입력 비활성화
        self.btn.setEnabled(False)
        self.edit.setEnabled(False)

        # 비동기 LLM 호출
        self.llm_thread = QThread()
        self.llm_worker = LLMWorker(self.router, backend_label, prompt, temperature=0.3)
        self.llm_worker.moveToThread(self.llm_thread)

        # 시그널 연결
        self.llm_thread.started.connect(self.llm_worker.run)
        self.llm_worker.sig_done.connect(self._on_llm_done)
        self.llm_worker.sig_error.connect(self._on_llm_error)
        self.llm_worker.sig_done.connect(self.llm_thread.quit)
        self.llm_worker.sig_error.connect(self.llm_thread.quit)
        self.llm_thread.finished.connect(self._on_llm_finished)

        # 스레드 시작
        self.llm_thread.start()

    def _on_llm_done(self, answer: str):
        """LLM 응답 성공"""
        backend_key = self._current_backend_key()

        # 응답에 컨텍스트 추가
        final_ans = answer
        if self._current_context:
            final_ans += f"\n\n---\n{self._current_context}"

        self._append_message("assistant", final_ans, backend_key=backend_key)

    def _on_llm_error(self, error_msg: str):
        """LLM 오류 처리"""
        self._append_message("assistant", f"❌ {error_msg}", backend_key=None)

    def _on_llm_finished(self):
        """LLM 처리 완료 (성공/실패 무관)"""
        # UI 다시 활성화
        self.btn.setEnabled(True)
        self.edit.setEnabled(True)
        self._current_context = ""  # 컨텍스트 초기화
