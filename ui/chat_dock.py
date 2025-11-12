# ui/chat_dock.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QComboBox, QListWidget, QListWidgetItem, QListView
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Qt, QThread, Signal, QObject
from typing import Optional

from core.llm_router import LLMRouter
from core.digital_persona import DigitalPersonaManager


# ========== LLM 비동기 Worker ==========
class LLMWorker(QObject):
    """LLM 호출을 백그라운드 스레드에서 처리하는 Worker"""
    sig_done = Signal(str)  # 성공 시 응답 텍스트
    sig_error = Signal(str)  # 오류 시 에러 메시지

    def __init__(self, router, backend, prompt, temperature, max_new_tokens=None):
        super().__init__()
        self.router = router
        self.backend = backend
        self.prompt = prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def run(self):
        """LLM 호출 실행 (별도 스레드에서)"""
        try:
            # max_new_tokens가 설정되었으면 전달 (Kanana 등 지원하는 모델용)
            kwargs = {"temperature": self.temperature}
            if self.max_new_tokens is not None:
                kwargs["max_new_tokens"] = self.max_new_tokens

            answer = self.router.complete(self.backend, self.prompt, **kwargs)
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
    "kanana": ("카나나", "resources/kanana.png")
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
    - 상단: Persona 선택 (Backend는 페르소나 설정에서 가져옴)
    - 중앙: 메시지 리스트(QListWidget, 아이콘 포함)
    - 하단: 입력창 + Send (Enter로도 전송)
    """
    def __init__(self, rag_store=None, persona_manager: Optional[DigitalPersonaManager] = None, default_backend: str = "openai:gpt-4o-mini", parent=None):
        super().__init__(parent)
        self.rag_store = rag_store
        self.persona_manager = persona_manager
        self.router = LLMRouter()
        self.active_persona_id = None  # 현재 선택된 페르소나 speaker_id
        self._system_prompt = "You are a helpful assistant."
        self._current_backend = default_backend  # 기본 백엔드 (Settings에서 설정 가능)
        self.setMinimumWidth(360)

        # LLM 비동기 처리용
        self.llm_thread = None
        self.llm_worker = None
        self._current_context = ""  # RAG 컨텍스트 임시 저장

        layout = QVBoxLayout(self)

        # === 상단 Persona 선택 ===
        row = QHBoxLayout()
        row.addWidget(QLabel("대화 상대"))
        self.cmb_persona = QComboBox()
        self.load_personas()
        self.cmb_persona.currentTextChanged.connect(self.on_persona_changed)
        row.addWidget(self.cmb_persona)

        row.addWidget(QLabel("Backend"))
        self.lbl_backend = QLabel("openai:gpt-4o-mini")
        self.lbl_backend.setStyleSheet("color: #6B7280; font-style: italic;")
        row.addWidget(self.lbl_backend)

        layout.addLayout(row)

        # === 중앙: 대화 뷰 (아이콘 포함 리스트) ===
        self.view = QListWidget()
        self.view.setIconSize(QSize(40, 40))
        self.view.setUniformItemSizes(False)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setWordWrap(True)
        # 텍스트 드래그 선택 및 복사 활성화
        self.view.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.view, 1)

        # 초기 상태 안내 (주석 처리 - 대답만 표시)
        # self._append_status(f"🧭 대화 상대: 없음 (회사 전체 챗봇) | backend: {self._current_backend}")

        # === 하단: 입력 ===
        sub = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("메시지를 입력하세요…")
        self.btn = QPushButton("Send")
        sub.addWidget(self.edit, 1)
        sub.addWidget(self.btn)
        layout.addLayout(sub)

        self.btn.clicked.connect(self.on_send)
        # Enter 키는 LLM 처리 중일 때 무시되도록 on_send에서 처리
        self.edit.returnPressed.connect(self._on_enter_pressed)

    def _on_enter_pressed(self):
        """Enter 키 입력 처리 (LLM 처리 중이면 무시)"""
        # LLM 처리 중이면 엔터 키 무시
        if self.llm_thread and self.llm_thread.isRunning():
            return
        self.on_send()

    # ---------- 페르소나 관리 ----------
    def load_personas(self):
        """페르소나 목록 로드 (드롭다운 갱신)"""
        self.cmb_persona.clear()
        self.cmb_persona.addItem("없음 (회사 전체)")

        if self.persona_manager:
            personas = self.persona_manager.get_all_personas()
            for persona in personas:
                display_text = f"{persona.display_name} ({persona.speaker_id})"
                self.cmb_persona.addItem(display_text, userData=persona.speaker_id)

    def refresh_personas(self):
        """외부에서 페르소나 갱신 요청 시 호출"""
        current_text = self.cmb_persona.currentText()
        self.load_personas()
        # 기존 선택 유지 시도
        index = self.cmb_persona.findText(current_text)
        if index >= 0:
            self.cmb_persona.setCurrentIndex(index)

    def set_default_backend(self, backend: str):
        """
        기본 LLM 백엔드 설정 (Settings에서 호출)

        Args:
            backend: 백엔드 ID (예: "openai:gpt-4o-mini")
        """
        # 모든 채팅에 Settings의 기본 백엔드 사용
        self._current_backend = backend
        self.lbl_backend.setText(backend)
        # 백엔드 변경 메시지 제거 (대답만 표시)
        # self._append_status(f"🔧 기본 LLM 백엔드 변경: {backend}")

    # ---------- 내부 유틸 ----------
    def _current_backend_key(self) -> str:
        return _norm_backend_key(self._current_backend)

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
        # 텍스트 선택 가능하게 설정
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsSelectable)
        # 대충 높이 가늠(본문 길이에 따라 늘려줌)
        # approx_lines = max(1, len(text) // 38 + 1)
        # it.setSizeHint(QSize(0, 26 + approx_lines * 18))
        self.view.addItem(it)
        self.view.scrollToBottom()

    # ---------- 이벤트 ----------
    def on_persona_changed(self, display_text: str):
        """페르소나 선택 변경 시"""
        if display_text.startswith("없음"):
            # 회사 전체 챗봇 - Settings에서 설정한 기본 백엔드 사용
            self.active_persona_id = None
            self._system_prompt = "You are a helpful assistant."
            # self._current_backend는 초기화 시 또는 set_default_backend()로 이미 설정됨
            self.lbl_backend.setText(self._current_backend)
            # 페르소나 변경 메시지 제거 (대답만 표시)
            # self._append_status(f"🧭 대화 상대: 없음 (회사 전체 챗봇) | backend: {self._current_backend}")
            return

        # 페르소나 선택 - Settings에서 설정한 기본 백엔드 사용
        index = self.cmb_persona.currentIndex()
        speaker_id = self.cmb_persona.itemData(index)

        if not speaker_id or not self.persona_manager:
            return

        persona = self.persona_manager.get_persona(speaker_id)
        if not persona:
            return

        self.active_persona_id = speaker_id
        self._system_prompt = persona.generate_system_prompt()
        # 페르소나별 백엔드는 사용하지 않고, Settings의 기본 백엔드 사용
        # self._current_backend는 그대로 유지 (Settings에서 설정한 값)
        self.lbl_backend.setText(self._current_backend)

        # 페르소나 변경 메시지 제거 (대답만 표시)
        # self._append_status(
        #     f"🧭 대화 상대: {persona.display_name} | backend: {self._current_backend}"
        # )

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
        backend_label = self._current_backend  # 페르소나 설정에서 가져온 백엔드 사용

        # Kanana 모델용 프롬프트 포맷 (단일 턴 생성)
        prompt = f"{sys_prompt}\n\n"
        if context_block:
            prompt += f"{context_block}\n\n"
        prompt += f"사용자: {q}\n어시스턴트: "  # Kanana 채팅 포맷

        # "생각 중..." 메시지 제거 (대답만 표시)
        # self._append_status("🤔 답변 생성 중...")

        # UI 입력 비활성화
        self.btn.setEnabled(False)
        self.edit.setEnabled(False)

        # 비동기 LLM 호출
        self.llm_thread = QThread()
        # Kanana 모델의 경우 max_new_tokens를 명시적으로 제한하여 과도한 생성 방지
        self.llm_worker = LLMWorker(self.router, backend_label, prompt, temperature=0.3, max_new_tokens=512)
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

        # RAG 컨텍스트 제거 - 대답만 표시
        final_ans = answer
        # if self._current_context:
        #     final_ans += f"\n\n---\n{self._current_context}"

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
