# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar,
    QTextEdit, QHBoxLayout, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from core.offline_meeting import process_audio_file
from core.notes_export import save_markdown, save_html, start_share_server
import os, webbrowser, datetime
from ui.meeting_settings import MeetingSettingsWidget
from typing import Optional

# class UploadMinutesWidget(QWidget):
#     def __init__(self):
#         super().__init__()
#         # ... (기존 위젯 구성)
#         # 헤더 영역에 설정 버튼 추가
#         top = QHBoxLayout()
#         top.addStretch(1)
#         self.btn_settings = QPushButton("⚙ 회의록 설정")
#         top.addWidget(self.btn_settings)
#         self.layout().insertLayout(1, top)  # 타이틀 다음 줄에 배치

#         self.settings_widget = MeetingSettingsWidget()
#         self._settings_cache = self.settings_widget.get_settings()

#         self.btn_settings.clicked.connect(self._open_settings)
        
class _Worker(QObject):
    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(
        self,
        path: str,
        asr_model: str,
        use_gpu: bool,
        diarize: bool,
        use_llm_summary: bool = False,
        llm_backend: Optional[str] = None,
        speaker_manager=None,
        persona_manager=None
    ):
        super().__init__()
        self.path = path
        self.asr_model = asr_model
        self.use_gpu = use_gpu
        self.diarize = diarize
        self.use_llm_summary = use_llm_summary
        self.llm_backend = llm_backend
        self.speaker_manager = speaker_manager
        self.persona_manager = persona_manager

    def run(self):
        try:
            res = process_audio_file(
                self.path,
                asr_model=self.asr_model,
                use_gpu=self.use_gpu,
                diarize=self.diarize,
                use_llm_summary=self.use_llm_summary,
                llm_backend=self.llm_backend,
                speaker_manager=self.speaker_manager,
                persona_manager=self.persona_manager
            )
            self.sig_done.emit(res)
        except Exception as e:
            self.sig_error.emit(str(e))

class UploadMinutesWidget(QWidget):
    def __init__(self, speaker_manager=None, persona_manager=None, rag_store=None,
                 auto_training_enabled=True, min_utterances_for_training=20):
        super().__init__()
        self.setObjectName("UploadMinutes")
        self.speaker_manager = speaker_manager
        self.persona_manager = persona_manager
        self.rag_store = rag_store
        self.auto_training_enabled = auto_training_enabled
        self.min_utterances_for_training = min_utterances_for_training
        self.training_workers = {}  # {speaker_id: PersonaTrainingWorker}
        L = QVBoxLayout(self)

        self.title = QLabel("<h2>모든 회의가 <span style='color:#3b82f6'>기록</span>되는 순간.</h2>"
                            "<div style='color:#6b7280'>오디오를 텍스트로, 대화를 인사이트로.<br>"
                            "AI가 당신의 회의를 완벽하게 기록합니다.</div>")
        self.title.setAlignment(Qt.AlignCenter)
        L.addWidget(self.title)

        # LLM 요약 옵션 추가
        from PySide6.QtWidgets import QCheckBox, QComboBox
        options_layout = QHBoxLayout()
        self.chk_llm_summary = QCheckBox("🤖 AI 요약 사용 (LLM)")
        self.chk_llm_summary.setChecked(False)
        self.chk_llm_summary.setToolTip("LLM을 사용하여 회의록을 지능적으로 요약합니다 (OpenAI API 키 필요)")
        options_layout.addWidget(self.chk_llm_summary)

        self.combo_llm_backend = QComboBox()
        self.combo_llm_backend.addItems([
            "openai:gpt-4o-mini",
            "openai:gpt-4o",
            "openai:gpt-3.5-turbo"
        ])
        self.combo_llm_backend.setEnabled(False)
        self.combo_llm_backend.setToolTip("사용할 LLM 모델 선택")
        options_layout.addWidget(self.combo_llm_backend)
        options_layout.addStretch()

        self.chk_llm_summary.toggled.connect(lambda checked: self.combo_llm_backend.setEnabled(checked))

        L.addLayout(options_layout)

        self.btn_upload = QPushButton("⬆  파일 업로드")
        self.btn_upload.setFixedHeight(44); self.btn_upload.setStyleSheet("font-weight:600;border-radius:10px;")
        L.addWidget(self.btn_upload, alignment=Qt.AlignCenter)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        L.addWidget(self.progress)

        self.md = QTextEdit(); self.md.setReadOnly(True); self.md.setVisible(False)
        L.addWidget(self.md)

        # action bar
        bar = QHBoxLayout()
        self.btn_copy = QPushButton("📋 복사"); self.btn_copy.setEnabled(False)
        self.btn_save_md = QPushButton("⬇ Markdown 저장"); self.btn_save_md.setEnabled(False)
        self.btn_save_html = QPushButton("⬇ HTML 저장"); self.btn_save_html.setEnabled(False)
        self.btn_share = QPushButton("🔗 URL 공유(로컬)"); self.btn_share.setEnabled(False)
        bar.addWidget(self.btn_copy); bar.addWidget(self.btn_save_md); bar.addWidget(self.btn_save_html); bar.addWidget(self.btn_share)
        L.addLayout(bar)

        self.url_box = QLineEdit(); self.url_box.setReadOnly(True); self.url_box.setPlaceholderText("공유 URL")
        self.url_box.setVisible(False); L.addWidget(self.url_box)

        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_copy.clicked.connect(self.on_copy)
        self.btn_save_md.clicked.connect(self.on_save_md)
        self.btn_save_html.clicked.connect(self.on_save_html)
        self.btn_share.clicked.connect(self.on_share)

        self._last_result = None
        self._last_html_path = None

    # ---------- slots ----------
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "오디오 파일 선택", "", "Audio/Video Files (*.wav *.mp3 *.m4a *.mp4 *.aac *.flac);;All Files (*)")
        if not path: return
        self.progress.setVisible(True); self.progress.setRange(0,0)
        self.md.setVisible(False); self.url_box.setVisible(False)
        # 모델/옵션은 간단히 고정 (필요하면 UI로 노출)
        asr_model, use_gpu, diarize = "medium", True, True

        # LLM 요약 옵션 가져오기
        use_llm_summary = self.chk_llm_summary.isChecked()
        llm_backend = self.combo_llm_backend.currentText() if use_llm_summary else None

        self.thread = QThread()
        self.worker = _Worker(
            path, asr_model, use_gpu, diarize,
            use_llm_summary, llm_backend,
            self.speaker_manager, self.persona_manager
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.sig_done.connect(self.on_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_done.connect(self.thread.quit); self.worker.sig_error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_done(self, res: dict):
        self._last_result = res
        self.progress.setVisible(False)
        self.md.setVisible(True)
        self.md.setPlainText(res["markdown"])
        self.btn_copy.setEnabled(True)
        self.btn_save_md.setEnabled(True)
        self.btn_save_html.setEnabled(True)
        self.btn_share.setEnabled(True)

        # 파일 처리 완료 후 자동 QLoRA 학습 트리거
        if self.auto_training_enabled and self.rag_store and res.get("segments"):
            self._trigger_auto_training(res["segments"])

        QMessageBox.information(self, "완료", "AI로 회의록을 생성했습니다.")

    def on_error(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "실패", f"처리 중 오류: {msg}")

    def on_copy(self):
        if not self._last_result: return
        self.md.selectAll(); self.md.copy()
        QMessageBox.information(self, "복사됨", "회의록 본문이 클립보드로 복사되었습니다.")

    def on_save_md(self):
        if not self._last_result: return
        from datetime import datetime
        fn = f"minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path, _ = QFileDialog.getSaveFileName(self, "Markdown 저장", fn, "Markdown (*.md)")
        if not path: return
        save_markdown(self._last_result["markdown"], path)
        QMessageBox.information(self, "저장됨", f"저장 위치:\n{path}")

    def on_save_html(self):
        if not self._last_result: return
        from datetime import datetime
        fn = f"minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path, _ = QFileDialog.getSaveFileName(self, "HTML 저장", fn, "HTML (*.html)")
        if not path: return
        from core.notes_export import save_html
        self._last_html_path = save_html(self._last_result["markdown"], path, title=self._last_result["title"])
        QMessageBox.information(self, "저장됨", f"저장 위치:\n{self._last_html_path}")

    def on_share(self):
        """
        로컬에서만 접근 가능한 공유 URL 제공 (간이 HTTP 서버)
        """
        if not self._last_result:
            return
        # HTML이 없으면 임시로 만든다
        from datetime import datetime
        if not self._last_html_path:
            out_dir = os.path.abspath("output/notes"); os.makedirs(out_dir, exist_ok=True)
            name = f"minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self._last_html_path = os.path.join(out_dir, name)
            save_html(self._last_result["markdown"], self._last_html_path, title=self._last_result["title"])

        base = os.path.dirname(self._last_html_path)
        url, port = start_share_server(base)
        file = os.path.basename(self._last_html_path)
        final = f"{url}/{file}"
        self.url_box.setVisible(True); self.url_box.setText(final); self.url_box.setCursorPosition(0)
        webbrowser.open(final)

    # ---------- QLoRA 자동 학습 ----------
    def _trigger_auto_training(self, segments: list):
        """
        파일 처리 완료 후 화자별 자동 학습 트리거 (순차 학습)

        Args:
            segments: 처리된 세그먼트 리스트
        """
        if not self.rag_store or not self.rag_store.ok:
            print("[WARN] RAG Store 없음 - 학습 불가")
            return

        # 세그먼트에서 화자 ID 추출
        speaker_ids = list(set(seg.get("speaker") for seg in segments if seg.get("speaker") and seg.get("speaker") != "Unknown"))

        if not speaker_ids:
            print("[WARN] 식별된 화자가 없음 - 학습 건너뜀")
            return

        print(f"[INFO] 파일 처리 완료: {len(speaker_ids)}명의 화자 발견, 순차 학습 시작")

        # 필터링: 발언 수 충분한 화자만 추출
        speakers_to_train = []
        for speaker_id in speaker_ids:
            try:
                # RAG에서 해당 화자의 발언 수 확인
                results = self.rag_store.search_by_speaker(speaker_id, query="", topk=1000)

                # 짧은 발언 필터링 (3단어 이상만 학습 대상)
                valid_utterances = [
                    utt for utt in results
                    if utt.get("text") and len(utt.get("text", "").strip().split()) >= 3
                ]
                utterance_count = len(valid_utterances)

                if utterance_count < self.min_utterances_for_training:
                    print(f"[INFO] {speaker_id}: 유효한 발언 수 부족 ({utterance_count}/{self.min_utterances_for_training}) - 학습 건너뜀")
                    print(f"       (전체: {len(results)}개, 필터링됨: {len(results) - utterance_count}개)")
                    continue

                # 화자 이름 가져오기
                speaker_name = speaker_id
                if self.speaker_manager:
                    speaker_name = self.speaker_manager.get_speaker_display_name(speaker_id)

                speakers_to_train.append((speaker_id, speaker_name, utterance_count))

            except Exception as e:
                print(f"[ERROR] {speaker_id} 학습 체크 실패: {e}")

        # 순차 학습: 한 명씩 완료 후 다음 사람 진행
        if speakers_to_train:
            print(f"[INFO] 총 {len(speakers_to_train)}명의 화자 순차 학습 시작")
            self._train_speakers_sequentially(speakers_to_train, index=0)

    def _train_speakers_sequentially(self, speakers_to_train: list, index: int):
        """
        화자들을 순차적으로 학습 (재귀함수)

        Args:
            speakers_to_train: [(speaker_id, speaker_name, utterance_count), ...] 리스트
            index: 현재 학습할 화자의 인덱스
        """
        if index >= len(speakers_to_train):
            # 모든 화자 학습 완료
            print(f"[INFO] ✅ 모든 화자 학습 완료!")
            return

        speaker_id, speaker_name, utterance_count = speakers_to_train[index]
        print(f"[INFO] 🔄 [{index + 1}/{len(speakers_to_train)}] {speaker_name} 학습 시작...")

        # 다음 화자 학습을 위한 콜백 등록
        def on_next_speaker():
            print(f"[INFO] ✅ {speaker_name} 학습 완료! 다음 화자 준비 중...")
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
        from core.persona_training_worker import PersonaTrainingWorker

        # 이미 학습 중인지 체크
        if speaker_id in self.training_workers:
            existing_worker = self.training_workers[speaker_id]
            if existing_worker.isRunning():
                print(f"[WARN] {speaker_name} 이미 학습 중")
                return

        # Worker 생성
        worker = PersonaTrainingWorker(
            rag_store=self.rag_store,
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

        # 학습 시작
        self.training_workers[speaker_id] = worker
        worker.start()

        print(f"[INFO] {speaker_name} QLoRA 학습 시작 (발언: {utterance_count}개)")

    def _start_training(self, speaker_id: str, speaker_name: str, utterance_count: int):
        """
        특정 화자의 QLoRA 학습 시작 (기본 버전)

        Args:
            speaker_id: 화자 ID
            speaker_name: 화자 이름
            utterance_count: 발언 수
        """
        from core.persona_training_worker import PersonaTrainingWorker

        # 이미 학습 중인지 체크
        if speaker_id in self.training_workers:
            existing_worker = self.training_workers[speaker_id]
            if existing_worker.isRunning():
                print(f"[WARN] {speaker_name} 이미 학습 중")
                return

        # Worker 생성
        worker = PersonaTrainingWorker(
            rag_store=self.rag_store,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            min_utterances=self.min_utterances_for_training,
            num_epochs=3,          # 원래 설정
            batch_size=4,          # 원래 설정
        )

        # 시그널 연결
        worker.sig_status.connect(self._on_training_status)
        worker.sig_progress.connect(self._on_training_progress)
        worker.sig_finished.connect(self._on_training_finished)
        worker.sig_error.connect(self._on_training_error)

        # 학습 시작
        self.training_workers[speaker_id] = worker
        worker.start()

        print(f"[INFO] {speaker_name} QLoRA 학습 시작 (발언: {utterance_count}개)")

    def _on_training_status(self, message: str):
        """학습 상태 메시지"""
        print(f"[TRAINING] {message}")

    def _on_training_progress(self, progress: int):
        """학습 진행률"""
        print(f"[TRAINING] Progress: {progress}%")

    def _on_training_finished(self, speaker_id: str, adapter_path: str):
        """학습 완료 처리"""
        speaker_name = speaker_id
        if self.speaker_manager:
            speaker_name = self.speaker_manager.get_speaker_display_name(speaker_id)

        print(f"[INFO] ✅ {speaker_name} 학습 완료!")
        print(f"[INFO]    어댑터: {adapter_path}")

        # DigitalPersona에 어댑터 경로 저장
        if self.persona_manager:
            try:
                persona = self.persona_manager.get_persona(speaker_id)
                if persona:
                    persona.qlora_adapter_path = adapter_path
                    self.persona_manager.save_persona(persona)
                    print(f"[INFO] 페르소나에 어댑터 경로 저장 완료")
            except Exception as e:
                print(f"[WARN] 어댑터 경로 저장 실패: {e}")

        # Worker 정리
        if speaker_id in self.training_workers:
            del self.training_workers[speaker_id]

    def _on_training_error(self, error_msg: str):
        """학습 에러 처리"""
        print(f"[ERROR] 학습 실패: {error_msg}")
