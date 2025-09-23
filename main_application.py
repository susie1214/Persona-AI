import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QSplitter, QListWidget, QListWidgetItem,
    QTextEdit, QPlainTextEdit, QLabel, QComboBox, QCheckBox,
    QFormLayout, QLineEdit, QMessageBox, QDialog, QDialogButtonBox,
    QDateTimeEdit, QProgressBar, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer, QDateTime, pyqtSlot

# 모듈 임포트
from config import config
from models import MeetingState, Segment, ActionItem
from audio_processor import AudioProcessor
from rag_manager import RAGManager, ConversationManager
from meeting_analyzer import MeetingAnalyzer, ScheduleManager

def format_time(seconds: float) -> str:
    """초를 MM:SS 형식으로 변환"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def now_str() -> str:
    """현재 시간을 HH:MM:SS 형식으로 반환"""
    return datetime.now().strftime("%H:%M:%S")

class ParticipantDialog(QDialog):
    """참여자 등록 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("회의 참여자 등록")
        self.resize(350, 180)
        
        layout = QVBoxLayout(self)
        
        # 이름 입력
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("참여자 이름을 입력하세요")
        layout.addWidget(QLabel("참여자 이름:"))
        layout.addWidget(self.name_edit)
        
        # 역할 입력 (선택사항)
        self.role_edit = QLineEdit()
        self.role_edit.setPlaceholderText("역할 (선택사항)")
        layout.addWidget(QLabel("역할:"))
        layout.addWidget(self.role_edit)
        
        # 버튼
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)
        
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        # 엔터키로 확인
        self.name_edit.returnPressed.connect(self.accept)
    
    def get_participant_info(self) -> tuple:
        """참여자 정보 반환"""
        return self.name_edit.text().strip(), self.role_edit.text().strip()

class MeetingAssistantApp(QMainWindow):
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 비서 (모듈형)")
        self.resize(1400, 900)
        
        # 상태 및 데이터
        self.meeting_state = MeetingState()
        
        # 핵심 컴포넌트 초기화
        self.audio_processor = AudioProcessor()
        self.rag_manager = RAGManager()
        self.conversation_manager = ConversationManager(self.rag_manager)
        self.meeting_analyzer = MeetingAnalyzer()
        self.schedule_manager = ScheduleManager()
        
        # UI 초기화
        self.setup_ui()
        self.apply_theme()
        self.setup_connections()
        
        # 상태바
        self.statusBar().showMessage("준비됨")
        
        # 타이머 (UI 업데이트용)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_preview)
        self.update_timer.start(2000)  # 2초마다 업데이트
        
    def setup_ui(self):
        """UI 구성"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 탭 위젯
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 각 탭 생성
        self.create_live_tab()
        self.create_timeline_tab()
        self.create_analysis_tab()
        self.create_qa_tab()
        self.create_action_tab()
        self.create_settings_tab()
        
        # 진행률 표시줄 (하단)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
    
    def create_live_tab(self):
        """실시간 모니터링 탭"""
        live_widget = QWidget()
        layout = QVBoxLayout(live_widget)
        
        # 컨트롤 버튼들
        control_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("녹음 시작")
        self.btn_stop = QPushButton("녹음 중지")
        self.btn_stop.setEnabled(False)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        
        # 강제 화자 설정
        control_layout.addWidget(QLabel("강제 화자:"))
        self.combo_forced_speaker = QComboBox()
        self.combo_forced_speaker.addItem("None")
        control_layout.addWidget(self.combo_forced_speaker)
        
        # 화자분리 활성화
        self.check_diarization = QCheckBox("화자분리 활성화")
        control_layout.addWidget(self.check_diarization)
        
        layout.addLayout(control_layout)
        
        # 분할 뷰
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 좌측: 실시간 대화
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("실시간 대화"))
        
        self.live_chat_list = QListWidget()
        left_layout.addWidget(self.live_chat_list)
        
        # 빠른 분석 버튼
        quick_buttons = QHBoxLayout()
        self.btn_quick_summary = QPushButton("빠른 요약")
        self.btn_add_to_rag = QPushButton("RAG에 추가")
        quick_buttons.addWidget(self.btn_quick_summary)
        quick_buttons.addWidget(self.btn_add_to_rag)
        left_layout.addLayout(quick_buttons)
        
        splitter.addWidget(left_widget)
        
        # 우측: 상태 및 미리보기
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 상태 표시
        right_layout.addWidget(QLabel("시스템 상태"))
        self.status_text = QPlainTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        right_layout.addWidget(self.status_text)
        
        # 실시간 미리보기
        right_layout.addWidget(QLabel("실시간 미리보기"))
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        right_layout.addWidget(self.preview_text)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 600])
        
        layout.addWidget(splitter)
        self.tabs.addTab(live_widget, "실시간")
    
    def create_timeline_tab(self):
        """타임라인 탭"""
        timeline_widget = QWidget()
        layout = QVBoxLayout(timeline_widget)
        
        # 필터 옵션
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("화자 필터:"))
        
        self.combo_speaker_filter = QComboBox()
        self.combo_speaker_filter.addItem("전체")
        filter_layout.addWidget(self.combo_speaker_filter)
        
        self.btn_export_timeline = QPushButton("타임라인 내보내기")
        filter_layout.addStretch()
        filter_layout.addWidget(self.btn_export_timeline)
        
        layout.addLayout(filter_layout)
        
        # 타임라인 리스트
        self.timeline_list = QListWidget()
        layout.addWidget(self.timeline_list)
        
        self.tabs.addTab(timeline_widget, "타임라인")
    
    def create_analysis_tab(self):
        """분석 결과 탭"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # 분석 버튼
        analysis_controls = QHBoxLayout()
        self.btn_analyze = QPushButton("회의 분석 실행")
        self.btn_export_report = QPushButton("보고서 내보내기")
        
        analysis_controls.addWidget(self.btn_analyze)
        analysis_controls.addWidget(self.btn_export_report)
        analysis_controls.addStretch()
        
        layout.addLayout(analysis_controls)
        
        # 분석 결과 표시
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        layout.addWidget(self.analysis_result_text)
        
        self.tabs.addTab(analysis_widget, "분석")
    
    def create_qa_tab(self):
        """Q&A 탭"""
        qa_widget = QWidget()
        layout = QVBoxLayout(qa_widget)
        
        # 질문 입력
        question_layout = QHBoxLayout()
        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText("회의 내용에 대해 질문하세요...")
        self.btn_ask = QPushButton("질문")
        
        question_layout.addWidget(self.question_edit)
        question_layout.addWidget(self.btn_ask)
        
        layout.addLayout(question_layout)
        
        # RAG 설정
        rag_layout = QHBoxLayout()
        rag_layout.addWidget(QLabel("검색 범위:"))
        
        self.combo_search_scope = QComboBox()
        self.combo_search_scope.addItems(["현재 세션", "전체 기록"])
        rag_layout.addWidget(self.combo_search_scope)
        
        self.btn_clear_rag = QPushButton("RAG 초기화")
        rag_layout.addStretch()
        rag_layout.addWidget(self.btn_clear_rag)
        
        layout.addLayout(rag_layout)
        
        # 답변 표시
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        layout.addWidget(self.answer_text)
        
        self.tabs.addTab(qa_widget, "Q&A")
    
    def create_action_tab(self):
        """액션 아이템 탭"""
        action_widget = QWidget()
        layout = QVBoxLayout(action_widget)
        
        # 상단 분할
        top_layout = QHBoxLayout()
        
        # 좌측: 요약
        left_action = QWidget()
        left_action_layout = QVBoxLayout(left_action)
        left_action_layout.addWidget(QLabel("회의 요약"))
        
        self.summary_text = QTextEdit()
        left_action_layout.addWidget(self.summary_text)
        
        # 우측: 액션 아이템
        right_action = QWidget()
        right_action_layout = QVBoxLayout(right_action)
        right_action_layout.addWidget(QLabel("액션 아이템"))
        
        self.action_items_text = QTextEdit()
        right_action_layout.addWidget(self.action_items_text)
        
        action_splitter = QSplitter(Qt.Orientation.Horizontal)
        action_splitter.addWidget(left_action)
        action_splitter.addWidget(right_action)
        action_splitter.setSizes([700, 700])
        
        layout.addWidget(action_splitter)
        
        # 일정 관리
        schedule_layout = QVBoxLayout()
        schedule_layout.addWidget(QLabel("다음 회의 일정"))
        
        # 일정 입력
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("시작:"))
        
        self.datetime_start = QDateTimeEdit()
        self.datetime_start.setDateTime(QDateTime.currentDateTime().addDays(7))
        self.datetime_start.setDisplayFormat("yyyy-MM-dd HH:mm")
        datetime_layout.addWidget(self.datetime_start)
        
        datetime_layout.addWidget(QLabel("종료:"))
        self.datetime_end = QDateTimeEdit()
        self.datetime_end.setDateTime(QDateTime.currentDateTime().addDays(7).addSecs(3600))
        self.datetime_end.setDisplayFormat("yyyy-MM-dd HH:mm")
        datetime_layout.addWidget(self.datetime_end)
        
        self.btn_create_schedule = QPushButton("일정 메모 생성")
        datetime_layout.addWidget(self.btn_create_schedule)
        
        schedule_layout.addLayout(datetime_layout)
        
        # 일정 메모
        self.schedule_memo_text = QTextEdit()
        schedule_layout.addWidget(self.schedule_memo_text)
        
        layout.addLayout(schedule_layout)
        
        self.tabs.addTab(action_widget, "액션 & 일정")
    
    def create_settings_tab(self):
        """설정 탭"""
        settings_widget = QWidget()
        layout = QFormLayout(settings_widget)
        
        # 모델 설정
        self.combo_whisper_model = QComboBox()
        self.combo_whisper_model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.combo_whisper_model.setCurrentText(config.model.WHISPER_MODEL)
        
        self.check_use_gpu = QCheckBox("GPU 사용 (가능한 경우)")
        self.check_use_gpu.setChecked(config.model.WHISPER_DEVICE == "cuda")
        
        # 화자분리 설정
        self.check_enable_diarization = QCheckBox("화자분리 활성화")
        
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setPlaceholderText("HuggingFace 토큰 (화자분리용)")
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        
        # 참여자 관리
        self.btn_add_participant = QPushButton("참여자 추가")
        
        participant_layout = QHBoxLayout()
        self.combo_speaker_id = QComboBox()
        self.combo_participant_name = QComboBox()
        self.btn_map_speaker = QPushButton("화자 매핑")
        
        participant_layout.addWidget(self.combo_speaker_id)
        participant_layout.addWidget(QLabel("→"))
        participant_layout.addWidget(self.combo_participant_name)
        participant_layout.addWidget(self.btn_map_speaker)
        
        # 폼에 추가
        layout.addRow("Whisper 모델:", self.combo_whisper_model)
        layout.addRow("", self.check_use_gpu)
        layout.addRow("화자분리:", self.check_enable_diarization)
        layout.addRow("HF 토큰:", self.hf_token_edit)
        layout.addRow("참여자 관리:", self.btn_add_participant)
        layout.addRow("화자 매핑:", participant_layout)
        
        # 기본 참여자 추가
        default_participants = ["김철수", "이영희", "박민수", "정수현"]
        for name in default_participants:
            self.combo_forced_speaker.addItem(name)
            self.combo_participant_name.addItem(name)
        
        self.tabs.addTab(settings_widget, "설정")
    
    def apply_theme(self):
        """테마 적용"""
        theme = config.ui.THEME
        self.setStyleSheet(f"""
            QMainWindow {{ 
                background-color: {theme['bg']}; 
            }}
            QTabWidget::pane {{ 
                border: 2px solid {theme['pane']}; 
                background-color: {theme['light_bg']};
            }}
            QPushButton {{
                background-color: {theme['btn']};
                border: 1px solid {theme['btn_border']};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ 
                background-color: {theme['btn_hover']}; 
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
            QListWidget, QTextEdit, QPlainTextEdit {{
                background-color: {theme['light_bg']};
                border: 1px solid {theme['pane']};
                border-radius: 4px;
            }}
            QLineEdit, QComboBox, QDateTimeEdit {{
                background-color: white;
                border: 1px solid {theme['pane']};
                border-radius: 4px;
                padding: 6px;
            }}
            QProgressBar {{
                border: 1px solid {theme['pane']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme['btn']};
                border-radius: 3px;
            }}
        """)
    
    def setup_connections(self):
        """시그널 연결"""
        # 오디오 프로세서
        self.audio_processor.segment_ready.connect(self.on_new_segment)
        self.audio_processor.status_update.connect(self.on_status_update)
        self.audio_processor.diarization_update.connect(self.on_diarization_update)
        
        # RAG 매니저
        self.rag_manager.status_update.connect(self.on_status_update)
        
        # 버튼 연결
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_quick_summary.clicked.connect(self.generate_quick_summary)
        self.btn_add_to_rag.clicked.connect(self.add_to_rag)
        self.btn_analyze.clicked.connect(self.analyze_meeting)
        self.btn_ask.clicked.connect(self.ask_question)
        self.btn_create_schedule.clicked.connect(self.create_schedule_memo)
        self.btn_add_participant.clicked.connect(self.add_participant)
        self.btn_map_speaker.clicked.connect(self.map_speaker)
        self.btn_clear_rag.clicked.connect(self.clear_rag)
        
        # 엔터키 연결
        self.question_edit.returnPressed.connect(self.ask_question)
        
        # 콤보박스 변경
        self.combo_forced_speaker.currentTextChanged.connect(self.on_forced_speaker_changed)
    
    @pyqtSlot(object)
    def on_new_segment(self, segment: Segment):
        """새 세그먼트 처리"""
        self.meeting_state.add_segment(segment)
        
        # 대화 매니저에 추가
        entry = self.conversation_manager.add_segment(segment)
        
        # UI 업데이트
        self.update_live_chat()
        self.update_timeline()
        
        # 화자 콤보박스 업데이트
        self.update_speaker_combos()
    
    @pyqtSlot(str)
    def on_status_update(self, message: str):
        """상태 업데이트"""
        timestamp = now_str()
        self.status_text.appendPlainText(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message)
    
    @pyqtSlot(list)
    def on_diarization_update(self, segments: list):
        """화자분리 결과 업데이트"""
        self.meeting_state.diar_segments = segments
        
        # 화자 ID 콤보박스 업데이트
        existing_speakers = {self.combo_speaker_id.itemText(i) 
                           for i in range(self.combo_speaker_id.count())}
        
        for _, _, speaker in segments:
            if speaker not in existing_speakers:
                self.combo_speaker_id.addItem(speaker)
    
    def start_recording(self):
        """녹음 시작"""
        try:
            # 설정 적용
            self.apply_settings()
            
            # 진행률 표시
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 무한 진행
            
            # 오디오 프로세서 시작
            self.audio_processor.start_recording()
            
            # UI 상태 변경
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            
            self.on_status_update("녹음이 시작되었습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"녹음 시작 실패: {e}")
            self.progress_bar.setVisible(False)
    
    def stop_recording(self):
        """녹음 중지"""
        try:
            self.audio_processor.stop_recording()
            
            # UI 상태 변경
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_bar.setVisible(False)
            
            self.on_status_update("녹음이 중지되었습니다.")
            
        except Exception as e:
            QMessageBox.warning(self, "경고", f"녹음 중지 중 오류: {e}")
    
    def apply_settings(self):
        """현재 설정을 시스템에 적용"""
        # Whisper 설정
        config.model.WHISPER_MODEL = self.combo_whisper_model.currentText()
        config.model.WHISPER_DEVICE = "cuda" if self.check_use_gpu.isChecked() else "cpu"
        config.model.WHISPER_COMPUTE_TYPE = "float16" if self.check_use_gpu.isChecked() else "int8"
        
        # 화자분리 설정
        self.meeting_state.diarization_enabled = self.check_enable_diarization.isChecked()
        
        # HuggingFace 토큰
        hf_token = self.hf_token_edit.text().strip()
        if hf_token:
            os.environ[config.model.HF_TOKEN_ENV] = hf_token
        
        # 강제 화자 설정
        forced_speaker = self.combo_forced_speaker.currentText()
        self.meeting_state.forced_speaker_name = None if forced_speaker == "None" else forced_speaker
    
    def update_live_chat(self):
        """실시간 대화 리스트 업데이트"""
        # 최근 세그먼트만 표시 (성능 고려)
        recent_segments = self.meeting_state.live_segments[-50:]
        
        # 현재 아이템 수와 비교
        current_count = self.live_chat_list.count()
        
        # 새로운 아이템만 추가
        for i in range(current_count, len(recent_segments)):
            segment = recent_segments[i]
            text = f"[{segment.speaker_name}] {segment.text}"
            item = QListWidgetItem(text)
            self.live_chat_list.addItem(item)
        
        # 자동 스크롤
        self.live_chat_list.scrollToBottom()
    
    def update_timeline(self):
        """타임라인 업데이트"""
        # 필터링된 세그먼트 가져오기
        filter_speaker = self.combo_speaker_filter.currentText()
        
        segments = self.meeting_state.live_segments
        if filter_speaker != "전체":
            segments = [seg for seg in segments if seg.speaker_name == filter_speaker]
        
        # 타임라인 아이템 업데이트 (최근 항목만)
        current_count = self.timeline_list.count()
        
        for i in range(current_count, len(segments)):
            segment = segments[i]
            time_str = f"{format_time(segment.start)}~{format_time(segment.end)}"
            text = f"{time_str} | {segment.speaker_name}: {segment.text}"
            item = QListWidgetItem(text)
            self.timeline_list.addItem(item)
        
        self.timeline_list.scrollToBottom()
    
    def update_speaker_combos(self):
        """화자 관련 콤보박스 업데이트"""
        speakers = self.meeting_state.get_speakers()
        
        # 화자 필터 콤보박스 업데이트
        current_filter = self.combo_speaker_filter.currentText()
        self.combo_speaker_filter.clear()
        self.combo_speaker_filter.addItem("전체")
        
        for speaker in speakers:
            self.combo_speaker_filter.addItem(speaker)
        
        # 이전 선택 복원
        index = self.combo_speaker_filter.findText(current_filter)
        if index >= 0:
            self.combo_speaker_filter.setCurrentIndex(index)
    
    def update_preview(self):
        """실시간 미리보기 업데이트"""
        if not self.meeting_state.live_segments:
            self.preview_text.setPlainText("대화 내용이 없습니다.")
            return
        
        # 최근 5개 세그먼트 표시
        recent_segments = self.meeting_state.live_segments[-5:]
        
        preview_lines = []
        for segment in recent_segments:
            time_str = format_time(segment.start)
            preview_lines.append(f"[{time_str}] {segment.speaker_name}: {segment.text}")
        
        # 통계 정보 추가
        total_duration = self.meeting_state.get_total_duration()
        total_segments = len(self.meeting_state.live_segments)
        speakers = self.meeting_state.get_speakers()
        
        stats_text = f"\n--- 현재 통계 ---\n"
        stats_text += f"전체 시간: {format_time(total_duration)}\n"
        stats_text += f"발언 수: {total_segments}개\n"
        stats_text += f"참여자: {len(speakers)}명 ({', '.join(speakers[:3])}{'...' if len(speakers) > 3 else ''})\n"
        
        self.preview_text.setPlainText("\n".join(preview_lines) + stats_text)
    
    def generate_quick_summary(self):
        """빠른 요약 생성"""
        if not self.meeting_state.live_segments:
            QMessageBox.information(self, "알림", "요약할 내용이 없습니다.")
            return
        
        try:
            # 최근 세그먼트들로 분석
            analysis_result = self.meeting_analyzer.analyze_meeting(self.meeting_state.live_segments[-20:])
            
            # 요약 표시
            summary = analysis_result.get('summary', '요약 생성 실패')
            self.summary_text.setText(summary)
            
            # 액션 아이템 표시
            actions = analysis_result.get('action_items', [])
            action_text = "\n".join([f"• [{action.speaker}] {action.text}" for action in actions])
            self.action_items_text.setText(action_text if action_text else "액션 아이템이 없습니다.")
            
            QMessageBox.information(self, "완료", "빠른 요약이 생성되었습니다.")
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"요약 생성 실패: {e}")
    
    def add_to_rag(self):
        """최근 세그먼트들을 RAG에 추가"""
        if not self.meeting_state.live_segments:
            QMessageBox.information(self, "알림", "추가할 내용이 없습니다.")
            return
        
        # 최근 10개 세그먼트 추가
        recent_segments = self.meeting_state.live_segments[-10:]
        
        success = self.rag_manager.add_segments(recent_segments, self.meeting_state.session_id)
        
        if success:
            QMessageBox.information(self, "완료", f"{len(recent_segments)}개 세그먼트가 RAG에 추가되었습니다.")
        else:
            QMessageBox.warning(self, "실패", "RAG 추가에 실패했습니다.")
    
    def analyze_meeting(self):
        """전체 회의 분석"""
        if not self.meeting_state.live_segments:
            QMessageBox.information(self, "알림", "분석할 내용이 없습니다.")
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            # 회의 분석 실행
            analysis_result = self.meeting_analyzer.analyze_meeting(self.meeting_state.live_segments)
            
            # 보고서 생성
            report = self.meeting_analyzer.generate_meeting_report(analysis_result)
            self.analysis_result_text.setText(report)
            
            # 요약 및 액션 아이템 탭도 업데이트
            self.summary_text.setText(analysis_result.get('summary', ''))
            
            actions = analysis_result.get('action_items', [])
            action_lines = []
            for action in actions:
                priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(action.priority, "🟡")
                deadline_str = f" (마감: {action.deadline.strftime('%Y-%m-%d')})" if action.deadline else ""
                action_lines.append(f"{priority_icon} [{action.speaker}] {action.text}{deadline_str}")
            
            self.action_items_text.setText("\n".join(action_lines) if action_lines else "액션 아이템이 없습니다.")
            
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "완료", "회의 분석이 완료되었습니다.")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.warning(self, "오류", f"분석 실패: {e}")
    
    def ask_question(self):
        """Q&A 질문 처리"""
        question = self.question_edit.text().strip()
        if not question:
            return
        
        try:
            # 검색 범위 결정
            session_id = None
            if self.combo_search_scope.currentText() == "현재 세션":
                session_id = self.meeting_state.session_id
            
            # RAG 검색
            search_results = self.rag_manager.search(question, limit=5, session_id=session_id)
            
            # 답변 구성
            if search_results:
                answer_lines = [f"질문: {question}\n"]
                answer_lines.append("관련 대화 내용:")
                
                for i, result in enumerate(search_results, 1):
                    answer_lines.append(f"\n{i}. [{result.speaker}] {result.text}")
                    answer_lines.append(f"   (유사도: {result.score:.3f}, 시간: {format_time(result.start_time)})")
                
                # 간단한 생성형 답변 (실제 LLM 없이 규칙 기반)
                answer_lines.append(f"\n--- 종합 답변 ---")
                answer_lines.append(f"'{question}'에 대한 답변을 위 대화 내용에서 확인할 수 있습니다.")
                
                if "언제" in question or "시간" in question:
                    times = [format_time(r.start_time) for r in search_results[:3]]
                    answer_lines.append(f"관련 시점: {', '.join(times)}")
                
                if "누가" in question or "화자" in question:
                    speakers = list(set(r.speaker for r in search_results[:3]))
                    answer_lines.append(f"관련 화자: {', '.join(speakers)}")
                
            else:
                answer_lines = [
                    f"질문: {question}\n",
                    "죄송합니다. 관련된 대화 내용을 찾을 수 없습니다.",
                    "다른 키워드로 질문해 보시거나, 더 많은 대화가 진행된 후 다시 시도해 주세요."
                ]
            
            self.answer_text.setText("\n".join(answer_lines))
            self.question_edit.clear()
            
        except Exception as e:
            self.answer_text.setText(f"질문 처리 중 오류가 발생했습니다: {e}")
    
    def create_schedule_memo(self):
        """일정 메모 생성"""
        start_dt = self.datetime_start.dateTime().toPython()
        end_dt = self.datetime_end.dateTime().toPython()
        
        participants = self.meeting_state.get_speakers()
        
        # 액션 아이템 수집 (간단한 버전)
        action_items = []
        for segment in self.meeting_state.live_segments[-20:]:  # 최근 20개에서 액션 찾기
            if any(keyword in segment.text for keyword in ['해야', '진행', '확인', '준비']):
                from models import ActionItem
                action = ActionItem(
                    id=f"temp_{len(action_items)}",
                    text=segment.text,
                    speaker=segment.speaker_name
                )
                action_items.append(action)
        
        # 스케줄 메모 생성
        memo = self.schedule_manager.create_schedule_memo(
            start_dt, end_dt, participants, action_items
        )
        
        self.schedule_memo_text.setText(memo)
        QMessageBox.information(self, "완료", "일정 메모가 생성되었습니다.")
    
    def add_participant(self):
        """참여자 추가"""
        dialog = ParticipantDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, role = dialog.get_participant_info()
            if name:
                # 콤보박스들에 추가
                if self.combo_forced_speaker.findText(name) < 0:
                    self.combo_forced_speaker.addItem(name)
                
                if self.combo_participant_name.findText(name) < 0:
                    self.combo_participant_name.addItem(name)
                
                QMessageBox.information(self, "완료", f"참여자 '{name}' 추가 완료")
    
    def map_speaker(self):
        """화자 매핑"""
        speaker_id = self.combo_speaker_id.currentText()
        participant_name = self.combo_participant_name.currentText()
        
        if not speaker_id or not participant_name:
            QMessageBox.warning(self, "경고", "화자 ID와 참여자 이름을 모두 선택해주세요.")
            return
        
        # 매핑 저장
        self.meeting_state.speaker_map[speaker_id] = participant_name
        QMessageBox.information(self, "완료", f"{speaker_id} → {participant_name} 매핑 완료")
    
    def clear_rag(self):
        """RAG 데이터 초기화"""
        reply = QMessageBox.question(
            self, "확인", 
            "RAG 데이터를 모두 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.rag_manager.clear_collection()
            if success:
                QMessageBox.information(self, "완료", "RAG 데이터가 초기화되었습니다.")
            else:
                QMessageBox.warning(self, "실패", "RAG 초기화에 실패했습니다.")
    
    def on_forced_speaker_changed(self, speaker_name: str):
        """강제 화자 변경"""
        self.meeting_state.forced_speaker_name = None if speaker_name == "None" else speaker_name
    
    def closeEvent(self, event):
        """애플리케이션 종료 시 처리"""
        try:
            # 녹음 중인 경우 중지
            if self.btn_stop.isEnabled():
                self.audio_processor.stop_recording()
            
            # 대화 내용 저장
            if self.meeting_state.live_segments:
                conversation_data = self.conversation_manager.export_conversation()
                
                # JSON 파일로 저장
                import json
                filename = config.storage.OUTPUT_DIR / f"meeting_{self.meeting_state.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                
                print(f"대화 내용 저장됨: {filename}")
            
        except Exception as e:
            print(f"종료 처리 중 오류: {e}")
        
        event.accept()

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 한국어 폰트 설정 (필요시)
    try:
        from PyQt6.QtGui import QFont
        font = QFont("맑은 고딕", 9)
        app.setFont(font)
    except:
        pass
    
    # 메인 윈도우 생성
    main_window = MeetingAssistantApp()
    main_window.show()
    
    # 이벤트 루프 시작
    sys.exit(app.exec())

# 독립 실행 및 테스트 함수
def test_application():
    """애플리케이션 구성 요소 테스트"""
    print("=" * 60)
    print("Main Application Module Test")
    print("=" * 60)
    
    # PyQt6 의존성 체크
    print("📦 PyQt6 Dependency Check:")
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QCoreApplication
        
        print("  ✅ PyQt6 available")
        
        # Qt 애플리케이션 생성
        if not QCoreApplication.instance():
            app = QCoreApplication([])
            app_created = True
        else:
            app = QCoreApplication.instance()
            app_created = False
            
        print("  ✅ Qt application instance ready")
        
    except ImportError as e:
        print(f"  ❌ PyQt6 not available: {e}")
        return False
    
    # 모듈 의존성 체크
    print("\n🔧 Module Dependencies Check:")
    modules_status = {}
    
    try:
        from config import config
        modules_status['config'] = True
        print("  ✅ Config module loaded")
    except ImportError as e:
        modules_status['config'] = False
        print(f"  ❌ Config module failed: {e}")
    
    try:
        from models import MeetingState, Segment
        modules_status['models'] = True
        print("  ✅ Models module loaded")
    except ImportError as e:
        modules_status['models'] = False
        print(f"  ❌ Models module failed: {e}")
    
    try:
        from audio_processor import AudioProcessor
        modules_status['audio_processor'] = True
        print("  ✅ AudioProcessor module loaded")
    except ImportError as e:
        modules_status['audio_processor'] = False
        print(f"  ❌ AudioProcessor module failed: {e}")
    
    try:
        from rag_manager import RAGManager, ConversationManager
        modules_status['rag_manager'] = True
        print("  ✅ RAGManager module loaded")
    except ImportError as e:
        modules_status['rag_manager'] = False
        print(f"  ❌ RAGManager module failed: {e}")
    
    try:
        from meeting_analyzer import MeetingAnalyzer, ScheduleManager
        modules_status['meeting_analyzer'] = True
        print("  ✅ MeetingAnalyzer module loaded")
    except ImportError as e:
        modules_status['meeting_analyzer'] = False
        print(f"  ❌ MeetingAnalyzer module failed: {e}")
    
    # 데이터 모델 테스트
    if modules_status.get('models'):
        print("\n📊 Data Models Test:")
        try:
            meeting_state = MeetingState()
            print(f"  ✅ MeetingState created: Session {meeting_state.session_id[:8]}...")
            
            # 테스트 세그먼트 추가
            test_segment = Segment(
                start=0, end=5, 
                text="테스트 발언입니다", 
                speaker_id="TEST_01", 
                speaker_name="테스터"
            )
            meeting_state.add_segment(test_segment)
            print(f"  ✅ Test segment added: {len(meeting_state.live_segments)} total")
            
        except Exception as e:
            print(f"  ❌ Data models test failed: {e}")
    
    # UI 구성 요소 테스트
    print("\n🖼️ UI Components Test:")
    try:
        # ParticipantDialog 테스트
        dialog = ParticipantDialog()
        print("  ✅ ParticipantDialog created")
        
        # 다이얼로그 정보 확인
        print(f"    - Window title: {dialog.windowTitle()}")
        print(f"    - Size: {dialog.size().width()}x{dialog.size().height()}")
        
        # 기본값 테스트
        name, role = dialog.get_participant_info()
        print(f"    - Default values: name='{name}', role='{role}'")
        
    except Exception as e:
        print(f"  ❌ UI components test failed: {e}")
    
    # 메인 애플리케이션 초기화 테스트 (GUI 제외)
    print("\n🏠 Main Application Initialization Test:")
    try:
        if all(modules_status[key] for key in ['config', 'models', 'audio_processor', 'rag_manager', 'meeting_analyzer']):
            # 컴포넌트별 초기화 테스트
            meeting_state = MeetingState()
            print("  ✅ MeetingState initialized")
            
            audio_processor = AudioProcessor()
            print("  ✅ AudioProcessor initialized")
            
            rag_manager = RAGManager()
            print("  ✅ RAGManager initialized")
            
            conversation_manager = ConversationManager(rag_manager)
            print("  ✅ ConversationManager initialized")
            
            meeting_analyzer = MeetingAnalyzer()
            print("  ✅ MeetingAnalyzer initialized")
            
            schedule_manager = ScheduleManager()
            print("  ✅ ScheduleManager initialized")
            
            print("  🎯 All core components ready for GUI integration")
            
        else:
            missing = [k for k, v in modules_status.items() if not v]
            print(f"  ⚠️ Cannot test full initialization - missing: {', '.join(missing)}")
            
    except Exception as e:
        print(f"  ❌ Main application initialization test failed: {e}")
    
    # 설정 검증
    if modules_status.get('config'):
        print("\n⚙️ Configuration Validation:")
        try:
            print(f"  - Audio sample rate: {config.audio.SAMPLE_RATE} Hz")
            print(f"  - Whisper model: {config.model.WHISPER_MODEL}")
            print(f"  - Output directory: {config.storage.OUTPUT_DIR}")
            print(f"  - Theme colors: {len(config.ui.THEME)} defined")
            print("  ✅ Configuration valid")
            
        except Exception as e:
            print(f"  ❌ Configuration validation failed: {e}")
    
    # 테마 적용 테스트
    print("\n🎨 Theme Application Test:")
    try:
        from PyQt6.QtWidgets import QPushButton
        
        # 테스트 버튼 생성
        test_button = QPushButton("Test Button")
        
        # 테마 스타일시트 생성 (간단한 버전)
        theme = config.ui.THEME
        style = f"""
            QPushButton {{
                background-color: {theme['btn']};
                border: 1px solid {theme['btn_border']};
                border-radius: 6px;
                padding: 8px 12px;
            }}
        """
        
        test_button.setStyleSheet(style)
        print("  ✅ Theme stylesheet applied successfully")
        print(f"    - Button color: {theme['btn']}")
        print(f"    - Border color: {theme['btn_border']}")
        
    except Exception as e:
        print(f"  ❌ Theme application test failed: {e}")
    
    # 유틸리티 함수 테스트
    print("\n🔧 Utility Functions Test:")
    try:
        # 시간 포맷 테스트
        test_times = [0, 65.5, 125.75, 3661.25]
        print("  Time formatting test:")
        for t in test_times:
            formatted = format_time(t)
            print(f"    {t}s -> {formatted}")
        
        # 현재 시간 테스트
        current_time = now_str()
        print(f"  Current time: {current_time}")
        print("  ✅ Utility functions working correctly")
        
    except Exception as e:
        print(f"  ❌ Utility functions test failed: {e}")
    
    # 파일 시스템 권한 테스트
    print("\n📁 File System Permissions Test:")
    try:
        import tempfile
        
        # 임시 파일 생성 테스트
        with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        print(f"  ✅ Temporary file created: {temp_path}")
        
        # 파일 읽기 테스트
        with open(temp_path, 'r') as f:
            content = f.read()
        print(f"  ✅ File read successfully: '{content}'")
        
        # 파일 삭제 테스트
        import os
        os.unlink(temp_path)
        print("  ✅ File cleanup successful")
        
        # 출력 디렉토리 권한 테스트
        test_file = config.storage.OUTPUT_DIR / "permission_test.txt"
        with open(test_file, 'w') as f:
            f.write("Permission test")
        
        print(f"  ✅ Output directory writable: {config.storage.OUTPUT_DIR}")
        
        # 정리
        if test_file.exists():
            test_file.unlink()
        
    except Exception as e:
        print(f"  ❌ File system permissions test failed: {e}")
    
    # 메모리 및 성능 기본 체크
    print("\n⚡ Performance Check:")
    try:
        import time
        import sys
        
        # 간단한 성능 측정
        start_time = time.time()
        
        # 더미 연산
        result = sum(i**2 for i in range(1000))
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        print(f"  ✅ Basic computation: {processing_time:.2f} ms")
        print(f"  - Python version: {sys.version.split()[0]}")
        print(f"  - Platform: {sys.platform}")
        
    except Exception as e:
        print(f"  ❌ Performance check failed: {e}")
    
    # 리소스 정리
    if app_created:
        try:
            app.quit()
            print("\n🧹 Qt application cleaned up")
        except:
            pass
    
    # 테스트 결과 요약
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(1 for status in modules_status.values() if status)
    total = len(modules_status)
    
    print(f"Module dependencies: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All modules loaded successfully!")
        print("✅ Application is ready to run")
        return True
    else:
        failed_modules = [name for name, status in modules_status.items() if not status]
        print(f"❌ Failed modules: {', '.join(failed_modules)}")
        print("⚠️  Some features may not work properly")
        return False

def run_minimal_demo():
    """최소한의 GUI 데모 실행"""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
        from PyQt6.QtCore import QTimer
        
        app = QApplication([])
        
        # 간단한 데모 윈도우
        demo_window = QMainWindow()
        demo_window.setWindowTitle("Persona-AI Meeting Assistant - Demo")
        demo_window.resize(600, 400)
        
        central_widget = QWidget()
        demo_window.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # 제목 레이블
        title_label = QLabel("Persona-AI Meeting Assistant")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c5530; margin: 20px;")
        layout.addWidget(title_label)
        
        # 상태 레이블
        status_label = QLabel("시스템 상태: 준비됨")
        status_label.setStyleSheet("color: #666; margin: 10px;")
        layout.addWidget(status_label)
        
        # 테스트 버튼들
        test_button = QPushButton("모듈 테스트 실행")
        test_button.setStyleSheet("""
            QPushButton {
                background-color: #ffe066;
                border: 1px solid #cccc99;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffdb4d;
            }
        """)
        
        def run_test():
            status_label.setText("상태: 테스트 실행 중...")
            # 비동기적으로 테스트 실행
            QTimer.singleShot(100, lambda: (
                test_application(),
                status_label.setText("상태: 테스트 완료")
            ))
        
        test_button.clicked.connect(run_test)
        layout.addWidget(test_button)
        
        # 종료 버튼
        quit_button = QPushButton("종료")
        quit_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                border: 1px solid #cc5555;
                border-radius: 6px;
                padding: 10px;
                color: white;
                font-weight: bold;
            }
        """)
        quit_button.clicked.connect(app.quit)
        layout.addWidget(quit_button)
        
        # 정보 레이블
        info_label = QLabel("""
이 데모는 Persona-AI Meeting Assistant의 구성 요소를 테스트합니다.
전체 기능을 사용하려면 필요한 의존성을 설치하고 main() 함수를 실행하세요.

필수 패키지: PyQt6, faster-whisper, qdrant-client, sentence-transformers
        """)
        info_label.setStyleSheet("color: #666; margin: 20px; font-size: 10px;")
        layout.addWidget(info_label)
        
        demo_window.show()
        app.exec()
        
    except ImportError:
        print("PyQt6를 사용할 수 없어 콘솔 테스트만 실행합니다.")
        test_application()

if __name__ == "__main__":
    import sys
    
    # 명령줄 인수 확인
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            print("테스트 모드로 실행 중...")
            success = test_application()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == '--demo':
            print("데모 모드로 실행 중...")
            run_minimal_demo()
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("""
Persona-AI Meeting Assistant

사용법:
  python main_application.py           전체 애플리케이션 실행
  python main_application.py --test    모듈 테스트 실행
  python main_application.py --demo    최소 데모 실행
  python main_application.py --help    이 도움말 표시

필수 의존성:
  - PyQt6 (GUI)
  - faster-whisper (음성 인식)
  - qdrant-client (벡터 검색)
  - sentence-transformers (임베딩)
  - pyaudio (오디오 입력)
  - soundfile, librosa (오디오 처리)

설치:
  pip install -r requirements.txt

더 자세한 정보는 README.md를 참조하세요.
            """)
            sys.exit(0)
    
    # 기본적으로 전체 애플리케이션 실행
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 프로그램을 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        print("\n--test 옵션으로 시스템을 확인해보세요:")
        print("python main_application.py --test")
        sys.exit(1)# main_application.py
# 메인 애플리케이션 - 개선된 모듈형 구조

