# gui.py
import sys
from typing import Dict
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, 
    QPushButton, QLabel, QComboBox, QSplitter, QMessageBox, QFrame,
    QTabWidget, QTextEdit, QCheckBox, QFileDialog, QProgressBar,
    QGroupBox
)

from config import Config
from diarization import RealTimeDiarization
from summary import SummaryService
from file_processor import FileProcessor

THEME = {
    "bg": "#e6f5e6",
    "pane": "#99cc99",
    "light_bg": "#fafffa",
    "btn": "#ffe066",
    "btn_hover": "#ffdb4d",
    "btn_border": "#cccc99",
}

class FileProcessingThread(QThread):
    """파일 처리를 위한 별도 스레드"""
    progress_updated = pyqtSignal(int)
    transcription_received = pyqtSignal(str, str, str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    processing_complete = pyqtSignal()
    
    def __init__(self, file_processor: FileProcessor, file_path: str):
        super().__init__()
        self.file_processor = file_processor
        self.file_path = file_path
        
        # 콜백 연결
        self.file_processor.on_progress = self.progress_updated.emit
        self.file_processor.on_transcription = self.transcription_received.emit
        self.file_processor.on_status_change = self.status_changed.emit
        self.file_processor.on_error = self.error_occurred.emit
        self.file_processor.on_processing_complete = self.processing_complete.emit
    
    def run(self):
        """스레드 실행"""
        self.file_processor.process_file(self.file_path)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona AI Assiatant")
        self.resize(1200, 900)

        # 설정 및 서비스 초기화
        self.config = Config()
        self.rt = RealTimeDiarization(self.config)
        self.summary_service = SummaryService(self.config)
        self.file_processor = FileProcessor(self.config)
        
        # 파일 처리 관련
        self.processing_thread = None

        # GUI 구성요소 초기화
        self.init_ui()
        self.init_connections()
        self.populate_devices()
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['bg']};
                color: black;
            }}
            QTabWidget::pane {{
                background: {THEME['pane']};
            }}
            QTextEdit {{
                background: {THEME['light_bg']};
                border: 1px solid {THEME['btn_border']};
                color: black;
            }}
            QPushButton {{
                background-color: {THEME['btn']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 5px;
                padding: 5px;
                color: black;
            }}
            QPushButton:hover {{
                background-color: {THEME['btn_hover']};
            }}
            QLabel {{
                background-color: transparent;
                color: black;
            }}
            QTabWidget, QComboBox, QCheckBox, QGroupBox {{
                color: black;
            }}
            QCheckBox {{
                border: 0.5px solid black;
            }}
        """) 
        #    QVBoxLayout, QHBoxLayout, QTextBrowser, 
        #    QPushButton, QLabel, QComboBox, QSplitter, QMessageBox, QFrame,
        #    QTabWidget, QTextEdit, QCheckBox, QFileDialog, QProgressBar,
        #    QGroupBox

    def init_ui(self):
        """UI 초기화"""
        # 상단 컨트롤 패널
        self.create_control_panel()
        
        # 메인 컨텐츠 영역 (탭으로 분리)
        self.create_main_tabs()
        
        # 전체 레이아웃 구성
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(self.control_layout)
        main_layout.addWidget(self.main_tabs, stretch=1)

    def create_main_tabs(self):
        """메인 탭 위젯 생성"""
        self.main_tabs = QTabWidget()
        
        # 실시간 처리 탭
        self.create_realtime_tab()
        
        # 파일 처리 탭
        self.create_file_processing_tab()
        
        # 요약 탭
        self.create_summary_tabs()

    def create_realtime_tab(self):
        """실시간 처리 탭 생성"""
        realtime_widget = QWidget()
        layout = QVBoxLayout(realtime_widget)
        
        # 실시간 전사 뷰
        self.chat_view = QTextBrowser()
        self.chat_view.setFrameShape(QFrame.Shape.StyledPanel)
        self.chat_view.setOpenExternalLinks(False)
        layout.addWidget(self.chat_view)
        
        self.main_tabs.addTab(realtime_widget, "실시간 처리")

    def create_file_processing_tab(self):
        """파일 처리 탭 생성"""
        file_widget = QWidget()
        layout = QVBoxLayout(file_widget)
        
        # 파일 선택 그룹
        file_group = QGroupBox("파일 처리")
        file_group_layout = QVBoxLayout(file_group)
        
        # 파일 선택 영역
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("선택된 파일: 없음")
        self.file_select_button = QPushButton("파일 선택")
        self.file_process_button = QPushButton("처리 시작")
        self.file_export_button = QPushButton("결과 내보내기")
        
        file_select_layout.addWidget(self.file_path_label, stretch=1)
        file_select_layout.addWidget(self.file_select_button)
        file_select_layout.addWidget(self.file_process_button)
        file_select_layout.addWidget(self.file_export_button)
        
        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 지원 형식 안내
        formats = self.file_processor.get_supported_formats()
        audio_formats = ", ".join(formats)
        
        format_info = QLabel(
            f"<b>지원 형식:</b><br>"
            f"• 오디오: {audio_formats}"
        )
        format_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        
        file_group_layout.addLayout(file_select_layout)
        file_group_layout.addWidget(self.progress_bar)
        file_group_layout.addWidget(format_info)
        
        # 파일 처리 결과 뷰
        self.file_result_view = QTextBrowser()
        self.file_result_view.setFrameShape(QFrame.Shape.StyledPanel)
        self.file_result_view.setOpenExternalLinks(False)
        
        layout.addWidget(file_group)
        layout.addWidget(self.file_result_view, stretch=1)
        
        self.main_tabs.addTab(file_widget, "파일 처리")

    def create_control_panel(self):
        """컨트롤 패널 생성"""
        self.control_layout = QHBoxLayout()
        
        # 실시간 처리 관련 컨트롤
        realtime_group = QGroupBox("실시간 처리")
        realtime_layout = QHBoxLayout(realtime_group)
        
        # 디바이스 선택
        realtime_layout.addWidget(QLabel("디바이스:"))
        self.device_combo = QComboBox()
        realtime_layout.addWidget(self.device_combo)
        
        # 제어 버튼들
        self.start_button = QPushButton("시작")
        self.stop_button = QPushButton("중지")
        
        realtime_layout.addWidget(self.start_button)
        realtime_layout.addWidget(self.stop_button)
        
        # 공통 컨트롤
        self.clear_button = QPushButton("내용 지우기")
        
        # 자동 요약 체크박스
        self.auto_summary_checkbox = QCheckBox("자동 요약")
        self.auto_summary_checkbox.setChecked(False)
        
        self.control_layout.addWidget(realtime_group)
        self.control_layout.addWidget(self.clear_button)
        self.control_layout.addWidget(self.auto_summary_checkbox)
        self.control_layout.addStretch()
        
        # 상태 표시
        self.status_label = QLabel("상태: 준비됨")
        self.control_layout.addWidget(self.status_label)

    def create_summary_tabs(self):
        """요약 탭들 생성"""
        # 요약 탭 위젯
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        self.summary_tabs = QTabWidget()
        
        # 전체 회의 요약 탭
        self.meeting_summary_view = QTextBrowser()
        self.meeting_summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        
        meeting_tab_widget = QWidget()
        meeting_layout = QVBoxLayout(meeting_tab_widget)
        
        meeting_button_layout = QHBoxLayout()
        self.meeting_summary_button = QPushButton("전체 회의 요약")
        meeting_button_layout.addWidget(self.meeting_summary_button)
        meeting_button_layout.addStretch()
        
        meeting_layout.addLayout(meeting_button_layout)
        meeting_layout.addWidget(self.meeting_summary_view)
        
        self.summary_tabs.addTab(meeting_tab_widget, "회의 요약")
        
        # 화자별 요약 탭
        self.speaker_summary_view = QTextBrowser()
        self.speaker_summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        
        speaker_tab_widget = QWidget()
        speaker_layout = QVBoxLayout(speaker_tab_widget)
        
        speaker_button_layout = QHBoxLayout()
        self.speaker_summary_button = QPushButton("화자별 요약 생성")
        speaker_button_layout.addWidget(self.speaker_summary_button)
        speaker_button_layout.addStretch()
        
        speaker_layout.addLayout(speaker_button_layout)
        speaker_layout.addWidget(self.speaker_summary_view)
        
        self.summary_tabs.addTab(speaker_tab_widget, "화자별 요약")
                
        # 화자별 요약 탭
        # summary_layout.addWidget(self.summary_tabs) 
        # self.main_tabs.addTab(summary_widget, "요약")
        # self.speaker_summary_button = QPushButton("화자별 요약 생성")
        # speaker_button_layout.addWidget(self.speaker_summary_button)
        # speaker_button_layout.addStretch()
        
        # speaker_layout.addLayout(speaker_button_layout)
        # speaker_layout.addWidget(self.speaker_summary_view)
        
        # self.summary_tabs.addTab(speaker_tab_widget, "화자별 요약")
        
        # 사용자 정의 요약 탭
        custom_tab_widget = QWidget()
        custom_layout = QVBoxLayout(custom_tab_widget)
        
        custom_button_layout = QHBoxLayout()
        self.custom_summary_button = QPushButton("사용자 정의 요약")
        custom_button_layout.addWidget(self.custom_summary_button)
        custom_button_layout.addStretch()
        
        self.custom_prompt_edit = QTextEdit()
        self.custom_prompt_edit.setMaximumHeight(80)
        self.custom_prompt_edit.setPlaceholderText("요약 프롬프트를 입력하세요...")
        self.custom_prompt_edit.setText("다음 대화 내용에서 핵심 키워드와 주요 결정사항을 정리해주세요.")
        
        self.custom_summary_view = QTextBrowser()
        self.custom_summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        
        custom_layout.addLayout(custom_button_layout)
        custom_layout.addWidget(QLabel("사용자 정의 프롬프트:"))
        custom_layout.addWidget(self.custom_prompt_edit)
        custom_layout.addWidget(self.custom_summary_view)
        
        self.summary_tabs.addTab(custom_tab_widget, "사용자 정의")
        
        summary_layout.addWidget(self.summary_tabs)
        
        self.main_tabs.addTab(summary_widget, "요약")
        
        # # 전체 회의 요약 탭
        # self.meeting_summary_view = QTextBrowser()
        # self.meeting_summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        
        # meeting_tab_widget = QWidget()
        # meeting_layout = QVBoxLayout(meeting_tab_widget)
        
        # meeting_button_layout = QHBoxLayout()
        # self.meeting_summary_button = QPushButton("전체 회의 요약")
        # meeting_button_layout.addWidget(self.meeting_summary_button)
        # meeting_button_layout.addStretch()
        
        # meeting_layout.addLayout(meeting_button_layout)
        # meeting_layout.addWidget(self.meeting_summary_view)
        
        # self.summary_tabs.addTab(meeting_tab_widget, "회의 요약")
        
        # # 사용자 정의 요약 탭
        # custom_tab_widget = QWidget()
        # custom_layout = QVBoxLayout(custom_tab_widget)
        
        # custom_button_layout = QHBoxLayout()
        # self.custom_summary_button = QPushButton("사용자 정의 요약")
        # custom_button_layout.addWidget(self.custom_summary_button)
        # custom_button_layout.addStretch()
        
        # self.custom_prompt_edit = QTextEdit()
        # self.custom_prompt_edit.setMaximumHeight(80)
        # self.custom_prompt_edit.setPlaceholderText("요약 프롬프트를 입력하세요...")
        # self.custom_prompt_edit.setText("다음 대화 내용에서 핵심 키워드와 주요 결정사항을 정리해주세요.")
        
        # self.custom_summary_view = QTextBrowser()
        # self.custom_summary_view.setFrameShape(QFrame.Shape.StyledPanel)
        
        # custom_layout.addLayout(custom_button_layout)
        # custom_layout.addWidget(QLabel("사용자 정의 프롬프트:"))
        # custom_layout.addWidget(self.custom_prompt_edit)
        # custom_layout.addWidget(self.custom_summary_view)
        
        # self.summary_tabs.addTab(custom_tab_widget, "사용자 정의")

    def init_connections(self):
        """시그널 연결"""
        # 실시간 처리 버튼 연결
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        self.clear_button.clicked.connect(self.on_clear)
        
        # 파일 처리 버튼 연결
        self.file_select_button.clicked.connect(self.on_file_select)
        self.file_process_button.clicked.connect(self.on_file_process)
        self.file_export_button.clicked.connect(self.on_file_export)
        
        # 요약 버튼 연결
        self.speaker_summary_button.clicked.connect(self.on_speaker_summary)
        self.meeting_summary_button.clicked.connect(self.on_meeting_summary)
        self.custom_summary_button.clicked.connect(self.on_custom_summary)
        
        # 실시간 화자분리 콜백 연결
        self.rt.on_transcription = self.on_transcription
        self.rt.on_status_change = self.on_status
        self.rt.on_error = self.on_error

    # 파일 처리 이벤트 핸들러들
    def on_file_select(self):
        """파일 선택"""
        formats = self.file_processor.get_supported_formats()
        
        # 파일 필터 생성
        filter_parts = []
        filter_parts.append("지원되는 오디오 파일 (*" + " *".join(formats) + ")")
        for fmt in formats:
            filter_parts.append(f"{fmt.upper()} 파일 (*{fmt})")
        filter_parts.append("모든 파일 (*.*)")
        
        file_filter = ";;".join(filter_parts)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "처리할 오디오 파일 선택", "", file_filter
        )
        
        if file_path:
            self.selected_file_path = file_path
            file_name = Path(file_path).name
            self.file_path_label.setText(f"선택된 파일: {file_name}")
            self.file_process_button.setEnabled(True)
        else:
            self.selected_file_path = None
            self.file_path_label.setText("선택된 파일: 없음")
            self.file_process_button.setEnabled(False)

    def on_file_process(self):
        """파일 처리 시작"""
        if not hasattr(self, 'selected_file_path') or not self.selected_file_path:
            QMessageBox.warning(self, "경고", "처리할 파일을 선택해주세요.")
            return
        
        # 모델 로딩 확인
        if not self.rt.models_loaded:
            self.status_label.setText("상태: 모델 로딩 중...")
            loaded = self.rt.load_models()
            if not loaded:
                QMessageBox.critical(self, "오류", "모델 로딩 실패")
                return
        
        # 파일 프로세서에 모델 전달
        if not self.file_processor.load_models_from_diarization(self.rt):
            QMessageBox.critical(self, "오류", "파일 프로세서 모델 로딩 실패")
            return
        
        # UI 상태 변경
        self.file_process_button.setEnabled(False)
        self.file_select_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.file_result_view.clear()
        
        # 파일 처리 스레드 시작
        print(f"[DEBUG] gui - selected file path : {self.selected_file_path}")
        self.processing_thread = FileProcessingThread(self.file_processor, self.selected_file_path)
        self.processing_thread.progress_updated.connect(self.on_file_progress)
        self.processing_thread.transcription_received.connect(self.on_file_transcription)
        self.processing_thread.status_changed.connect(self.on_status)
        self.processing_thread.error_occurred.connect(self.on_error)
        self.processing_thread.processing_complete.connect(self.on_file_processing_complete)
        self.processing_thread.start()

    def on_file_progress(self, progress: int):
        """파일 처리 진행률 업데이트"""
        self.progress_bar.setValue(progress)

    def on_file_transcription(self, timestamp: str, speaker: str, text: str):
        """파일 처리 전사 결과 수신"""
        self.file_result_view.append(f"[{timestamp}] <b>{speaker}</b>: {text}")
        self.file_result_view.verticalScrollBar().setValue(
            self.file_result_view.verticalScrollBar().maximum()
        )

    def on_file_processing_complete(self):
        """파일 처리 완료"""
        self.file_process_button.setEnabled(True)
        self.file_select_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.file_export_button.setEnabled(True)
        
        # 결과 통계 표시
        results = self.file_processor.get_processing_results()
        self.file_result_view.append(
            f"\n<b>처리 완료!</b><br>"
            f"총 화자 수: {results['total_speakers']}<br>"
            f"총 발화 수: {results['total_segments']}"
        )

    def on_file_export(self):
        """파일 처리 결과 내보내기"""
        if not hasattr(self, 'file_processor') or not self.file_processor.transcription_results:
            QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "결과 내보내기", "", 
            "텍스트 파일 (*.txt);;JSON 파일 (*.json)"
        )
        
        if file_path:
            format_type = 'json' if selected_filter.startswith('JSON') else 'txt'
            success = self.file_processor.export_results_to_file(file_path, format_type)
            
            if success:
                QMessageBox.information(self, "완료", f"결과가 성공적으로 저장되었습니다:\n{file_path}")
            else:
                QMessageBox.critical(self, "오류", "파일 저장에 실패했습니다.")

    def populate_devices(self):
        """오디오 디바이스 목록 채우기"""
        devices = self.rt.get_available_audio_devices()
        self.device_combo.clear()
        for idx, name in devices.items():
            self.device_combo.addItem(f"{name} ({idx})", idx)

    # 실시간 처리 이벤트 핸들러들
    def on_transcription(self, timestamp: str, speaker: str, text: str):
        """전사 결과 수신"""
        self.chat_view.append(f"[{timestamp}] <b>{speaker}</b>: {text}")
        self.chat_view.verticalScrollBar().setValue(
            self.chat_view.verticalScrollBar().maximum()
        )
        
        # 자동 요약이 활성화된 경우 주기적으로 요약 업데이트
        if self.auto_summary_checkbox.isChecked():
            # 여기서는 간단히 구현하지만, 실제로는 더 정교한 로직이 필요
            pass

    def on_status(self, message: str):
        """상태 변경"""
        self.status_label.setText(f"상태: {message}")

    def on_error(self, message: str):
        """오류 발생"""
        self.chat_view.append(f"<span style='color:red'>[ERROR] {message}</span>")

    def on_start(self):
        """녹음 시작"""
        device_index = self.device_combo.currentData()
        if not self.rt.models_loaded:
            self.status_label.setText("상태: 모델 로딩 중...")
            loaded = self.rt.load_models()
            if not loaded:
                QMessageBox.critical(self, "오류", "모델 로딩 실패")
                return
        
        success = self.rt.start_recording(device_index)
        if success:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def on_stop(self):
        """녹음 중지"""
        self.rt.stop_recording()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_clear(self):
        """내용 지우기"""
        reply = QMessageBox.question(
            self, "확인", "모든 내용을 지우시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 실시간 처리 결과 지우기
            self.chat_view.clear()
            self.rt.clear_speaker_texts()
            
            # 파일 처리 결과 지우기
            self.file_result_view.clear()
            if hasattr(self, 'file_processor'):
                self.file_processor.transcription_results.clear()
                self.file_processor.speaker_texts.clear()
            
            # 요약 결과 지우기
            self.speaker_summary_view.clear()
            self.meeting_summary_view.clear()
            self.custom_summary_view.clear()

    # 요약 이벤트 핸들러들 (실시간/파일 처리 결과 모두 지원)
    def get_current_speaker_texts(self) -> Dict:
        """현재 활성 탭에 따른 화자 텍스트 반환"""
        current_tab = self.main_tabs.currentIndex()
        
        # if current_tab == 0:  # 실시간 처리 탭
        #     return self.rt.get_speaker_texts()
        # elif current_tab == 1:  # 파일 처리 탭
        #     if hasattr(self, 'file_processor'):
        #         return self.file_processor.speaker_texts
        
        # return {}
        
        return self.rt.get_speaker_texts()

    def on_speaker_summary(self):
        """화자별 요약 생성"""
        self.speaker_summary_view.clear()
        speaker_texts = self.get_current_speaker_texts()
        
        if not speaker_texts:
            self.speaker_summary_view.append("<i>아직 전사된 데이터가 없습니다.</i>")
            return

        if not self.summary_service.is_available():
            self.speaker_summary_view.append("<i>OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다.</i>")
            return

        self.speaker_summary_view.append("<b>화자별 요약을 생성 중...</b><br>")
        QApplication.processEvents()  # UI 업데이트
        
        summaries = self.summary_service.summarize_all_speakers(speaker_texts)
        
        self.speaker_summary_view.clear()
        for speaker, summary in summaries.items():
            self.speaker_summary_view.append(f"<b>{speaker} 요약:</b><br>{summary}<br><br>")

    def on_meeting_summary(self):
        """전체 회의 요약 생성"""
        self.meeting_summary_view.clear()
        speaker_texts = self.get_current_speaker_texts()
        
        if not speaker_texts:
            self.meeting_summary_view.append("<i>아직 전사된 데이터가 없습니다.</i>")
            return

        if not self.summary_service.is_available():
            self.meeting_summary_view.append("<i>OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다.</i>")
            return

        self.meeting_summary_view.append("<b>전체 회의 요약을 생성 중...</b>")
        QApplication.processEvents()  # UI 업데이트
        
        meeting_summary = self.summary_service.create_meeting_summary(speaker_texts)
        
        self.meeting_summary_view.clear()
        self.meeting_summary_view.append(f"<b>회의 요약:</b><br><br>{meeting_summary}")

    def on_custom_summary(self):
        """사용자 정의 요약 생성"""
        self.custom_summary_view.clear()
        speaker_texts = self.get_current_speaker_texts()
        
        if not speaker_texts:
            self.custom_summary_view.append("<i>아직 전사된 데이터가 없습니다.</i>")
            return

        if not self.summary_service.is_available():
            self.custom_summary_view.append("<i>OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다.</i>")
            return

        custom_prompt = self.custom_prompt_edit.toPlainText().strip()
        if not custom_prompt:
            QMessageBox.warning(self, "경고", "사용자 정의 프롬프트를 입력해주세요.")
            return

        # 모든 화자의 텍스트를 합치기
        all_texts = []
        for speaker, texts in speaker_texts.items():
            for text in texts:
                all_texts.append(f"{speaker}: {text}")
        
        combined_text = "\n".join(all_texts)
        
        self.custom_summary_view.append("<b>사용자 정의 요약을 생성 중...</b>")
        QApplication.processEvents()  # UI 업데이트
        
        custom_summary = self.summary_service.create_custom_summary(combined_text, custom_prompt)
        
        self.custom_summary_view.clear()
        self.custom_summary_view.append(f"<b>프롬프트:</b> {custom_prompt}<br><br>")
        self.custom_summary_view.append(f"<b>결과:</b><br>{custom_summary}")

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())