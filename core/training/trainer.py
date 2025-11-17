# -*- coding: utf-8 -*-
# core/training/trainer.py
"""
QLoRA 페르소나 학습을 백그라운드에서 수행하는 Worker
회의 종료 시 자동으로 화자별 학습을 진행
"""

import os
import traceback
from typing import Optional, Dict
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QLabel

from .dataset import PersonaDatasetGenerator
from core.rag import RagStore


class PersonaTrainingWorker(QThread):
    """
    백그라운드 QLoRA 학습 Worker

    Signals:
        sig_status: 상태 메시지 (str)
        sig_progress: 진행률 (int, 0-100)
        sig_finished: 학습 완료 (speaker_id: str, adapter_path: str)
        sig_error: 에러 발생 (error_msg: str)
    """

    sig_status = Signal(str)
    sig_progress = Signal(int)  # 0-100
    sig_finished = Signal(str, str)  # (speaker_id, adapter_path)
    sig_error = Signal(str)

    def __init__(
        self,
        rag_store: RagStore,
        speaker_id: str,
        speaker_name: Optional[str] = None,
        min_utterances: int = 20,
        num_epochs: int = 1,
        batch_size: int = 2,
        base_model: str = "models/kanana-1.5-2.1b-instruct"
    ):
        super().__init__()
        self.rag_store = rag_store
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name or speaker_id
        self.min_utterances = min_utterances
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_model = base_model

        self._is_running = True

    def run(self):
        """학습 프로세스 실행"""
        try:
            # 1. 데이터셋 생성 (0-30%)
            self.sig_status.emit(f"📊 {self.speaker_name} 데이터셋 생성 중...")
            self.sig_progress.emit(5)

            dataset_path = self._generate_dataset()
            if not dataset_path:
                self.sig_error.emit(f"데이터셋 생성 실패: 발언 수 부족 (최소 {self.min_utterances}개 필요)")
                return

            self.sig_progress.emit(30)

            # 2. QLoRA 학습 (30-90%)
            # 실제 학습은 train_persona.py의 PersonaTrainer 사용
            # 하지만 PEFT가 없을 수 있으므로 먼저 체크
            self.sig_status.emit(f"🧠 {self.speaker_name} 말투 학습 중...")

            adapter_path = self._train_qlora(dataset_path)
            if not adapter_path:
                self.sig_error.emit("QLoRA 학습 실패: PEFT 라이브러리 미설치 또는 GPU 메모리 부족")
                return

            self.sig_progress.emit(90)

            # 3. 완료 (90-100%)
            self.sig_status.emit(f"✅ {self.speaker_name} 학습 완료!")
            self.sig_progress.emit(100)
            self.sig_finished.emit(self.speaker_id, adapter_path)

        except Exception as e:
            error_msg = f"학습 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            self.sig_error.emit(error_msg)

    def _generate_dataset(self) -> Optional[str]:
        """데이터셋 생성"""
        try:
            generator = PersonaDatasetGenerator(output_dir="data/persona_datasets")
            dataset_path = generator.generate_dataset_from_rag(
                rag_store=self.rag_store,
                speaker_id=self.speaker_id,
                speaker_name=self.speaker_name,
                min_utterances=self.min_utterances
            )
            return dataset_path
        except Exception as e:
            print(f"[ERROR] Dataset generation failed: {e}")
            return None

    def _train_qlora(self, dataset_path: str) -> Optional[str]:
        """QLoRA 학습 실행"""
        try:
            # PEFT 사용 가능 여부 체크
            try:
                from train_persona import PersonaTrainer, PersonaTrainingConfig, TRAIN_AVAILABLE
            except ImportError:
                print("[ERROR] train_persona module not found")
                return None

            if not TRAIN_AVAILABLE:
                print("[ERROR] PEFT not available - QLoRA training requires peft library")
                return None

            # 학습 설정
            config = PersonaTrainingConfig(
                base_model=self.base_model,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                output_dir="adapters",
                use_4bit=True,
                use_fp16=True,
            )

            # 학습 실행
            trainer = PersonaTrainer(config)

            # 진행률 업데이트를 위한 콜백 (간단한 시뮬레이션)
            # 실제로는 Trainer의 callback을 사용해야 하지만, 여기서는 간소화
            self.sig_progress.emit(40)

            adapter_path = trainer.train(
                dataset_path=dataset_path,
                speaker_id=self.speaker_id,
                speaker_name=self.speaker_name
            )

            return adapter_path

        except Exception as e:
            print(f"[ERROR] QLoRA training failed: {e}")
            traceback.print_exc()
            return None

    def stop(self):
        """학습 중단"""
        self._is_running = False
        self.quit()


class TrainingProgressWidget(QWidget):
    """
    학습 진행 상황을 표시하는 위젯
    프로그레스 바 + 상태 메시지
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.hide()  # 초기에는 숨김

    def _setup_ui(self):
        from PySide6.QtWidgets import QProgressBar, QVBoxLayout, QHBoxLayout, QPushButton

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 상태 레이블
        self.lbl_status = QLabel("학습 준비 중...")
        self.lbl_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2d5016;
            padding: 5px;
        """)
        layout.addWidget(self.lbl_status)

        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #99cc99;
                border-radius: 8px;
                background-color: #fafffa;
                text-align: center;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66cc66, stop:1 #99ff99
                );
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 상세 정보 레이블
        self.lbl_details = QLabel("")
        self.lbl_details.setStyleSheet("font-size: 11px; color: #666; padding: 3px;")
        layout.addWidget(self.lbl_details)

        # 전체 컨테이너 스타일
        self.setStyleSheet("""
            TrainingProgressWidget {
                background-color: #e6f5e6;
                border: 2px solid #99cc99;
                border-radius: 10px;
            }
        """)

    def update_status(self, message: str):
        """상태 메시지 업데이트"""
        self.lbl_status.setText(message)

    def update_progress(self, value: int):
        """진행률 업데이트 (0-100)"""
        self.progress_bar.setValue(value)
        self.lbl_details.setText(f"진행률: {value}%")

    def set_error(self, error_msg: str):
        """에러 표시"""
        self.lbl_status.setText(f"❌ 학습 실패")
        self.lbl_details.setText(error_msg[:100])
        self.lbl_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #cc0000;
            padding: 5px;
        """)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ff6666;
                border-radius: 8px;
                background-color: #fff0f0;
                text-align: center;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #ff6666;
                border-radius: 6px;
            }
        """)

    def set_success(self):
        """성공 표시"""
        self.lbl_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #006600;
            padding: 5px;
        """)

    def reset(self):
        """초기화"""
        self.lbl_status.setText("학습 준비 중...")
        self.lbl_details.setText("")
        self.progress_bar.setValue(0)
        self.lbl_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2d5016;
            padding: 5px;
        """)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #99cc99;
                border-radius: 8px;
                background-color: #fafffa;
                text-align: center;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66cc66, stop:1 #99ff99
                );
                border-radius: 6px;
            }
        """)
