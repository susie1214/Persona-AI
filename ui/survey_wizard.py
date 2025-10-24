# -*- coding: utf-8 -*-
"""
디지털 페르소나 사전 지식 입력 마법사
- 화자에 대한 사전 지식(role, expertise, personality, communication style 등)을 수집
- DigitalPersonaManager를 통해 페르소나 정보를 저장하고 강화
- persona_updated(str) 시그널로 상위 컴포넌트에 갱신 알림
"""
import os
import json
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWizard,
    QWizardPage,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QMessageBox,
    QHBoxLayout,
    QSpinBox,
)
from PySide6.QtCore import Signal

# 디지털 페르소나 관리자
from core.digital_persona import DigitalPersonaManager
from core.persona_store import PersonaStore
from core.rag_store import RagStore
from core.voice_store import VoiceStore


class DigitalPersonaPriorKnowledgeWizard(QWizard):
    """
    디지털 페르소나 사전 지식 입력 마법사
    - 화자의 역할, 전문성, 성격, 커뮤니케이션 스타일 등을 수집
    - DigitalPersonaManager를 통해 페르소나 강화
    """

    # 페르소나가 업데이트되면 speaker_id를 전달
    persona_updated = Signal(str)

    def __init__(
        self,
        speaker_id: str,
        display_name: str,
        persona_manager: Optional[DigitalPersonaManager] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"디지털 페르소나 설정 - {display_name}")
        self.speaker_id = speaker_id
        self.display_name = display_name

        # DigitalPersonaManager 초기화
        if persona_manager:
            self.persona_manager = persona_manager
        else:
            voice_store = VoiceStore()
            rag_store = RagStore()
            persona_store = PersonaStore()
            self.persona_manager = DigitalPersonaManager(
                voice_store=voice_store,
                rag_store=rag_store,
                persona_store=persona_store,
                storage_path="data/digital_personas"
            )

        # ------- Page 1: 기본 정보 및 역할 -------
        self.page_basic = QWizardPage()
        self.page_basic.setTitle("기본 정보 및 역할")
        L1 = QVBoxLayout(self.page_basic)

        self.edit_role = QLineEdit()
        self.edit_role.setPlaceholderText("예: 백엔드 개발자, 프로덕트 매니저, 디자이너")

        self.edit_department = QLineEdit()
        self.edit_department.setPlaceholderText("예: 개발팀, 기획팀, 디자인팀")

        self.edit_expertise = QTextEdit()
        self.edit_expertise.setPlaceholderText(
            "전문 분야 (쉼표로 구분):\n예: Python, FastAPI, 데이터베이스 설계, 마이크로서비스"
        )
        self.edit_expertise.setFixedHeight(80)

        L1.addWidget(QLabel("👤 역할/직책:"))
        L1.addWidget(self.edit_role)
        L1.addWidget(QLabel("🏢 부서/팀:"))
        L1.addWidget(self.edit_department)
        L1.addWidget(QLabel("💡 전문 분야:"))
        L1.addWidget(self.edit_expertise)

        # ------- Page 2: 성격 및 커뮤니케이션 스타일 -------
        self.page_personality = QWizardPage()
        self.page_personality.setTitle("성격 및 커뮤니케이션 스타일")
        L2 = QVBoxLayout(self.page_personality)

        self.edit_personality = QTextEdit()
        self.edit_personality.setPlaceholderText(
            "성격 키워드 (쉼표로 구분):\n예: 분석적, 논리적, 협력적, 창의적, 세심함"
        )
        self.edit_personality.setFixedHeight(70)

        self.cmb_tone = QComboBox()
        self.cmb_tone.addItems(["명확/직설적", "정중/공식적", "친근/편안함", "데이터 중심"])

        self.cmb_format = QComboBox()
        self.cmb_format.addItems(
            ["개조식, 결론 우선", "서술식, 맥락 중심", "키워드 중심", "표/차트 활용"]
        )

        self.cmb_sentence_len = QComboBox()
        self.cmb_sentence_len.addItems(["짧고 간결하게", "적당한 길이", "상세하게"])

        self.edit_jargon = QTextEdit()
        self.edit_jargon.setPlaceholderText(
            "자주 쓰는 전문용어/표현 (쉼표로 구분):\n예: 피봇, ASAP, 애자일, KPI"
        )
        self.edit_jargon.setFixedHeight(70)

        L2.addWidget(QLabel("🎭 성격 키워드:"))
        L2.addWidget(self.edit_personality)
        L2.addWidget(QLabel("💬 선호 말투:"))
        L2.addWidget(self.cmb_tone)
        L2.addWidget(QLabel("📝 의사소통 형식:"))
        L2.addWidget(self.cmb_format)
        L2.addWidget(QLabel("📏 문장 길이 선호:"))
        L2.addWidget(self.cmb_sentence_len)
        L2.addWidget(QLabel("🔤 자주 쓰는 용어/표현:"))
        L2.addWidget(self.edit_jargon)

        # ------- Page 3: 경력 및 추가 정보 -------
        self.page_career = QWizardPage()
        self.page_career.setTitle("경력 및 추가 정보")
        L3 = QVBoxLayout(self.page_career)

        # 경력 연수
        career_layout = QHBoxLayout()
        self.spin_career_years = QSpinBox()
        self.spin_career_years.setMinimum(0)
        self.spin_career_years.setMaximum(50)
        self.spin_career_years.setValue(0)
        career_layout.addWidget(QLabel("💼 경력 연수:"))
        career_layout.addWidget(self.spin_career_years)
        career_layout.addWidget(QLabel("년"))
        career_layout.addStretch()

        self.edit_education = QLineEdit()
        self.edit_education.setPlaceholderText("예: 컴퓨터공학 학사, MBA")

        self.edit_skills = QTextEdit()
        self.edit_skills.setPlaceholderText(
            "주요 기술/도구 (쉼표로 구분):\n예: Docker, Kubernetes, AWS, PostgreSQL"
        )
        self.edit_skills.setFixedHeight(70)

        self.edit_interests = QTextEdit()
        self.edit_interests.setPlaceholderText(
            "관심 분야/학습 주제 (쉼표로 구분):\n예: 머신러닝, 클라우드 아키텍처, UX 디자인"
        )
        self.edit_interests.setFixedHeight(70)

        L3.addLayout(career_layout)
        L3.addWidget(QLabel("🎓 학력:"))
        L3.addWidget(self.edit_education)
        L3.addWidget(QLabel("🛠️ 주요 기술/도구:"))
        L3.addWidget(self.edit_skills)
        L3.addWidget(QLabel("📚 관심 분야:"))
        L3.addWidget(self.edit_interests)

        # ------- Page 4: 추가 정보 및 동의 -------
        self.page_settings = QWizardPage()
        self.page_settings.setTitle("추가 정보 및 동의")
        L4 = QVBoxLayout(self.page_settings)

        # LLM 백엔드 선택 제거 - Settings 탭에서 전역 설정 사용

        self.edit_memo = QTextEdit()
        self.edit_memo.setPlaceholderText(
            "추가 메모/특이사항:\n예: 특정 주제에 대한 선호도, 금지어, 특별 지시사항 등"
        )
        self.edit_memo.setFixedHeight(80)

        self.chk_consent = QCheckBox("디지털 페르소나 생성 및 학습 목적 데이터 활용에 동의합니다.")

        # LLM 백엔드 위젯 제거
        L4.addWidget(QLabel("📋 추가 메모:"))
        L4.addWidget(self.edit_memo)
        L4.addWidget(QLabel(""))
        L4.addWidget(self.chk_consent)

        # 페이지 등록
        self.addPage(self.page_basic)
        self.addPage(self.page_personality)
        self.addPage(self.page_career)
        self.addPage(self.page_settings)

        # Finish 시그널 연결
        self.accepted.connect(self.on_finish)

    # --- 내부 유틸 ---
    @staticmethod
    def _split_csv(text: str) -> List[str]:
        return [t.strip() for t in (text or "").split(",") if t.strip()]

    # --- 제출 처리 ---
    def on_finish(self):
        """
        사전 지식을 수집하고 DigitalPersonaManager를 통해 페르소나를 강화
        """
        # 사전 지식 딕셔너리 구성
        prior_knowledge = {
            "role": self.edit_role.text().strip(),
            "department": self.edit_department.text().strip(),
            "expertise": self._split_csv(self.edit_expertise.toPlainText()),
            "personality_keywords": self._split_csv(self.edit_personality.toPlainText()),
            "communication_style": {
                "tone": self.cmb_tone.currentText(),
                "format": self.cmb_format.currentText(),
                "sentence_length": self.cmb_sentence_len.currentText(),
                "jargon": self._split_csv(self.edit_jargon.toPlainText()),
            },
            "career": {
                "years": self.spin_career_years.value(),
                "education": self.edit_education.text().strip(),
                "skills": self._split_csv(self.edit_skills.toPlainText()),
                "interests": self._split_csv(self.edit_interests.toPlainText()),
            },
            # LLM 백엔드는 Settings 탭에서 전역 설정 사용 (페르소나별 설정 제거)
            # "llm_backend": self.cmb_backend.currentText(),
            "memo": self.edit_memo.toPlainText().strip(),
        }

        # DigitalPersonaManager를 통해 페르소나 강화
        try:
            success = self.persona_manager.enrich_from_prior_knowledge(
                speaker_id=self.speaker_id,
                prior_knowledge=prior_knowledge
            )

            if success:
                QMessageBox.information(
                    self,
                    "페르소나 업데이트 완료",
                    f"'{self.display_name}'의 디지털 페르소나가 성공적으로 업데이트되었습니다."
                )

                # 페르소나 업데이트 신호 발행
                self.persona_updated.emit(self.speaker_id)
            else:
                QMessageBox.warning(
                    self,
                    "페르소나 업데이트 실패",
                    f"페르소나를 찾을 수 없습니다: {self.speaker_id}\n먼저 음성 데이터가 수집되어야 합니다."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류 발생",
                f"페르소나 업데이트 중 오류가 발생했습니다.\n{e}"
            )

        # 선택적: 레거시 호환을 위한 파일 백업
        try:
            os.makedirs("data/persona", exist_ok=True)
            backup_payload = {
                "speaker_id": self.speaker_id,
                "display_name": self.display_name,
                "prior_knowledge": prior_knowledge,
                "consent": True,
            }
            with open(f"data/persona/{self.speaker_id}.json", "w", encoding="utf-8") as f:
                json.dump(backup_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save backup file: {e}")

    # Finish 눌렀을 때 동의 체크 강제
    def accept(self):
        if not self.chk_consent.isChecked():
            QMessageBox.information(
                self, "동의 필요", "디지털 페르소나 생성 및 학습 목적 데이터 활용에 동의해 주세요."
            )
            return
        super().accept()


# 레거시 호환을 위한 별칭 유지
PersonaSurveyWizard = DigitalPersonaPriorKnowledgeWizard
