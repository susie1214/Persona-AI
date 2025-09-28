# -*- coding: utf-8 -*-
"""
설문 마법사 → 저장소 업데이트 → 챗봇 반영
- 설문 완료 시 PersonaStore.update_from_survey(...)로 저장소 갱신
- 디스크에도 data/persona/{user_id}.json 로 저장 (백업/호환 목적)
- persona_updated(str) 시그널로 상위(메인/챗봇)에게 갱신 알림
"""
import os
import json
from typing import List, Dict

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
)
from PySide6.QtCore import Signal

# 저장소 어댑터 (외부 모듈)
# - 인터페이스 가정:
#   PersonaStore().update_from_survey(name: str, survey: dict) -> None
#   PersonaStore().build_system_prompt(name: str) -> str   (선택)
from core.persona_store import PersonaStore


class PersonaSurveyWizard(QWizard):
    """
    개인 맞춤 페르소나 설문 마법사
    """

    # 예: "조진경" 같은 표시 이름을 전달
    persona_updated = Signal(str)

    def __init__(self, user_id: str = "jkj", display_name: str = "조진경", parent=None):
        super().__init__(parent)
        self.setWindowTitle("개인 맞춤 페르소나 설문")
        self.user_id = user_id  # 파일 저장용 식별자
        self.display_name = display_name  # UI/Store용 이름(표시명)
        self.store = PersonaStore()

        # ------- Page 1: 글쓰기/보고 스타일 -------
        self.page_style = QWizardPage()
        self.page_style.setTitle("글쓰기/보고 스타일")
        L1 = QVBoxLayout(self.page_style)

        self.cmb_tone = QComboBox()
        self.cmb_tone.addItems(["명확/직설", "정중/공식", "친근/편안", "데이터 중심"])

        self.cmb_format = QComboBox()
        self.cmb_format.addItems(
            ["개조식, 결론 우선", "서술식", "키워드 중심", "표/차트 활용"]
        )

        self.cmb_sentence_len = QComboBox()
        self.cmb_sentence_len.addItems(["짧게", "적당히", "길어도 됨"])

        self.txt_jargon = QTextEdit()
        self.txt_jargon.setPlaceholderText(
            "자주 쓰는 표현/전문용어/줄임말 (여러 개 입력 가능)"
        )

        L1.addWidget(QLabel("선호 말투:"))
        L1.addWidget(self.cmb_tone)
        L1.addWidget(QLabel("요약/보고 형식:"))
        L1.addWidget(self.cmb_format)
        L1.addWidget(QLabel("문장 길이:"))
        L1.addWidget(self.cmb_sentence_len)
        L1.addWidget(QLabel("전문 용어/표현:"))
        L1.addWidget(self.txt_jargon)

        # ------- Page 2: 키워드/알람/리포트 포커스 -------
        self.page_prefs = QWizardPage()
        self.page_prefs.setTitle("선호도/알림/리포트 포커스")
        L2 = QVBoxLayout(self.page_prefs)

        self.edit_keywords = QLineEdit()
        self.edit_keywords.setPlaceholderText(
            "자주 쓰는 키워드 (쉼표로 구분: ASAP, 애자일, 피봇 ...)"
        )

        self.cmb_alarm_default = QComboBox()
        self.cmb_alarm_default.addItems(
            ["회의 10분 전", "회의 30분 전", "회의 1시간 전", "전날 저녁", "이틀 전"]
        )

        self.edit_alarm_fields = QLineEdit()
        self.edit_alarm_fields.setPlaceholderText(
            "알람 포함 정보 (쉼표로: 제목, 목적, 참석자, 준비물 ...)"
        )

        self.cmb_report_focus = QComboBox()
        self.cmb_report_focus.addItems(
            ["정확성", "간결성", "데이터 시각화", "KPI/Action"]
        )

        self.chk_english = QCheckBox("영어 병기 허용")

        L2.addWidget(QLabel("자주 쓰는 키워드:"))
        L2.addWidget(self.edit_keywords)
        L2.addWidget(QLabel("알람 기본값:"))
        L2.addWidget(self.cmb_alarm_default)
        L2.addWidget(QLabel("알람에 포함할 필드:"))
        L2.addWidget(self.edit_alarm_fields)
        L2.addWidget(QLabel("보고서 포커스:"))
        L2.addWidget(self.cmb_report_focus)
        L2.addWidget(self.chk_english)

        # ------- Page 3: 엔진/메모 & 동의 -------
        self.page_engine = QWizardPage()
        self.page_engine.setTitle("엔진/메모 & 동의")
        L3 = QVBoxLayout(self.page_engine)

        self.cmb_backend = QComboBox()
        self.cmb_backend.addItems(
            [
                "openai:gpt-4o-mini",
                "ollama:llama3",
                "ax:A.X-4.0",
                "midm:Midm-2.0-Mini-Instruct",
            ]
        )

        self.memo = QTextEdit()
        self.memo.setPlaceholderText("추가 메모(어투/예외/도메인 선호/금지어 등)")

        self.chk_consent = QCheckBox("개인화 학습·서비스 제공 목적 활용에 동의합니다.")

        L3.addWidget(QLabel("기본 백엔드:"))
        L3.addWidget(self.cmb_backend)
        L3.addWidget(QLabel("메모:"))
        L3.addWidget(self.memo)
        L3.addWidget(self.chk_consent)

        # 페이지 등록
        self.addPage(self.page_style)
        self.addPage(self.page_prefs)
        self.addPage(self.page_engine)

        # Finish(완료) 시그널 → on_finish
        # - accepted는 Finish 버튼이 눌리고 검증을 통과했을 때 방출
        self.accepted.connect(self.on_finish)

    # --- 내부 유틸 ---
    @staticmethod
    def _split_csv(text: str) -> List[str]:
        return [t.strip() for t in (text or "").split(",") if t.strip()]

    # --- 제출 처리 ---
    def on_finish(self):
        """
        Finish 후 호출: 저장소 업데이트 + 파일 저장 + 신호 발행
        (동의 체크는 accept()에서 보장)
        """
        survey: Dict = {
            "tone": self.cmb_tone.currentText(),
            "summary_format": self.cmb_format.currentText(),
            "sentence_len": self.cmb_sentence_len.currentText(),
            "jargon": self.txt_jargon.toPlainText().strip(),
            "keywords": self._split_csv(self.edit_keywords.text()),
            "alarm": self.cmb_alarm_default.currentText(),
            "alarm_fields": self._split_csv(self.edit_alarm_fields.text()),
            "report_focus": self.cmb_report_focus.currentText(),
            "backend": self.cmb_backend.currentText(),
            "english_bilingual": bool(self.chk_english.isChecked()),
            "memo": self.memo.toPlainText().strip(),
        }

        # 1) 저장소 갱신
        try:
            self.store.update_from_survey(self.display_name, survey)
        except Exception as e:
            QMessageBox.critical(
                self,
                "저장소 업데이트 실패",
                f"PersonaStore 업데이트 중 오류가 발생했습니다.\n{e}",
            )
            # 저장소 실패해도 파일 백업은 시도
        # 2) 디스크 백업 (호환 유지)
        try:
            os.makedirs("data/persona", exist_ok=True)
            payload = {
                "user_id": self.user_id,
                "name": self.display_name,
                "style": {
                    "tone": survey["tone"],
                    "format": survey["summary_format"],
                    "sentence_len": survey["sentence_len"],
                    "jargon": survey["jargon"],
                    "keywords": survey["keywords"],
                },
                "alerts": {
                    "remind": survey["alarm"],
                    "fields": survey["alarm_fields"],
                },
                "report_focus": survey["report_focus"],
                "backend": survey["backend"],
                "english_bilingual": survey["english_bilingual"],
                "memo": survey["memo"],
                "consent": True,
            }
            with open(f"data/persona/{self.user_id}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(
                self, "파일 저장 경고", f"로컬 백업 저장에 실패했습니다.\n{e}"
            )

        # 3) 상위에 "이름" 신호 발행 → 챗봇/세션이 즉시 반영하도록 훅 연결
        try:
            self.persona_updated.emit(self.display_name)
        except Exception as e:
            # 시그널 에러는 드물지만, 안전망
            QMessageBox.warning(
                self, "신호 발행 경고", f"페르소나 갱신 신호 발행 중 문제 발생\n{e}"
            )

    # Finish 눌렀을 때 동의 체크 강제
    def accept(self):
        if not self.chk_consent.isChecked():
            QMessageBox.information(
                self, "동의 필요", "개인화 학습·서비스 제공 목적 활용에 동의해 주세요."
            )
            return
        super().accept()
