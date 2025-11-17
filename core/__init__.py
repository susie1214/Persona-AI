# -*- coding: utf-8 -*-
# core/__init__.py
"""
Core Package

다양한 AI 기능을 제공하는 코어 패키지:
- audio: 오디오 처리 및 화자 분리
- speaker: 화자 관리 및 음성 저장소
- persona: 디지털 페르소나 및 QLoRA 어댑터
- llm: 다양한 LLM 백엔드 라우팅
- rag: RAG (Retrieval Augmented Generation)
- training: 모델 학습
- analysis: 회의 분석 및 요약
"""

# 주요 모듈 import (편의성)
from .audio import (
    AudioWorker,
    DiarizationWorker,
    Segment,
    MeetingState,
    fmt_time,
    now_str,
)
from .speaker import Speaker, SpeakerManager, VoiceStore
from .persona import (
    DigitalPersona,
    DigitalPersonaManager,
    AdapterManager,
    PersonaStore,
)
from .llm import LLMRouter, LLM
from .rag import RagStore
from .training import PersonaDatasetGenerator, PersonaTrainingWorker, TrainingProgressWidget
from .analysis import (
    render_summary_html_from_segments,
    actions_from_segments,
    render_actions_table_html,
    extract_agenda,
    llm_summarize,
    extract_schedules_from_summary,
)

__all__ = [
    # audio
    "AudioWorker",
    "DiarizationWorker",
    "Segment",
    "MeetingState",
    "fmt_time",
    "now_str",
    # speaker
    "Speaker",
    "SpeakerManager",
    "VoiceStore",
    # persona
    "DigitalPersona",
    "DigitalPersonaManager",
    "AdapterManager",
    "PersonaStore",
    # llm
    "LLMRouter",
    "LLM",
    # rag
    "RagStore",
    # training
    "PersonaDatasetGenerator",
    "PersonaTrainingWorker",
    "TrainingProgressWidget",
    # analysis
    "render_summary_html_from_segments",
    "actions_from_segments",
    "render_actions_table_html",
    "extract_agenda",
    "llm_summarize",
    "extract_schedules_from_summary",
]
