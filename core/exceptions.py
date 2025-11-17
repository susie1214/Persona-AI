# -*- coding: utf-8 -*-
# core/exceptions.py
"""
Persona-AI Custom Exception Hierarchy

모든 Persona-AI 예외는 PersonaAIError를 상속합니다.
"""

from typing import Optional


class PersonaAIError(Exception):
    """
    모든 Persona-AI 예외의 기본 클래스

    Attributes:
        message: 에러 메시지
        code: 에러 코드 (내부 분류용)
        details: 추가 상세 정보
    """

    code: str = "UNKNOWN"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[str] = None
    ):
        self.message = message
        self.code = code or self.__class__.code
        self.details = details
        super().__init__(f"[{self.code}] {message}")


# ============================================================================
# Audio & STT Exceptions
# ============================================================================

class AudioError(PersonaAIError):
    """오디오 처리 관련 에러"""
    code = "AUDIO_ERROR"


class AudioDeviceNotFoundError(AudioError):
    """오디오 디바이스를 찾을 수 없음"""
    code = "AUDIO_DEVICE_NOT_FOUND"


class AudioInitializationError(AudioError):
    """오디오 초기화 실패"""
    code = "AUDIO_INIT_FAILED"


class AudioStreamError(AudioError):
    """오디오 스트림 처리 실패"""
    code = "AUDIO_STREAM_ERROR"


class STTError(AudioError):
    """Speech-to-Text 변환 실패"""
    code = "STT_ERROR"


class DiarizationError(AudioError):
    """화자 분리(Diarization) 실패"""
    code = "DIARIZATION_ERROR"


# ============================================================================
# Speaker Management Exceptions
# ============================================================================

class SpeakerError(PersonaAIError):
    """화자 관리 관련 에러"""
    code = "SPEAKER_ERROR"


class SpeakerNotFoundError(SpeakerError):
    """화자를 찾을 수 없음"""
    code = "SPEAKER_NOT_FOUND"


class SpeakerAlreadyExistsError(SpeakerError):
    """화자가 이미 존재함"""
    code = "SPEAKER_ALREADY_EXISTS"


class VoiceStoreError(SpeakerError):
    """음성 임베딩 저장소 에러"""
    code = "VOICE_STORE_ERROR"


# ============================================================================
# Persona & Personalization Exceptions
# ============================================================================

class PersonaError(PersonaAIError):
    """페르소나 관련 에러"""
    code = "PERSONA_ERROR"


class PersonaNotFoundError(PersonaError):
    """페르소나를 찾을 수 없음"""
    code = "PERSONA_NOT_FOUND"


class AdapterError(PersonaError):
    """QLoRA 어댑터 관련 에러"""
    code = "ADAPTER_ERROR"


class AdapterLoadError(AdapterError):
    """어댑터 로드 실패"""
    code = "ADAPTER_LOAD_ERROR"


class AdapterNotFoundError(AdapterError):
    """어댑터를 찾을 수 없음"""
    code = "ADAPTER_NOT_FOUND"


class PersonaStoreError(PersonaError):
    """페르소나 저장소 에러"""
    code = "PERSONA_STORE_ERROR"


# ============================================================================
# LLM & Model Exceptions
# ============================================================================

class ModelError(PersonaAIError):
    """모델 관련 에러"""
    code = "MODEL_ERROR"


class ModelLoadError(ModelError):
    """모델 로드 실패"""
    code = "MODEL_LOAD_ERROR"


class ModelNotFoundError(ModelError):
    """모델을 찾을 수 없음"""
    code = "MODEL_NOT_FOUND"


class InferenceError(ModelError):
    """모델 추론 실패"""
    code = "INFERENCE_ERROR"


class LLMRouterError(ModelError):
    """LLM 라우터 에러"""
    code = "LLM_ROUTER_ERROR"


class UnsupportedBackendError(LLMRouterError):
    """지원하지 않는 LLM 백엔드"""
    code = "UNSUPPORTED_BACKEND"


# ============================================================================
# RAG & Vector Store Exceptions
# ============================================================================

class RAGError(PersonaAIError):
    """RAG(검색 증강 생성) 관련 에러"""
    code = "RAG_ERROR"


class SearchError(RAGError):
    """벡터 검색 실패"""
    code = "SEARCH_ERROR"


class VectorStoreError(RAGError):
    """벡터 데이터베이스 에러"""
    code = "VECTOR_STORE_ERROR"


class CollectionError(RAGError):
    """컬렉션 관련 에러"""
    code = "COLLECTION_ERROR"


class DocumentUploadError(RAGError):
    """문서 업로드 실패"""
    code = "DOCUMENT_UPLOAD_ERROR"


class EmbeddingError(RAGError):
    """임베딩 생성 실패"""
    code = "EMBEDDING_ERROR"


# ============================================================================
# Training & Learning Exceptions
# ============================================================================

class TrainingError(PersonaAIError):
    """모델 학습 관련 에러"""
    code = "TRAINING_ERROR"


class DatasetError(TrainingError):
    """학습 데이터셋 생성 실패"""
    code = "DATASET_ERROR"


class InsufficientDataError(TrainingError):
    """학습에 필요한 데이터 부족"""
    code = "INSUFFICIENT_DATA"


class TrainingTimeoutError(TrainingError):
    """학습 시간 초과"""
    code = "TRAINING_TIMEOUT"


class GPUOutOfMemoryError(TrainingError):
    """GPU 메모리 부족"""
    code = "GPU_OOM"


# ============================================================================
# Analysis & Export Exceptions
# ============================================================================

class AnalysisError(PersonaAIError):
    """분석 관련 에러"""
    code = "ANALYSIS_ERROR"


class SummaryError(AnalysisError):
    """회의 요약 생성 실패"""
    code = "SUMMARY_ERROR"


class ExportError(AnalysisError):
    """내보내기 실패"""
    code = "EXPORT_ERROR"


class DocumentProcessingError(AnalysisError):
    """문서 처리 실패"""
    code = "DOCUMENT_PROCESSING_ERROR"


# ============================================================================
# Configuration & Validation Exceptions
# ============================================================================

class ConfigError(PersonaAIError):
    """설정 관련 에러"""
    code = "CONFIG_ERROR"


class ValidationError(PersonaAIError):
    """입력 검증 실패"""
    code = "VALIDATION_ERROR"


class EnvironmentError(PersonaAIError):
    """환경 설정 에러"""
    code = "ENV_ERROR"


# ============================================================================
# System & I/O Exceptions
# ============================================================================

class FileOperationError(PersonaAIError):
    """파일 작업 실패"""
    code = "FILE_OP_ERROR"


class DirectoryNotFoundError(FileOperationError):
    """디렉터리를 찾을 수 없음"""
    code = "DIR_NOT_FOUND"


class PermissionError(FileOperationError):
    """권한 부족"""
    code = "PERMISSION_DENIED"


class StorageError(PersonaAIError):
    """저장소 관련 에러"""
    code = "STORAGE_ERROR"


class DatabaseError(StorageError):
    """데이터베이스 에러"""
    code = "DATABASE_ERROR"
