# -*- coding: utf-8 -*-
# core/speaker/__init__.py
"""
Speaker Management Package

화자 식별, 프로필 관리, 음성 임베딩 저장소
"""

from ..speaker import Speaker, SpeakerManager
from ..voice_store import VoiceStore

__all__ = [
    'Speaker',
    'SpeakerManager',
    'VoiceStore',
]
