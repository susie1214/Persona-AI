# -*- coding: utf-8 -*-
# ui/base_tab.py
"""
BaseTab: 모든 탭 위젯의 기본 클래스
"""

from abc import ABC, abstractmethod
from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal


class BaseTab(QWidget, ABC):
    """
    모든 탭 위젯의 기본 클래스

    각 탭은 다음을 구현해야 함:
    - setup_ui(): UI 레이아웃 구성
    - connect_signals(): 신호 연결
    """

    # 각 탭이 발생시킬 수 있는 공통 신호들
    sig_status = Signal(str)           # 상태 메시지
    sig_error = Signal(str)            # 에러 메시지

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None  # 외부 컨트롤러 (MeetingConsole)

        # 기본 레이아웃
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

    @abstractmethod
    def setup_ui(self):
        """UI 레이아웃 구성 (자식 클래스에서 구현)"""
        pass

    @abstractmethod
    def connect_signals(self):
        """신호 연결 (자식 클래스에서 구현)"""
        pass

    def set_controller(self, controller):
        """외부 컨트롤러 설정"""
        self.controller = controller

    def emit_status(self, msg: str):
        """상태 메시지 발생"""
        self.sig_status.emit(msg)

    def emit_error(self, msg: str):
        """에러 메시지 발생"""
        self.sig_error.emit(msg)
