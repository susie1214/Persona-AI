# run_gui.py
import sys
from unittest.mock import MagicMock

# --- 백엔드 모듈 Mocking ---
# UI를 단독으로 띄우기 위해, 로딩이 오래 걸리거나 의존성이 복잡한
# core 모듈들을 임시 객체(MagicMock)로 대체합니다.
# 이렇게 하면 해당 모듈의 실제 코드가 실행되지 않습니다.
sys.modules['core.audio'] = MagicMock()
sys.modules['core.diarization'] = MagicMock()
sys.modules['core.summarizer'] = MagicMock()
sys.modules['core.rag_store'] = MagicMock()
sys.modules['core.adapter'] = MagicMock()

# 일부 UI 컴포넌트도 Mocking 처리합니다.
# sys.modules['ui.survey_wizard'] = MagicMock()
# sys.modules['ui.chat_dock'] = MagicMock()
# sys.modules['ui.meeting_notes'] = MagicMock()

# --- Mock 객체의 반환값 설정 ---
# UI 코드 중 특정 함수의 반환값이 필요한 경우, 미리 지정해줍니다.
# 예: 요약 함수는 항상 "UI 미리보기 모드입니다." 라는 텍스트를 반환하도록 설정
mock_summarizer = sys.modules['core.summarizer']
mock_summarizer.simple_summarize.return_value = "UI 미리보기 모드입니다."

# --- GUI 실행 ---
from PySide6.QtWidgets import QApplication
# Mock 설정이 끝난 후, GUI 클래스를 import 합니다.
from ui.meeting_console import MeetingConsole

if __name__ == "__main__":
    """
    이 스크립트는 app.py 대신 GUI만 빠르게 확인하고 싶을 때 사용합니다.
    'python run_gui.py' 명령으로 실행하세요.
    """
    app = QApplication(sys.argv)
    
    # MeetingConsole을 생성할 때, 내부의 core 관련 객체들은
    # 모두 MagicMock으로 대체되었으므로 실제 백엔드 로직은 실행되지 않습니다.
    win = MeetingConsole()
    win.show()
    
    sys.exit(app.exec())
