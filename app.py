# app.py
import os, sys, traceback, faulthandler, logging
from PySide6.QtWidgets import QApplication
from ui.meeting_console import MeetingConsole

# 경고 소음 줄이기
import warnings
warnings.filterwarnings("ignore", module="pyannote.audio")

# CPU 강제 실행을 원하면 .env에 FORCE_CPU=1
if os.getenv("FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 크래시 원인 파악 로그
os.makedirs("output/logs", exist_ok=True)
LOGFILE = "output/logs/app.log"
logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
faulthandler.enable(open("output/logs/faulthandler.log", "w"))

def excepthook(etype, value, tb):
    msg = "".join(traceback.format_exception(etype, value, tb))
    print(msg, file=sys.stderr)
    logging.error("UNCAUGHT EXCEPTION:\n%s", msg)
    # 창이 그냥 닫히지 않도록 마지막 메시지 출력 대기
    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Fatal Error", msg[:2000])
    except Exception:
        pass
    sys.__excepthook__(etype, value, tb)

sys.excepthook = excepthook

def main():
    app = QApplication(sys.argv)
    win = MeetingConsole()
    win.show()
    try:
        sys.exit(app.exec())
    except Exception:
        # Qt 내부 예외도 로그로 남김
        logging.exception("Qt main loop exception")
        raise

if __name__ == "__main__":
    main()
