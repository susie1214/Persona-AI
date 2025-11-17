# app.py
import os, sys, traceback, faulthandler, logging
from PySide6.QtWidgets import QApplication
from ui.meeting_console import MeetingConsole
from PySide6.QtGui import QIcon

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"[INFO] .env 파일 로드 완료")
    if os.getenv("HF_TOKEN"):
        print(f"[INFO] HF_TOKEN 설정됨: {os.getenv('HF_TOKEN', "None")[:10]}...")
    else:
        print(f"[WARNING] HF_TOKEN이 .env에 없거나 비어있습니다.")
except ImportError:
    print("[WARNING] python-dotenv가 설치되지 않았습니다. .env 파일을 사용하려면 'pip install python-dotenv'를 실행하세요.")
except Exception as e:
    print(f"[ERROR] .env 파일 로드 실패: {e}")

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

    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Fatal Error", msg[:2000])
    except Exception:
        pass
    sys.__excepthook__(etype, value, tb)

sys.excepthook = excepthook

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./resources/PersonaJK.png"))
    
    win = MeetingConsole()
    win.show()
    
    try:
        sys.exit(app.exec())
    except Exception:
        logging.exception("Qt main loop exception")
        raise

if __name__ == "__main__":
    main()
