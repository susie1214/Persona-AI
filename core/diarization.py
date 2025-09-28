# core/diarization.py
import os, time, threading
from PySide6.QtCore import QObject, Signal

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

PYANNOTE_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV = "HF_TOKEN"


class DiarizationWorker(QObject):
    sig_status = Signal(str)
    sig_diar_done = Signal(list)  # list[(start,end,spk)]

    def __init__(self, state, interval_sec=30):
        super().__init__()
        self.state = state
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        if not self.state.diarization_enabled:
            return
        if PyannotePipeline is None:
            self.sig_status.emit("pyannote 미설치로 diarization 비활성.")
            return
        if not os.getenv(HF_TOKEN_ENV, ""):
            self.sig_status.emit(f"{HF_TOKEN_ENV} 미설정으로 diarization 비활성.")
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        self.sig_status.emit("Diarization thread started.")

    def stop(self):
        self._stop.set()

    def _loop(self):
        try:
            pipeline = PyannotePipeline.from_pretrained(
                PYANNOTE_PIPELINE_NAME, use_auth_token=os.getenv(HF_TOKEN_ENV)
            )
        except Exception as e:
            self.sig_status.emit(f"Diar pipeline 로드 실패: {e}")
            return

        last_emit_hash = None
        while not self._stop.is_set():
            time.sleep(self.interval)
            path = self.state.raw_audio_path
            if not path or not os.path.exists(path):
                continue
            try:
                diar = pipeline(path)
                results = []
                for turn, _, speaker in diar.itertracks(yield_label=True):
                    results.append((turn.start, turn.end, speaker))
                h = hash(tuple(results))
                if h != last_emit_hash:
                    self.sig_diar_done.emit(results)
                    last_emit_hash = h
                    self.sig_status.emit(f"Diar segments: {len(results)}")
            except Exception as e:
                self.sig_status.emit(f"Diarization error: {e}")
