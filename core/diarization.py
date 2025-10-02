# core/diarization.py
import os, time, threading
import numpy as np
from PySide6.QtCore import QObject, Signal
from core.speaker import SpeakerManager

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

PYANNOTE_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV = "HF_TOKEN"


class DiarizationWorker(QObject):
    sig_status = Signal(str)
    sig_diar_done = Signal(list)  # list[(start,end,speaker_id,confidence)]
    sig_new_speaker = Signal(str, str)  # (speaker_id, display_name)

    def __init__(self, state, speaker_manager=None, interval_sec=30):
        super().__init__()
        self.state = state
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thr = None
        # SpeakerManager 공유 (없으면 새로 생성)
        self.speaker_manager = speaker_manager if speaker_manager else SpeakerManager()

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

        while not self._stop.is_set():
            time.sleep(self.interval)
            path = self.state.raw_audio_path
            if not path or not os.path.exists(path):
                continue
            try:
                diar = pipeline(path)

                # 임베딩 추출 시도
                try:
                    embeddings = pipeline.embedding_model(path)
                except AttributeError:
                    # embedding_model이 없는 경우 임베딩 없이 처리
                    embeddings = None

                results = []
                for turn, _, pyannote_speaker in diar.itertracks(yield_label=True):
                    # 임베딩이 있는 경우에만 화자 식별 수행
                    if embeddings is not None:
                        try:
                            # 현재 화자(turn)에 해당하는 임베딩을 추출
                            embedding = embeddings.crop(turn)
                            embedding = np.mean(embedding, axis=0)

                            # 화자 식별 또는 새 화자 생성
                            speaker_id, confidence = self.speaker_manager.identify_speaker(embedding, 0.72)
                            display_name = self.speaker_manager.get_speaker_display_name(speaker_id)

                            # 새로운 화자인 경우 신호 발송
                            if speaker_id not in self.state.speaker_map:
                                self.sig_new_speaker.emit(speaker_id, display_name)

                            results.append((turn.start, turn.end, speaker_id, confidence))

                        except Exception as e:
                            # 임베딩 처리 실패 시 기본 화자 ID 사용
                            self.sig_status.emit(f"Embedding error: {e}")
                            results.append((turn.start, turn.end, pyannote_speaker, 0.5))
                    else:
                        # 임베딩이 없는 경우 pyannote의 기본 화자 ID 사용
                        results.append((turn.start, turn.end, pyannote_speaker, 0.5))

                self.sig_diar_done.emit(results)
                self.sig_status.emit(f"화자 분리 완료: {len(results)}개 구간")
            except Exception as e:
                self.sig_status.emit(f"Diarization error: {e}")

    def get_speaker_manager(self) -> SpeakerManager:
        """화자 매니저 반환"""
        return self.speaker_manager
