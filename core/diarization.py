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
        # 마지막 처리 위치 추적 (증분 처리용)
        self._last_processed_time = 0.0

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
        # Pipeline 로드 (한 번만)
        try:
            if PyannotePipeline is None:
                self.sig_status.emit("pyannote.audio가 설치되지 않았습니다.")
                return

            pipeline = PyannotePipeline.from_pretrained(
                PYANNOTE_PIPELINE_NAME, use_auth_token=os.getenv(HF_TOKEN_ENV)
            )
            self.sig_status.emit(f"✓ Pyannote pipeline 로드 완료 ({PYANNOTE_PIPELINE_NAME})")
        except Exception as e:
            self.sig_status.emit(f"Diar pipeline 로드 실패: {e}")
            return

        last_file_size = 0  # 파일 크기 변화 추적 (최적화)

        while not self._stop.is_set():
            time.sleep(self.interval)
            path = self.state.raw_audio_path

            if not path or not os.path.exists(path):
                continue

            # 파일 크기가 변하지 않았으면 스킵 (최적화)
            try:
                current_size = os.path.getsize(path)
                if current_size == last_file_size and last_file_size > 0:
                    continue
                last_file_size = current_size
            except Exception:
                pass

            try:
                diar = pipeline(path)

                # 임베딩 추출 시도
                embeddings = None
                try:
                    if hasattr(pipeline, 'embedding_model'):
                        embeddings = pipeline.embedding_model(path)
                except Exception as emb_err:
                    # 임베딩 추출 실패는 치명적이지 않음
                    print(f"[WARN] Embedding extraction failed: {emb_err}")

                results = []
                for turn, _, pyannote_speaker in diar.itertracks(yield_label=True):
                    # 임베딩 기반 화자 식별
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
                            print(f"[WARN] Speaker embedding error: {e}")
                            results.append((turn.start, turn.end, pyannote_speaker, 0.5))
                    else:
                        # 임베딩이 없는 경우 pyannote의 기본 화자 ID 사용
                        results.append((turn.start, turn.end, pyannote_speaker, 0.5))

                self.sig_diar_done.emit(results)
                self.sig_status.emit(f"✓ 화자 분리 완료: {len(results)}개 구간")

            except Exception as e:
                self.sig_status.emit(f"Diarization error: {e}")
                import traceback
                print(f"[ERROR] Diarization failed:\n{traceback.format_exc()}")

    def get_speaker_manager(self) -> SpeakerManager:
        """화자 매니저 반환"""
        return self.speaker_manager