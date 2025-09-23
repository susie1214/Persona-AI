# persona_meeting_assistant.py
# Persona-AI 실시간 회의 비서 (PyQt6 완성본)
# - 마이크 -> (옵션) pyannote 자동 화자분리 -> faster-whisper STT
# - 참가자 등록/매핑 Dialog
# - 에코그린×옐로우 테마
# - Q&A: Qdrant in-memory RAG (문맥 근거 표시)
# - QLoRA 어댑터 스위칭 훅 (옵션)
# - Action/Schedule: QDateTimeEdit(TimeEdit)로 일정 메모 편집 + 알림 데모

import sys, os, io, time, tempfile, datetime, threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
import soundfile as sf
import dateparser

# ---------- Config ----------
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_SECONDS = 6
DEFAULT_MODEL = "medium"  # faster-whisper size
USE_GPU_DEFAULT = True

DIARIZATION_ENABLED_DEFAULT = False
PYANNOTE_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV = "HF_TOKEN"

THEME = {
    "bg": "#e6f5e6",
    "pane": "#99cc99",
    "light_bg": "#fafffa",
    "btn": "#ffe066",
    "btn_hover": "#ffdb4d",
    "btn_border": "#cccc99",
}

# ---------- Imports (runtime guards) ----------
try:
    import pyaudio
except Exception as e:
    pyaudio = None
    print("[WARN] PyAudio not available:", e)

try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    print("[WARN] faster-whisper not available:", e)

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

# Embedding / RAG
EMBED_AVAILABLE = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except Exception as e:
    EMBED_AVAILABLE = False
    print("[WARN] qdrant-client not available:", e)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# QLoRA(옵션)
PEFT_AVAILABLE = True
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except Exception:
    PEFT_AVAILABLE = False

# ---------- Qt ----------
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QDateTime
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QPlainTextEdit,
    QLabel,
    QTabWidget,
    QSplitter,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QDateTimeEdit,
)


# ---------- Data Structures ----------
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str = "Unknown"  # pyannote speaker label (e.g., SPEAKER_00)
    speaker_name: str = "Unknown"  # mapped human name


@dataclass
class MeetingState:
    live_segments: List[Segment] = field(default_factory=list)
    diar_segments: List[Tuple[float, float, str]] = field(
        default_factory=list
    )  # (start,end,speakerX)
    speaker_map: Dict[str, str] = field(default_factory=dict)  # SPEAKER_00 -> "신현택"
    summary: str = ""
    actions: List[str] = field(default_factory=list)
    schedule_note: str = ""
    diarization_enabled: bool = DIARIZATION_ENABLED_DEFAULT
    forced_speaker_name: Optional[str] = None  # manual override (combo)
    use_gpu: bool = USE_GPU_DEFAULT
    asr_model: str = DEFAULT_MODEL
    raw_audio_path: str = ""  # rolling wav path
    audio_time_elapsed: float = 0.0


# ---------- Utils ----------
def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


ACTION_VERBS = [
    "해야",
    "해주세요",
    "진행",
    "확인",
    "정리",
    "검토",
    "공유",
    "작성",
    "업로드",
    "보고",
    "회의",
    "예약",
    "훈련",
    "배포",
    "테스트",
    "구매",
    "설치",
]


def simple_summarize(segments: List[Segment], max_len=10) -> str:
    lines = [f"[{seg.speaker_name}] {seg.text}" for seg in segments if seg.text]
    return "\n".join(lines[-max_len:]) if lines else "요약할 내용이 없습니다."


def extract_actions(segments: List[Segment]) -> List[str]:
    acts = []
    for s in segments:
        if any(v in s.text for v in ACTION_VERBS):
            deadline = dateparser.parse(s.text, languages=["ko"])
            dstr = f" (기한: {deadline.strftime('%Y-%m-%d %H:%M')})" if deadline else ""
            acts.append(f"- [{s.speaker_name}] {s.text}{dstr}")
    # dedup
    uniq, seen = [], set()
    for a in acts:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


# ---------- Participant Dialog ----------
class ParticipantDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("회의 참가자 등록")
        self.resize(320, 150)
        layout = QVBoxLayout(self)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("참가자 이름 입력 (예: )")
        layout.addWidget(self.edit_name)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_name(self) -> str:
        return self.edit_name.text().strip()


# ---------- Audio Worker (Mic -> buffer -> chunk STT, rolling WAV) ----------
class AudioWorker(QObject):
    sig_transcript = pyqtSignal(object)  # Segment
    sig_status = pyqtSignal(str)

    def __init__(self, state: MeetingState):
        super().__init__()
        self.state = state
        self._stop = threading.Event()
        self.audio = None
        self.stream = None
        self._buf = io.BytesIO()
        self._buf_lock = threading.Lock()
        self._frames_elapsed = 0
        self.model = None

    def init_asr(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper 미설치")
        device = "cuda" if self.state.use_gpu else "cpu"
        compute = "float16" if self.state.use_gpu else "int8"
        self.sig_status.emit(f"Loading Whisper '{self.state.asr_model}' on {device}...")
        try:
            self.model = WhisperModel(
                self.state.asr_model, device=device, compute_type=compute
            )
        except Exception as e:
            self.sig_status.emit(f"ASR GPU 실패 -> CPU 재시도 ({e})")
            self.model = WhisperModel(
                self.state.asr_model, device="cpu", compute_type="int8"
            )
        self.sig_status.emit("ASR model ready.")

    def start(self):
        if pyaudio is None:
            raise RuntimeError("PyAudio 미설치")
        self._stop.clear()
        self.init_asr()

        # rolling wav file
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="raw_meeting_")
        os.close(fd)
        self.state.raw_audio_path = path
        # 프리헤더 기록
        sf.write(
            path,
            np.zeros((1, CHANNELS), dtype=np.float32),
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio.get_format_from_width(SAMPLE_WIDTH),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=int(SAMPLE_RATE * 0.2),  # 200ms
            stream_callback=self._on_audio,
        )
        self.stream.start_stream()
        threading.Thread(target=self._chunk_loop, daemon=True).start()
        self.sig_status.emit("Audio capture started.")

    def stop(self):
        self._stop.set()
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except:
            pass
        self.sig_status.emit("Audio capture stopped.")

    def _on_audio(self, in_data, frame_count, time_info, status):
        # buffer에 추가 + rolling 파일 append
        with self._buf_lock:
            self._buf.write(in_data)
            self._frames_elapsed += frame_count
            self.state.audio_time_elapsed += frame_count / SAMPLE_RATE

        # append to wav
        data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        data_np = data_np.reshape(-1, CHANNELS) if CHANNELS == 1 else data_np
        try:
            # append: soundfile는 append 모드가 없어 임시 합치기 → 파일 크기 커짐 감수
            # 경량화를 위해 실제 서비스에선 RAW PCM ring buffer + 주기적 리렌더 권장
            existing, sr = sf.read(
                self.state.raw_audio_path, dtype="float32", always_2d=True
            )
            sf.write(
                self.state.raw_audio_path,
                np.vstack([existing, data_np]),
                SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception as e:
            self.sig_status.emit(f"WAV append fail: {e}")

        return (None, pyaudio.paContinue)

    def _pull_chunk_wav(self) -> Optional[bytes]:
        with self._buf_lock:
            seconds = self._frames_elapsed / SAMPLE_RATE
            if seconds < CHUNK_SECONDS:
                return None
            need_frames = int(CHUNK_SECONDS * SAMPLE_RATE)
            raw = self._buf.getvalue()
            need_bytes = need_frames * SAMPLE_WIDTH * CHANNELS
            if len(raw) < need_bytes:
                return None
            chunk = raw[:need_bytes]
            remain = raw[need_bytes:]
            self._buf = io.BytesIO()
            self._buf.write(remain)
            self._frames_elapsed -= need_frames

        # to wav bytes
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = audio_np.reshape(-1, CHANNELS)
        mem = io.BytesIO()
        sf.write(mem, audio_np, SAMPLE_RATE, format="WAV")
        return mem.getvalue()

    def _chunk_loop(self):
        # 주기적으로 CHUNK_SECONDS 만큼을 STT
        while not self._stop.is_set():
            wav_bytes = self._pull_chunk_wav()
            if wav_bytes is None:
                time.sleep(0.1)
                continue
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                    tf.write(wav_bytes)
                    tf.flush()
                    segments, info = self.model.transcribe(
                        tf.name, language="ko", vad_filter=True
                    )
                    # CHUNK 내 상대 sec -> 전체 타임라인 sec(근사): 끝 시점 기준으로 정렬
                    # 여기서는 chunk가 연속이므로 state.audio_time_elapsed 값을 힌트로 사용 X
                    # 대신 모델 segment가 절대 시간이 없으므로 chunk 내 상대값만 표기
                    for s in segments:
                        seg = Segment(
                            start=s.start, end=s.end, text=(s.text or "").strip()
                        )
                        # 화자 지정: 강제 이름 우선, 아니면 diar 결과로 매핑
                        if self.state.forced_speaker_name:
                            seg.speaker_name = self.state.forced_speaker_name
                            seg.speaker_id = "FORCED"
                        else:
                            # diar 결과로 겹침 다수결
                            spk_id = self._infer_speaker_id(seg.start, seg.end)
                            seg.speaker_id = spk_id
                            seg.speaker_name = self.state.speaker_map.get(
                                spk_id, spk_id
                            )
                        self.sig_transcript.emit(seg)
            except Exception as e:
                self.sig_status.emit(f"STT error: {e}")
                time.sleep(0.2)

    def _infer_speaker_id(self, s: float, e: float) -> str:
        # CHUNK 상대시간과 diarization 절대시간의 정합성:
        # - 이 MVP에선 "chunk 구간 내 상대 시각"과 "전체 diar 구간"의 완벽 동기화는 어렵다.
        # - 실서비스: chunk 절대 오프셋(누적 재생시간 or frame index)을 추적해 매핑 권장.
        # - 여기서는 겹침 판단을 "상대"로 근사 (데모 목적)
        overlaps = [
            spk for (ds, de, spk) in self.state.diar_segments if not (e < ds or s > de)
        ]
        if not overlaps:
            return "Unknown"
        # 다수결
        return max(set(overlaps), key=overlaps.count)


# ---------- Diarization Worker (periodic) ----------
class DiarizationWorker(QObject):
    sig_status = pyqtSignal(str)
    sig_diar_done = pyqtSignal(list)  # list[(start,end,spk)]

    def __init__(self, state: MeetingState, interval_sec=30, window_sec=60):
        super().__init__()
        self.state = state
        self.interval = interval_sec
        self.window = window_sec
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        if not self.state.diarization_enabled:
            return
        if PyannotePipeline is None:
            self.sig_status.emit("pyannote 미설치로 diarization 비활성.")
            return
        token = os.getenv(HF_TOKEN_ENV, "")
        if not token:
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
                # 윈도우 구간 처리(전체 파일 대상 단순 처리)
                diar = pipeline(path)
                results = []
                for turn, _, speaker in diar.itertracks(yield_label=True):
                    results.append((turn.start, turn.end, speaker))
                # 중복 방지
                h = hash(tuple(results))
                if h != last_emit_hash:
                    self.sig_diar_done.emit(results)
                    last_emit_hash = h
                    self.sig_status.emit(f"Diar segments: {len(results)}")
            except Exception as e:
                self.sig_status.emit(f"Diarization error: {e}")


# ---------- RAG Store (Qdrant in-memory) ----------
class RagStore:
    def __init__(self):
        self.ok = False
        self.client = None
        self.collection = "meeting_ctx"
        self.embed_dim = 384
        self.model = None

        if not EMBED_AVAILABLE:
            return
        # try in-memory first
        try:
            self.client = QdrantClient(":memory:")
            self.ok = True
        except Exception:
            # try localhost
            try:
                self.client = QdrantClient(host="127.0.0.1", port=6333)
                self.ok = True
            except Exception as e:
                print("[WARN] Qdrant connection failed:", e)
                self.ok = False

        if self.ok:
            try:
                if SentenceTransformer is not None:
                    self.model = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    self.embed_dim = self.model.get_sentence_embedding_dimension()
                else:
                    # fallback: hashing embedding
                    self.model = None
                    self.embed_dim = 256

                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embed_dim, distance=Distance.COSINE
                    ),
                )
            except Exception as e:
                print("[WARN] Create collection failed:", e)
                self.ok = False

        self._id_seq = 1

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            # very simple hashing fallback
            vecs = []
            for t in texts:
                v = np.zeros(self.embed_dim, dtype=np.float32)
                for i, ch in enumerate(t.encode("utf-8")):
                    v[i % self.embed_dim] += (ch % 13) / 13.0
                # normalize
                n = np.linalg.norm(v) + 1e-9
                vecs.append((v / n).tolist())
            return vecs
        em = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if isinstance(em, np.ndarray):
            return em.tolist()
        return em

    def upsert_segments(self, segs: List[Segment]):
        if not self.ok:
            return
        payloads = []
        texts = []
        ids = []
        for s in segs:
            text = f"[{s.speaker_name}] {s.text}"
            texts.append(text)
            ids.append(self._id_seq)
            payloads.append(
                {
                    "speaker_id": s.speaker_id,
                    "speaker_name": s.speaker_name,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
            )
            self._id_seq += 1
        vecs = self._embed(texts)
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vecs, payloads)
        ]
        try:
            self.client.upsert(self.collection, points=points)
        except Exception as e:
            print("[WARN] RAG upsert failed:", e)

    def search(self, query: str, topk=5) -> List[Dict]:
        if not self.ok:
            return []
        vec = self._embed([query])[0]
        try:
            res = self.client.search(self.collection, query_vector=vec, limit=topk)
            out = []
            for r in res:
                pl = r.payload or {}
                pl["_score"] = r.score
                out.append(pl)
            return out
        except Exception as e:
            print("[WARN] RAG search failed:", e)
            return []


# ---------- Adapter (QLoRA) Manager (optional) ----------
class AdapterManager:
    def __init__(self):
        self.available = PEFT_AVAILABLE
        self.base_model_id = None
        self.base_model = None
        self.tokenizer = None
        self.loaded_adapters: Dict[str, PeftModel] = {}
        self.active_adapter = None

    def load_base(self, base_model_id: str = "EleutherAI/pythia-410m"):
        if not self.available:
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
            self.base_model_id = base_model_id
            return True
        except Exception as e:
            print("[WARN] load_base failed:", e)
            return False

    def load_adapter(self, name: str, adapter_path: str):
        if not self.available or self.base_model is None:
            return False
        try:
            peft_m = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.loaded_adapters[name] = peft_m
            return True
        except Exception as e:
            print("[WARN] load_adapter failed:", e)
            return False

    def set_active(self, name: Optional[str]):
        self.active_adapter = name if name in self.loaded_adapters else None

    def respond(self, prompt: str) -> str:
        # 데모: 어댑터 활성화 여부에 따라 톤만 살짝 바꿈(실제 생성은 생략)
        if self.active_adapter:
            return f"(어댑터:{self.active_adapter}) {prompt} -> 친근하고 공손한 톤으로 답변합니다."
        return f"{prompt} -> 기본 톤 답변(데모)."


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    sig_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persona-AI 실시간 회의 비서 (PyQt)")
        self.resize(1280, 860)

        self.state = MeetingState()
        self.audio_worker = AudioWorker(self.state)
        self.audio_worker.sig_transcript.connect(self.on_segment)
        self.audio_worker.sig_status.connect(self.on_status)

        self.diar_worker = DiarizationWorker(self.state)
        self.diar_worker.sig_status.connect(self.on_status)
        self.diar_worker.sig_diar_done.connect(self.on_diar_done)

        self.rag = RagStore()
        self.adapter = AdapterManager()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_live_tab()
        self._build_timeline_tab()
        self._build_qa_tab()
        self._build_action_tab()
        self._build_settings_tab()
        self._apply_theme()

        # timer: preview summary
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(1000)

    # ----- UI Builders -----
    def _build_live_tab(self):
        self.live_root = QWidget()
        L = QVBoxLayout(self.live_root)

        # Top controls
        bar = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_sum = QPushButton("Summarize")
        self.btn_add2rag = QPushButton("Index to RAG")
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addStretch(1)
        bar.addWidget(self.btn_sum)
        bar.addWidget(self.btn_add2rag)
        L.addLayout(bar)

        # Mid controls (speaker / diar on/off)
        mid = QHBoxLayout()
        mid.addWidget(QLabel("Forced Speaker:"))
        self.cmb_forced = QComboBox()
        self.cmb_forced.addItem("None")
        mid.addWidget(self.cmb_forced)

        self.chk_diar = QCheckBox("Auto Diarization (pyannote)")
        self.chk_diar.setChecked(self.state.diarization_enabled)
        mid.addWidget(self.chk_diar)

        L.addLayout(mid)

        # Split view
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        Lv = QVBoxLayout(left)
        self.list_chat = QListWidget()
        Lv.addWidget(self.list_chat)
        splitter.addWidget(left)

        right = QWidget()
        Rv = QVBoxLayout(right)
        Rv.addWidget(QLabel("Status"))
        self.txt_status = QPlainTextEdit()
        self.txt_status.setReadOnly(True)
        Rv.addWidget(self.txt_status)
        Rv.addWidget(QLabel("Preview (Summary)"))
        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        Rv.addWidget(self.txt_preview)
        splitter.addWidget(right)
        splitter.setSizes([900, 380])

        L.addWidget(splitter)
        self.tabs.addTab(self.live_root, "Live")

        # events
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_sum.clicked.connect(self.on_summarize)
        self.btn_add2rag.clicked.connect(self.on_index_to_rag)
        self.cmb_forced.currentTextChanged.connect(self.on_forced_changed)
        self.chk_diar.stateChanged.connect(self.on_diar_toggle)

    def _build_timeline_tab(self):
        self.timeline_root = QWidget()
        L = QVBoxLayout(self.timeline_root)
        self.timeline = QListWidget()
        L.addWidget(self.timeline)
        self.tabs.addTab(self.timeline_root, "Timeline")

    def _build_qa_tab(self):
        root = QWidget()
        L = QVBoxLayout(root)
        top = QHBoxLayout()
        self.edit_q = QLineEdit()
        self.edit_q.setPlaceholderText("질문 입력 (RAG + QLoRA 톤)")
        self.btn_ans = QPushButton("Answer")
        top.addWidget(self.edit_q)
        top.addWidget(self.btn_ans)
        L.addLayout(top)

        self.cmb_adapter = QComboBox()
        self.cmb_adapter.addItem("None")
        self.btn_load_base = QPushButton("Load Base (QLoRA)")
        self.btn_add_adapter = QPushButton("Add Adapter…")
        tool = QHBoxLayout()
        tool.addWidget(QLabel("Tone Adapter:"))
        tool.addWidget(self.cmb_adapter)
        tool.addWidget(self.btn_load_base)
        tool.addWidget(self.btn_add_adapter)
        L.addLayout(tool)

        self.txt_ans = QTextEdit()
        self.txt_ans.setReadOnly(True)
        L.addWidget(self.txt_ans)

        self.tabs.addTab(root, "Q&A")

        self.btn_ans.clicked.connect(self.on_answer)
        self.btn_load_base.clicked.connect(self.on_load_base)
        self.btn_add_adapter.clicked.connect(self.on_add_adapter)
        self.cmb_adapter.currentTextChanged.connect(self.on_adapter_changed)

    def _build_action_tab(self):
        root = QWidget()
        L = QVBoxLayout(root)

        L.addWidget(QLabel("회의 전체요약"))
        self.txt_summary = QTextEdit()
        L.addWidget(self.txt_summary)

        L.addWidget(QLabel("Action Items"))
        self.txt_actions = QTextEdit()
        L.addWidget(self.txt_actions)

        # Schedule area with QDateTimeEdit
        row = QHBoxLayout()
        row.addWidget(QLabel("다음 회의 시작"))
        self.dt_start = QDateTimeEdit()
        self.dt_start.setDateTime(QDateTime.currentDateTime().addDays(7))
        self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm")
        row.addWidget(self.dt_start)

        row.addWidget(QLabel("종료"))
        self.dt_end = QDateTimeEdit()
        self.dt_end.setDateTime(QDateTime.currentDateTime().addDays(7).addSecs(3600))
        self.dt_end.setDisplayFormat("yyyy-MM-dd HH:mm")
        row.addWidget(self.dt_end)

        self.btn_sched_memo = QPushButton("Make Schedule Memo")
        row.addWidget(self.btn_sched_memo)
        L.addLayout(row)

        L.addWidget(QLabel("다음 회의 메모"))
        self.txt_sched = QTextEdit()
        L.addWidget(self.txt_sched)

        self.tabs.addTab(root, "Action & Schedule")
        self.btn_sched_memo.clicked.connect(self.on_make_schedule)

    def _build_settings_tab(self):
        root = QWidget()
        F = QFormLayout(root)
        self.cmb_asr = QComboBox()
        for m in ["tiny", "base", "small", "medium", "large-v3"]:
            self.cmb_asr.addItem(m)
        self.cmb_asr.setCurrentText(DEFAULT_MODEL)
        self.chk_gpu = QCheckBox("Use GPU if available")
        self.chk_gpu.setChecked(USE_GPU_DEFAULT)
        self.chk_diar2 = QCheckBox("Auto Diarization")
        self.chk_diar2.setChecked(DIARIZATION_ENABLED_DEFAULT)
        self.edit_hf = QLineEdit()
        self.edit_hf.setPlaceholderText(f"{HF_TOKEN_ENV} (HuggingFace token)")
        self.btn_add_participant = QPushButton("참가자 추가")
        self.cmb_map_id = QComboBox()
        self.cmb_map_id.setEditable(False)
        self.cmb_map_name = QComboBox()
        self.cmb_map_name.setEditable(False)
        self.btn_bind_map = QPushButton("스피커ID ↔ 이름 매핑")

        F.addRow("Whisper Model", self.cmb_asr)
        F.addRow("", self.chk_gpu)
        F.addRow("Auto Diarization", self.chk_diar2)
        F.addRow("HF Token", self.edit_hf)
        F.addRow("", self.btn_add_participant)
        F.addRow(QLabel("pyannote SpeakerID"), self.cmb_map_id)
        F.addRow(QLabel("Participant Name"), self.cmb_map_name)
        F.addRow("", self.btn_bind_map)
        self.tabs.addTab(root, "Settings")

        self.btn_add_participant.clicked.connect(self.on_add_participant)
        self.btn_bind_map.clicked.connect(self.on_bind_map)
        self.chk_diar2.stateChanged.connect(self.on_diar_toggle_settings)

        # preload default names
        for n in ["신현택", "박길실", "조진경"]:
            self.cmb_forced.addItem(n)
            self.cmb_map_name.addItem(n)

    def _apply_theme(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{ background-color: {THEME['bg']}; }}
            QTabWidget::pane {{ border: 2px solid {THEME['pane']}; }}
            QPushButton {{
                background-color: {THEME['btn']};
                border: 1px solid {THEME['btn_border']};
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {THEME['btn_hover']}; }}
            QListWidget, QTextEdit, QPlainTextEdit {{
                background-color: {THEME['light_bg']};
                border: 1px solid {THEME['pane']};
            }}
            QLineEdit, QComboBox, QDateTimeEdit {{
                background-color: #ffffff;
                border: 1px solid {THEME['pane']};
                border-radius: 6px;
                padding: 4px 6px;
            }}
        """
        )

    # ----- Events -----
    def on_start(self):
        # apply settings
        self.state.use_gpu = self.chk_gpu.isChecked()
        self.state.asr_model = self.cmb_asr.currentText()
        self.state.diarization_enabled = (
            self.chk_diar.isChecked() or self.chk_diar2.isChecked()
        )
        tok = self.edit_hf.text().strip()
        if tok:
            os.environ[HF_TOKEN_ENV] = tok

        try:
            self.audio_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Start failed: {e}")
            return

        # diar thread
        if self.state.diarization_enabled:
            self.diar_worker.start()

        self.on_status("Started.")

    def on_stop(self):
        try:
            self.audio_worker.stop()
            self.diar_worker.stop()
        except:
            pass
        self.on_status("Stopped.")

    def on_summarize(self):
        self.state.summary = simple_summarize(self.state.live_segments, max_len=12)
        self.state.actions = extract_actions(self.state.live_segments)
        self.txt_summary.setText(self.state.summary)
        self.txt_actions.setText(
            "\n".join(self.state.actions) if self.state.actions else "(액션아이템 없음)"
        )
        QMessageBox.information(self, "Done", "요약/액션아이템 생성 완료")

    def on_index_to_rag(self):
        if not self.rag.ok:
            QMessageBox.warning(self, "RAG", "Qdrant 사용 불가(미설치/연결 실패).")
            return
        # 최근 50줄 인덱싱 (중복 허용 데모)
        self.rag.upsert_segments(self.state.live_segments[-50:])
        QMessageBox.information(self, "RAG", "최근 발언을 RAG 인덱싱했습니다.")

    def on_forced_changed(self, text):
        self.state.forced_speaker_name = None if (text == "None") else text

    def on_diar_toggle(self):
        self.state.diarization_enabled = self.chk_diar.isChecked()
        self.chk_diar2.setChecked(self.state.diarization_enabled)

    def on_diar_toggle_settings(self):
        self.state.diarization_enabled = self.chk_diar2.isChecked()
        self.chk_diar.setChecked(self.state.diarization_enabled)

    def on_add_participant(self):
        dlg = ParticipantDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name = dlg.get_name()
            if not name:
                return
            # 새 pyannote speakerID 예약
            spk_id = f"SPEAKER_{len(self.state.speaker_map):02d}"
            self.state.speaker_map[spk_id] = name
            # UI에 반영
            self.cmb_map_id.addItem(spk_id)
            if self.cmb_forced.findText(name) < 0:
                self.cmb_forced.addItem(name)
            if self.cmb_map_name.findText(name) < 0:
                self.cmb_map_name.addItem(name)
            QMessageBox.information(self, "등록 완료", f"{spk_id} → {name}")

    def on_bind_map(self):
        spk_id = self.cmb_map_id.currentText().strip()
        name = self.cmb_map_name.currentText().strip()
        if not spk_id or not name:
            QMessageBox.warning(self, "매핑", "스피커ID와 이름을 선택하세요.")
            return
        self.state.speaker_map[spk_id] = name
        QMessageBox.information(self, "매핑 완료", f"{spk_id} → {name}")

    def on_make_schedule(self):
        s = self.dt_start.dateTime().toString("yyyy-MM-dd HH:mm")
        e = self.dt_end.dateTime().toString("HH:mm")
        memo = f"다음 회의: {s} ~ {e}\n참석자: {', '.join(sorted(set([seg.speaker_name for seg in self.state.live_segments if seg.speaker_name!='Unknown'])))}\n안건: 액션아이템 점검"
        self.state.schedule_note = memo
        self.txt_sched.setText(memo)
        QMessageBox.information(self, "메모 생성", "다음 회의 메모를 작성했습니다.")

    def on_answer(self):
        q = self.edit_q.text().strip()
        if not q:
            return
        # RAG search
        ctx = self.rag.search(q, topk=5) if self.rag.ok else []
        lines = [f"- [{c.get('speaker_name','?')}] {c.get('text','')}" for c in ctx]
        ctx_block = "\n".join(lines) if lines else "(근거 없음)"

        # Adapter tone
        ans = self.adapter.respond(f"Q: {q}")
        self.txt_ans.setText(f"{ans}\n\n[근거]\n{ctx_block}")

    def on_load_base(self):
        if not self.adapter.available:
            QMessageBox.warning(self, "QLoRA", "transformers/peft 미설치로 비활성.")
            return
        ok = self.adapter.load_base()
        QMessageBox.information(
            self, "QLoRA", "Base loaded." if ok else "Base load 실패."
        )

    def on_add_adapter(self):
        if not self.adapter.available or self.adapter.base_model is None:
            QMessageBox.warning(self, "QLoRA", "Base 모델 먼저 로드하세요.")
            return
        # 데모: 로컬 디렉터리 path를 환경변수로 받을 수도 있음
        # 여기선 간단히 input dialog 대신 고정 키로 등록
        name = f"adapter_{self.cmb_adapter.count()}"
        # 사용자는 실제 어댑터 경로를 아래에 넣어야 동작
        adapter_path = os.getenv("QLORA_ADAPTER_PATH", "")
        ok = self.adapter.load_adapter(name, adapter_path) if adapter_path else False
        self.cmb_adapter.addItem(name)
        QMessageBox.information(
            self, "QLoRA", f"Adapter '{name}' 추가 {'성공' if ok else '(더미 등록)'}."
        )

    def on_adapter_changed(self, name):
        self.adapter.set_active(None if name == "None" else name)

    # ----- Signals -----
    def on_status(self, msg: str):
        self.txt_status.appendPlainText(f"{now_str()}  {msg}")

    def on_segment(self, seg: Segment):
        # UI append
        self.state.live_segments.append(seg)
        live = f"[{seg.speaker_name}] {seg.text}"
        self.list_chat.addItem(QListWidgetItem(live))
        self.list_chat.scrollToBottom()

        tline = f"{fmt_time(seg.start)}~{fmt_time(seg.end)} | {seg.speaker_name}: {seg.text}"
        self.timeline.addItem(QListWidgetItem(tline))
        self.timeline.scrollToBottom()

    def on_diar_done(self, results: List[Tuple[float, float, str]]):
        self.state.diar_segments = results
        # SpeakerID 콤보 갱신
        existing = set(self._combo_items(self.cmb_map_id))
        for _, _, spk in results:
            if spk not in existing:
                self.cmb_map_id.addItem(spk)
                existing.add(spk)
        self.on_status(f"Diarization updated ({len(results)} segments).")

    def _combo_items(self, combo: QComboBox) -> List[str]:
        return [combo.itemText(i) for i in range(combo.count())]

    def _refresh_preview(self):
        self.txt_preview.setPlainText(simple_summarize(self.state.live_segments))


# ---------- main ----------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
