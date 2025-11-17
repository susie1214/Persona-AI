# core/stt_stream.py
import queue, threading, time, json, os
from faster_whisper import WhisperModel


class STTWorker(threading.Thread):
    def __init__(self, meeting_id, lang="ko", model_size="small", device="cpu", compute_type="int8"):
        super().__init__(daemon=True)
        self.meeting_id = meeting_id
        self.lang = lang
        self.q = queue.Queue()
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # CUDA 오류 시 CPU로 폴백
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        os.makedirs("data/meetings", exist_ok=True)
        self.out_path = f"data/meetings/{meeting_id}.jsonl"
        self._running = True

    def push_audio_chunk(self, pcm16_bytes):
        self.q.put(pcm16_bytes)

    def run(self):
        with open(self.out_path, "a", encoding="utf-8") as f:
            while self._running:
                try:
                    audio = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                # 데모: chunk 단위로 곧장 디코드(실서비스는 ring buffer+VAD)
                segments, _ = self.model.transcribe(audio=audio, language=self.lang)
                for seg in segments:
                    rec = {
                        "ts": [seg.start, seg.end],
                        "speaker": "S?",
                        "text": seg.text,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stop(self):
        self._running = False
