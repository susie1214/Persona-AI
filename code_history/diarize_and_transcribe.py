# diarize_and_transcribe.py  (pyannote + faster-whisper only)
import os, re, json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MERGED_AUDIO = os.getenv("MERGED_AUDIO", "./output/meeting_merged.wav")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # CPU 기본

SPEAKER_MAP_ENV = os.getenv("SPEAKER_MAP", "진경,현택,교수님").split(",")
NAME_MAP = {f"SPEAKER_{i+1}": name.strip() for i, name in enumerate(SPEAKER_MAP_ENV)}

os.makedirs("./output", exist_ok=True)


# ------------------ 1) 화자 분리 (pyannote) ------------------
def run_diarization_with_pyannote(audio_path: str, hf_token: str):
    from huggingface_hub import login

    if hf_token:
        try:
            login(token=hf_token)
        except Exception:
            pass

    from pyannote.audio import Pipeline

    # 3.1 파이프라인은 내부적으로 segmentation/embedding 불러와 처리
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    ann = pipe(audio_path)
    segments = []
    for segment, _, label in ann.itertracks(yield_label=True):
        segments.append((float(segment.start), float(segment.end), str(label)))
    return segments


# ------------------ 2) STT (faster-whisper) ------------------
# (파일 상단의 WhisperModel import 아래로 교체)
from faster_whisper import WhisperModel


def build_whisper():
    model_name = os.getenv("WHISPER_MODEL", "medium")
    device = os.getenv("WHISPER_DEVICE", "auto")
    ctype = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    try:
        print(f"[INFO] init whisper: model={model_name}, device={device}, type={ctype}")
        return WhisperModel(model_name, device=device, compute_type=ctype)
    except Exception as e:
        print("[WARN] GPU init failed → fallback to CPU int8:", e)
        return WhisperModel(model_name, device="cpu", compute_type="int8")


whisper = build_whisper()


whisper = WhisperModel(
    WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
)


def transcribe_clip(wav: np.ndarray):
    segments, _ = whisper.transcribe(
        wav.astype(np.float32),
        language="ko",
        vad_filter=True,
        beam_size=1,
        word_timestamps=True,
    )
    text_parts, words = [], []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start) if w.start is not None else None,
                        "end": float(w.end) if w.end is not None else None,
                        "prob": getattr(w, "probability", None),
                    }
                )
    return "".join(text_parts).strip(), words


def hhmmss(x):
    x = max(0, x)
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------ 3) 메인 ------------------
def main():
    # 0) 오디오 로드
    wav, sr = sf.read(MERGED_AUDIO, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]

    # 1) 화자 분리
    print("[INFO] Running diarization (pyannote)…")
    diar = run_diarization_with_pyannote(MERGED_AUDIO, HF_TOKEN)
    diar.sort(key=lambda x: x[0])
    print(f"[INFO] turns: {len(diar)}")

    # 2) 턴별 STT
    rows = []
    for i, (start, end, label) in enumerate(diar, 1):
        s_i = max(0, int(start * sr))
        e_i = min(len(wav), int(end * sr))
        if e_i <= s_i:
            continue
        clip = wav[s_i:e_i]
        text, words = transcribe_clip(clip)
        print(f"  [{i}/{len(diar)}] {label} {start:.2f}-{end:.2f}s → {len(text)} chars")
        if not text:
            continue
        speaker_name = NAME_MAP.get(label, label)
        rows.append(
            {
                "speaker": speaker_name,
                "start": float(start),
                "end": float(end),
                "text": text,
                "words": words,
            }
        )

    # 3) 저장
    rows = sorted(rows, key=lambda r: r["start"])
    json_path = "./output/transcript.json"
    csv_path = "./output/transcript.csv"
    txt_path = "./output/transcript.txt"
    sum_path = "./output/summary_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(
        [
            {
                "speaker": r["speaker"],
                "start": r["start"],
                "end": r["end"],
                "text": r["text"],
            }
            for r in rows
        ]
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                f"[{hhmmss(r['start'])}–{hhmmss(r['end'])}] {r['speaker']}: {r['text']}\n"
            )

    actions = [
        r
        for r in rows
        if re.search(r"(하겠습니다|처리|담당|일정|까지|진행하)", r["text"])
    ]
    decisions = [r for r in rows if re.search(r"(결정|정하|합의|승인|채택)", r["text"])]
    issues = [
        r for r in rows if re.search(r"(문제|이슈|버그|지연|리스크|차질)", r["text"])
    ]

    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("# 회의 요약 보고서\n\n")
        f.write("## Action Items\n")
        if actions:
            [f.write(f"- ({r['speaker']}) {r['text']}\n") for r in actions]
        else:
            f.write("- (없음)\n")
        f.write("\n## Decisions\n")
        if decisions:
            [f.write(f"- ({r['speaker']}) {r['text']}\n") for r in decisions]
        else:
            f.write("- (없음)\n")
        f.write("\n## Issues\n")
        if issues:
            [f.write(f"- ({r['speaker']}) {r['text']}\n") for r in issues]
        else:
            f.write("- (없음)\n")

    # 4) 시각화
    # Waveform
    t = np.arange(len(wav)) / sr
    plt.figure(figsize=(12, 3))
    plt.plot(t, wav, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig("./output/waveform.png", dpi=160)
    plt.close()

    # Spectrogram
    import librosa.display as lbd

    S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    lbd.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig("./output/spectrogram.png", dpi=160)
    plt.close()

    # Timeline(Gantt)
    dur = df.assign(duration=df["end"] - df["start"])
    speakers = list(dur.groupby("speaker").size().index)
    colors = {spk: plt.cm.tab20(i % 20) for i, spk in enumerate(speakers)}
    plt.figure(figsize=(12, 2 + 0.5 * len(speakers)))
    for i, spk in enumerate(speakers):
        segs = df[df["speaker"] == spk]
        for _, r in segs.iterrows():
            plt.hlines(i, r["start"], r["end"], colors=[colors[spk]], linewidth=6)
    plt.yticks(range(len(speakers)), speakers)
    plt.xlabel("Time (s)")
    plt.title("Timeline (by Speaker)")
    plt.tight_layout()
    plt.savefig("./output/timeline_gantt.png", dpi=180)
    plt.close()

    # Speaking time
    agg = dur.groupby("speaker")["duration"].sum().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    agg.plot(kind="bar")
    plt.ylabel("총 발화 시간 (초)")
    plt.title("스피커별 총 발화 시간")
    plt.tight_layout()
    plt.savefig("./output/speaking_time.png", dpi=160)
    plt.close()

    print("\n[DONE] files:")
    for p in [
        json_path,
        csv_path,
        txt_path,
        sum_path,
        "./output/waveform.png",
        "./output/spectrogram.png",
        "./output/timeline_gantt.png",
        "./output/speaking_time.png",
    ]:
        print(" -", os.path.abspath(p))


if __name__ == "__main__":
    main()
