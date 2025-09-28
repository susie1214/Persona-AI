import os
import glob
import numpy as np
import soundfile as sf
import librosa

# 입력/출력 폴더
AUDIO_DIR = "./audio_in"
OUT = "./output/meeting_merged.wav"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# mp3, wav 파일 모두 수집
files = sorted(
    glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    + glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
)

if not files:
    raise FileNotFoundError(f"No audio files found in {AUDIO_DIR}")

print(f"[INFO] Found {len(files)} files")

merged = []
target_sr = 16000  # Whisper 호환 샘플레이트
for i, f in enumerate(files, 1):
    # librosa로 로드 (자동으로 wav/mp3 읽기 지원)
    y, sr = librosa.load(f, sr=target_sr, mono=True)
    merged.append(y)
    print(f"[{i}/{len(files)}] Loaded {os.path.basename(f)} ({len(y)/target_sr:.1f}s)")

# 하나의 numpy array로 연결
final_audio = np.concatenate(merged, axis=0)

# WAV 저장
sf.write(OUT, final_audio, target_sr)
print(f"[DONE] Saved merged audio → {OUT} ({len(final_audio)/target_sr:.1f}s)")
