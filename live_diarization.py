import os
import sys
import argparse
import platform
import datetime
import torch

from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter

DEBUG = True
DEFAULT_SR = 16000

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log_env(dev):
    if DEBUG:
        print(f"[DEBUG] OS: {platform.system()}")
        print(f"[DEBUG] Torch device: {dev}")

def ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def run_file_diarization(file_path: str, out_path: str | None = None, sr: int = DEFAULT_SR, dev=None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    src = FileAudioSource(file_path, sample_rate=sr)
    pipe = SpeakerDiarization()
    if dev is not None:
        pipe.to(dev)

    # 출력 경로 결정
    if out_path is None:
        stem, _ = os.path.splitext(file_path)
        out_path = stem + ".rttm"
    ensure_parent(out_path)

    inf = StreamingInference(pipe, src, do_plot=False)
    inf.attach_observers(RTTMWriter(src.uri, out_path))

    if DEBUG:
        print(f"[DEBUG] File diarization 시작: {file_path}")
    _ = inf()  # 블로킹; 완료되면 RTTM이 저장됨
    if DEBUG:
        print(f"[DEBUG] RTTM 저장 완료: {out_path}")
    return out_path

def run_microphone_diarization(device_id: int | None = None, out_dir: str = "./outputs", sr: int = DEFAULT_SR, dev=None):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"microphone_{ts}.rttm")

    src = MicrophoneAudioSource(device=device_id)
    pipe = SpeakerDiarization()
    if dev is not None:
        pipe.to(dev)

    inf = StreamingInference(pipe, src, do_plot=False)
    inf.attach_observers(RTTMWriter(src.uri, out_path))

    print("마이크 화자 분리 시작 (종료: Ctrl+C)")
    try:
        _ = inf()  # 마이크 스트림: Ctrl+C로 종료
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        print(f"RTTM 저장 위치: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Speaker diarization with diart")
    sub = parser.add_subparsers(dest="mode", required=True)

    pm = sub.add_parser("mic", help="마이크 실시간 화자 분리")
    pm.add_argument("--device-id", type=int, default=None, help="사운드 디바이스 ID (기본값: 기본 마이크)")
    pm.add_argument("--out-dir", default="./outputs", help="RTTM 저장 폴더")
    pm.add_argument("--sr", type=int, default=DEFAULT_SR, help="샘플레이트(Hz)")

    pf = sub.add_parser("file", help="오디오 파일 화자 분리")
    pf.add_argument("path", help="입력 오디오 경로(.wav 등)")
    pf.add_argument("--out", default=None, help="출력 RTTM 경로(미지정 시 입력 파일명 기반)")
    pf.add_argument("--sr", type=int, default=DEFAULT_SR, help="샘플레이트(Hz)")

    args = parser.parse_args()
    dev = pick_device()
    log_env(dev)

    if args.mode == "mic":
        run_microphone_diarization(device_id=args.device_id, out_dir=args.out_dir, sr=args.sr, dev=dev)
    else:
        run_file_diarization(args.path, out_path=args.out, sr=args.sr, dev=dev)

if __name__ == "__main__":
    main()