#!/usr/bin/env python3
"""
화자 분리 및 음성 인식에 필요한 모델들을 다운로드하는 스크립트
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml


def create_directories():
    """필요한 디렉토리 생성"""
    base_dir = Path("C:/Persona-AI/models")
    diart_dir = base_dir / "diart_model"
    whisper_dir = base_dir / "whisper-small-ct2"

    diart_dir.mkdir(parents=True, exist_ok=True)
    whisper_dir.mkdir(parents=True, exist_ok=True)

    return diart_dir, whisper_dir


def download_pyannote_models(diart_dir):
    """Pyannote 화자 분리 모델들 다운로드"""
    print("[INFO] Downloading pyannote segmentation model...")
    snapshot_download(
        repo_id="pyannote/segmentation-3.0",
        local_dir=str(diart_dir / "segmentation-3.0"),
        allow_patterns=["*.bin", "*.json"],
    )

    print("[INFO] Downloading wespeaker embedding model...")
    snapshot_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        local_dir=str(diart_dir / "wespeaker-voxceleb-resnet34-LM"),
        allow_patterns=["*.bin", "*.json"],
    )


def download_whisper_model(whisper_dir):
    """Faster-Whisper 모델 다운로드"""
    print("[INFO] Downloading Faster-Whisper model...")
    snapshot_download(
        repo_id="guillaumekln/faster-whisper-small", local_dir=str(whisper_dir)
    )


def create_config_file(diart_dir):
    """Pyannote 설정 파일 생성"""
    config = {
        "params": {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7045654963945799,
            },
            "embedding": {"batch_size": 32, "window": "sliding"},
            "segmentation": {"min_duration_off": 0.09791355370864306},
        }
    }

    config_path = diart_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] Config file created: {config_path}")


def main():
    """메인 함수"""
    print("=== AI 모델 다운로드 시작 ===")

    try:
        # 1. 디렉토리 생성
        diart_dir, whisper_dir = create_directories()
        print(f"[INFO] Directories created: {diart_dir}, {whisper_dir}")

        # 2. Pyannote 모델들 다운로드
        download_pyannote_models(diart_dir)

        # 3. Whisper 모델 다운로드
        download_whisper_model(whisper_dir)

        # 4. 설정 파일 생성
        create_config_file(diart_dir)

        print("\n=== 다운로드 완료! ===")
        print("모든 모델이 성공적으로 다운로드되었습니다.")
        print(f"Pyannote 모델 위치: {diart_dir}")
        print(f"Whisper 모델 위치: {whisper_dir}")

    except Exception as e:
        print(f"[ERROR] 다운로드 중 오류 발생: {e}")
        print("인터넷 연결과 디스크 용량을 확인해주세요.")


if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    print("필요한 패키지가 설치되어 있는지 확인하세요:")
    print("pip install huggingface_hub PyYAML")
    print()

    main()
