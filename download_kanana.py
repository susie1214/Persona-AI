#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kakao Kanana-1.5-v-3b-instruct 모델 다운로드 스크립트
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_kanana_model():
    """
    Kakao Kanana-1.5-v-3b-instruct 모델을 로컬에 다운로드
    """
    model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    local_path = Path("models") / "kanana-1.5-2.1b-instruct"

    print(f"📥 Kanana 모델 다운로드 시작...")
    print(f"   모델: {model_id}")
    print(f"   저장 위치: {local_path}")
    print()

    # 디렉토리 생성
    os.makedirs(local_path.parent, exist_ok=True)

    # 이미 다운로드되어 있는지 확인
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 모델이 이미 다운로드되어 있습니다: {local_path}")
        return str(local_path)

    try:
        # HuggingFace에서 다운로드
        print(f"🔄 다운로드 중... (모델 크기: ~6GB, 시간이 걸릴 수 있습니다)")

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"\n✅ 다운로드 완료!")
        print(f"   저장 위치: {local_path}")
        return str(local_path)

    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print(f"\n해결 방법:")
        print(f"1. 인터넷 연결 확인")
        print(f"2. HuggingFace 토큰 설정 (필요 시):")
        print(f"   export HF_TOKEN=your_token_here")
        print(f"3. 디스크 공간 확인 (최소 10GB 필요)")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Kakao Kanana-1.5-v-3b-instruct 모델 다운로드")
    print("=" * 60)
    print()

    result = download_kanana_model()

    if result:
        print()
        print("=" * 60)
        print("✅ 모든 작업 완료!")
        print("=" * 60)
        print()
        print("다음 명령으로 모델을 사용할 수 있습니다:")
        print()
        print("  from core.llm_kanana import KananaLLM")
        print("  llm = KananaLLM(use_4bit=True)")
        print("  response = llm.complete('안녕하세요')")
        print()
        print("또는 챗봇에서:")
        print("  kanana:kakaocorp/kanana-1.5-v-3b-instruct")
        print()
    else:
        print()
        print("=" * 60)
        print("❌ 다운로드 실패")
        print("=" * 60)
        exit(1)
