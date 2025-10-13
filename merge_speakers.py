#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
중복 화자 병합 스크립트

이미 생성된 중복 화자들을 자동으로 병합합니다.
둘이서 대화했는데 12명의 speaker가 생성된 경우 이 스크립트를 실행하세요.
"""

from core.speaker import SpeakerManager

def main():
    print("=" * 60)
    print("중복 화자 병합 스크립트")
    print("=" * 60)

    # SpeakerManager 로드
    manager = SpeakerManager()

    print(f"\n현재 화자 수: {len(manager.speakers)}")
    print("\n현재 화자 목록:")
    for speaker in manager.speakers:
        emb_count = len(speaker.embeddings)
        print(f"  - {speaker.speaker_id} ({speaker.display_name}): 임베딩 {emb_count}개")

    if len(manager.speakers) < 2:
        print("\n병합할 화자가 없습니다.")
        return

    # 유사도 임계값 선택
    print("\n유사도 임계값을 선택하세요:")
    print("  1. 0.60 (매우 관대 - 많은 화자 병합)")
    print("  2. 0.70 (권장 - 적절한 병합)")
    print("  3. 0.80 (엄격 - 확실한 것만 병합)")

    choice = input("\n선택 (1-3, 엔터는 기본값 2): ").strip()

    threshold_map = {
        "1": 0.60,
        "2": 0.70,
        "3": 0.80,
        "": 0.70  # 기본값
    }

    threshold = threshold_map.get(choice, 0.70)
    print(f"\n임계값 {threshold}로 병합을 시작합니다...\n")

    # 병합 실행
    merged_count = manager.merge_similar_speakers(similarity_threshold=threshold)

    print("\n" + "=" * 60)
    if merged_count > 0:
        print(f"✅ {merged_count}개 화자가 병합되었습니다!")
        print(f"현재 화자 수: {len(manager.speakers)}")
        print("\n병합 후 화자 목록:")
        for speaker in manager.speakers:
            emb_count = len(speaker.embeddings)
            print(f"  - {speaker.speaker_id} ({speaker.display_name}): 임베딩 {emb_count}개")
    else:
        print("병합할 유사한 화자가 없습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
