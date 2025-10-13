#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaker별 RAG 검색 테스트 스크립트
"""

from core.rag_store import RagStore
from core.speaker import SpeakerManager
import json

def test_rag_basic():
    """기본 RAG 기능 테스트"""
    print("=" * 60)
    print("1. RAG Store 초기화 테스트")
    print("=" * 60)

    rag = RagStore(persist_path="./qdrant_db")

    if not rag.ok:
        print("[ERROR] RAG store 초기화 실패!")
        return False

    print("[OK] RAG store 초기화 성공")
    return rag


def test_insert_mock_data(rag):
    """테스트 데이터 삽입"""
    print("\n" + "=" * 60)
    print("2. 테스트 데이터 삽입")
    print("=" * 60)

    # 가상의 회의 발언 데이터
    mock_segments = [
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "데이터베이스 성능 이슈가 발생했습니다. 인덱스 최적화가 필요합니다.",
            "start": 0.0,
            "end": 5.0
        },
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "캐시 레이어를 Redis로 추가하면 응답 속도가 개선될 것 같습니다.",
            "start": 10.0,
            "end": 15.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "프론트엔드 번들 크기가 너무 큽니다. 코드 스플리팅을 적용하겠습니다.",
            "start": 20.0,
            "end": 25.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "React Query를 도입해서 서버 상태 관리를 개선하고 있습니다.",
            "start": 30.0,
            "end": 35.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "전체 아키텍처를 마이크로서비스로 전환하는 방안을 검토해보세요.",
            "start": 40.0,
            "end": 45.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "CI/CD 파이프라인을 GitHub Actions로 구축하면 좋을 것 같습니다.",
            "start": 50.0,
            "end": 55.0
        },
    ]

    count = rag.upsert_segments(mock_segments)
    print(f"[OK] {count}개 세그먼트 삽입 완료")

    return mock_segments


def test_search_basic(rag):
    """기본 검색 테스트"""
    print("\n" + "=" * 60)
    print("3. 기본 검색 테스트")
    print("=" * 60)

    queries = [
        "데이터베이스 성능 문제",
        "프론트엔드 최적화",
        "CI/CD 파이프라인",
    ]

    for query in queries:
        print(f"\n[Query] {query}")
        results = rag.search(query, topk=3)

        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['speaker_name']}] {r['text'][:50]}... (score: {r['_score']:.3f})")
        else:
            print("  검색 결과 없음")


def test_speaker_filter(rag):
    """Speaker별 필터링 테스트"""
    print("\n" + "=" * 60)
    print("4. Speaker별 필터링 테스트")
    print("=" * 60)

    # 특정 화자 검색
    speakers = ["speaker_01", "speaker_02", "speaker_03"]

    for speaker_id in speakers:
        print(f"\n[Speaker] {speaker_id}")
        results = rag.search_by_speaker(speaker_id, query="", topk=5)

        if results:
            speaker_name = results[0].get('speaker_name', speaker_id)
            print(f"  화자 이름: {speaker_name}")
            print(f"  총 발언: {len(results)}개")
            for i, r in enumerate(results, 1):
                print(f"    {i}. {r['text'][:40]}...")
        else:
            print("  발언 없음")


def test_speaker_stats(rag):
    """화자 통계 테스트"""
    print("\n" + "=" * 60)
    print("5. 화자 통계 테스트")
    print("=" * 60)

    speakers = rag.get_all_speakers()
    print(f"\n총 {len(speakers)}명의 화자")

    for speaker_id in speakers:
        stats = rag.get_speaker_stats(speaker_id)
        if stats:
            print(f"\n[{stats['speaker_name']}] ({speaker_id})")
            print(f"  총 발언 수: {stats['total_utterances']}")
            print(f"  평균 발언 길이: {stats['avg_length']:.1f}자")
            print(f"  총 발언 시간: {stats['total_duration']:.1f}초")


def test_speaker_query(rag):
    """화자 특정 + 쿼리 검색 테스트"""
    print("\n" + "=" * 60)
    print("6. 화자 특정 쿼리 검색 테스트")
    print("=" * 60)

    test_cases = [
        ("speaker_01", "데이터베이스"),
        ("speaker_02", "프론트엔드"),
        ("speaker_03", "아키텍처"),
    ]

    for speaker_id, query in test_cases:
        print(f"\n[Query] '{query}' by {speaker_id}")
        results = rag.search(query, topk=2, speaker_id=speaker_id)

        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['speaker_name']}] {r['text'][:50]}... (score: {r['_score']:.3f})")
        else:
            print("  검색 결과 없음")


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 60)
    print("Speaker별 RAG 검색 테스트")
    print("=" * 60)

    # 1. RAG 초기화
    rag = test_rag_basic()
    if not rag:
        return

    # 2. 테스트 데이터 삽입
    test_insert_mock_data(rag)

    # 3. 기본 검색
    test_search_basic(rag)

    # 4. Speaker 필터링
    test_speaker_filter(rag)

    # 5. Speaker 통계
    test_speaker_stats(rag)

    # 6. Speaker 특정 쿼리
    test_speaker_query(rag)

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
