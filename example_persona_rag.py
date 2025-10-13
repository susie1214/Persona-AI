#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaker별 RAG + QLoRA 디지털 페르소나 전체 워크플로우 예제

이 스크립트는 다음을 시연합니다:
1. RAG Store에 화자별 발언 저장
2. Speaker별 검색 및 통계
3. QLoRA 학습 데이터셋 생성
4. (선택) 어댑터 학습 및 추론
"""

from core.rag_store import RagStore
from core.persona_training import PersonaDatasetGenerator

def step1_setup_rag():
    """Step 1: RAG Store 초기화 및 데이터 삽입"""
    print("=" * 80)
    print("Step 1: RAG Store 설정 및 데이터 삽입")
    print("=" * 80)

    # RAG Store 초기화
    rag = RagStore(persist_path="./qdrant_db")

    if not rag.ok:
        print("[ERROR] RAG store 초기화 실패!")
        return None

    # 샘플 회의 데이터
    meeting_segments = [
        # 김태진 (speaker_01) - 백엔드/DB 전문가
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "데이터베이스 인덱스 최적화가 시급합니다. 현재 쿼리 응답 시간이 3초를 넘어가고 있어요.",
            "start": 0.0,
            "end": 5.0
        },
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "Redis 캐시 레이어를 추가하면 API 응답 속도를 50% 이상 개선할 수 있을 것 같습니다.",
            "start": 10.0,
            "end": 15.0
        },
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "샤딩 전략도 재검토가 필요합니다. 특히 사용자 테이블의 핫스팟이 문제예요.",
            "start": 20.0,
            "end": 25.0
        },
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "PostgreSQL의 EXPLAIN ANALYZE 결과를 보면 Full Scan이 너무 많이 발생하고 있습니다.",
            "start": 30.0,
            "end": 35.0
        },
        {
            "speaker_id": "speaker_01",
            "speaker_name": "김태진",
            "text": "Connection pool 설정도 다시 조정해야 합니다. 현재 max_connections가 너무 낮아요.",
            "start": 40.0,
            "end": 45.0
        },

        # 이현택 (speaker_02) - 프론트엔드 전문가
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "번들 사이즈가 5MB를 넘어서 초기 로딩이 너무 느립니다. Code splitting을 적용해야 해요.",
            "start": 50.0,
            "end": 55.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "React Query를 도입해서 서버 상태 관리를 개선하고 있습니다. 캐싱도 자동으로 되서 좋아요.",
            "start": 60.0,
            "end": 65.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "Lazy loading과 Suspense를 활용하면 사용자 경험이 훨씬 좋아질 것 같습니다.",
            "start": 70.0,
            "end": 75.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "웹팩 설정을 최적화해서 빌드 시간도 30% 단축했어요. Tree shaking이 제대로 동작하도록 수정했습니다.",
            "start": 80.0,
            "end": 85.0
        },
        {
            "speaker_id": "speaker_02",
            "speaker_name": "이현택",
            "text": "TypeScript strict 모드를 켜면 런타임 에러가 확실히 줄어듭니다. 점진적으로 적용 중입니다.",
            "start": 90.0,
            "end": 95.0
        },

        # 박교수 (speaker_03) - 아키텍처/리더
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "전체 시스템을 마이크로서비스 아키텍처로 전환하는 것을 고려해봐야 합니다.",
            "start": 100.0,
            "end": 105.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "CI/CD 파이프라인을 GitHub Actions로 구축하면 배포 자동화가 훨씬 쉬워질 겁니다.",
            "start": 110.0,
            "end": 115.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "모니터링 시스템을 Prometheus + Grafana로 통합하는 게 좋겠어요.",
            "start": 120.0,
            "end": 125.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "보안 감사를 정기적으로 실시하고, 취약점 스캐닝을 자동화해야 합니다.",
            "start": 130.0,
            "end": 135.0
        },
        {
            "speaker_id": "speaker_03",
            "speaker_name": "박교수",
            "text": "문서화가 부족한 것 같습니다. API 문서를 OpenAPI 스펙으로 작성하면 좋겠어요.",
            "start": 140.0,
            "end": 145.0
        },
    ]

    # 데이터 삽입
    count = rag.upsert_segments(meeting_segments)
    print(f"\n✅ {count}개 발언 삽입 완료")

    return rag


def step2_test_rag_search(rag):
    """Step 2: RAG 검색 테스트"""
    print("\n" + "=" * 80)
    print("Step 2: RAG 검색 테스트")
    print("=" * 80)

    # 2.1 일반 검색
    print("\n[2.1] 일반 검색: '데이터베이스 최적화'")
    results = rag.search("데이터베이스 최적화", topk=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['speaker_name']}] {r['text'][:60]}... (score: {r['_score']:.3f})")

    # 2.2 Speaker별 검색
    print("\n[2.2] 김태진의 모든 발언")
    results = rag.search_by_speaker("speaker_01", topk=10)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['text'][:60]}...")

    # 2.3 Speaker별 특정 쿼리
    print("\n[2.3] 이현택의 React 관련 발언")
    results = rag.search("React", speaker_id="speaker_02", topk=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['text'][:60]}... (score: {r['_score']:.3f})")

    # 2.4 Speaker 통계
    print("\n[2.4] Speaker 통계")
    speakers = rag.get_all_speakers()
    for speaker_id in speakers:
        stats = rag.get_speaker_stats(speaker_id)
        print(f"\n  [{stats['speaker_name']}] ({speaker_id})")
        print(f"    - 총 발언: {stats['total_utterances']}개")
        print(f"    - 평균 길이: {stats['avg_length']:.1f}자")


def step3_generate_datasets(rag):
    """Step 3: QLoRA 학습 데이터셋 생성"""
    print("\n" + "=" * 80)
    print("Step 3: QLoRA 학습 데이터셋 생성")
    print("=" * 80)

    # 데이터셋 생성기 초기화
    generator = PersonaDatasetGenerator(output_dir="data/persona_datasets")

    # 모든 Speaker의 데이터셋 생성 (최소 5개 발언)
    datasets = generator.generate_all_datasets(rag, min_utterances=5)

    print(f"\n✅ 총 {len(datasets)}개 데이터셋 생성 완료:")
    for speaker_id, filepath in datasets.items():
        print(f"  - {speaker_id}: {filepath}")

    return datasets


def step4_train_adapters(datasets):
    """Step 4: (선택) QLoRA 어댑터 학습"""
    print("\n" + "=" * 80)
    print("Step 4: QLoRA 어댑터 학습 (선택)")
    print("=" * 80)

    print("\n학습을 시작하려면 다음 명령어를 실행하세요:")
    print()

    for speaker_id, dataset_path in datasets.items():
        print(f"# {speaker_id} 학습")
        print(f"python train_persona.py \\")
        print(f"  --dataset {dataset_path} \\")
        print(f"  --speaker-id {speaker_id} \\")
        print(f"  --epochs 3 \\")
        print(f"  --batch-size 4")
        print()

    print("학습에는 GPU가 권장되며, 각 Speaker당 10-30분 소요됩니다.")
    print("학습된 어댑터는 adapters/{speaker_id}/final/ 에 저장됩니다.")


def step5_test_inference():
    """Step 5: (선택) 추론 테스트"""
    print("\n" + "=" * 80)
    print("Step 5: 추론 테스트 (선택)")
    print("=" * 80)

    print("\n어댑터 학습이 완료되면 다음처럼 사용할 수 있습니다:")
    print()
    print("```python")
    print("from core.adapter import AdapterManager")
    print("from core.rag_store import RagStore")
    print()
    print("# 1. RAG Store 로드")
    print("rag = RagStore('./qdrant_db')")
    print()
    print("# 2. 어댑터 관리자 초기화")
    print("adapter_mgr = AdapterManager(use_4bit=True)")
    print("adapter_mgr.load_base('Qwen/Qwen2.5-3B-Instruct')")
    print("adapter_mgr.load_all_adapters('adapters')")
    print()
    print("# 3. RAG + QLoRA 하이브리드 답변")
    print("query = '데이터베이스 성능을 어떻게 개선할 수 있나요?'")
    print()
    print("# RAG 검색")
    print("context = rag.search(query, speaker_id='speaker_01', topk=3)")
    print()
    print("# QLoRA 답변 생성 (김태진 페르소나)")
    print("response = adapter_mgr.respond_with_context(")
    print("    query=query,")
    print("    rag_context=context,")
    print("    speaker_id='speaker_01'")
    print(")")
    print()
    print("print(f'[김태진] {response}')")
    print("```")


def main():
    """메인 워크플로우"""
    print("\n" + "=" * 80)
    print("Speaker별 RAG + QLoRA 디지털 페르소나 전체 워크플로우")
    print("=" * 80)

    # Step 1: RAG 설정
    rag = step1_setup_rag()
    if not rag:
        return

    # Step 2: RAG 검색 테스트
    step2_test_rag_search(rag)

    # Step 3: 데이터셋 생성
    datasets = step3_generate_datasets(rag)

    # Step 4: 학습 가이드
    step4_train_adapters(datasets)

    # Step 5: 추론 가이드
    step5_test_inference()

    print("\n" + "=" * 80)
    print("워크플로우 완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. 생성된 데이터셋 확인: data/persona_datasets/")
    print("2. (선택) QLoRA 학습 실행: python train_persona.py ...")
    print("3. (선택) 추론 테스트: python core/adapter.py")
    print()


if __name__ == "__main__":
    main()
