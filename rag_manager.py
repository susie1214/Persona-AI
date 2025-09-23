# RAG(Retrieval Augmented Generation) 관리 모듈

import os
import uuid
import numpy as np
from typing import List, Dict, Optional, Any
from PyQt6.QtCore import QObject, pyqtSignal

from config import config
from models import Segment, ConversationEntry, SearchResult

# Runtime imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDING_AVAILABLE = False

class EmbeddingManager:
    """임베딩 생성 및 관리"""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.embed_dim = 384
        self.initialize()
    
    def initialize(self):
        """임베딩 모델 초기화"""
        if not EMBEDDING_AVAILABLE:
            print("[WARNING] sentence-transformers 미설치 - 해시 기반 임베딩 사용")
            self.embed_dim = 256
            return
        
        try:
            self.model = SentenceTransformer(config.model.EMBEDDING_MODEL)
            self.embed_dim = self.model.get_sentence_embedding_dimension()
            print(f"임베딩 모델 로드 완료: {config.model.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"임베딩 모델 로드 실패: {e} - 해시 기반 임베딩 사용")
            self.model = None
            self.embed_dim = 256
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩 벡터로 변환"""
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            except Exception as e:
                print(f"임베딩 생성 실패: {e} - 해시 기반 폴백")
        
        # 해시 기반 폴백 임베딩
        return self._hash_embedding(texts)
    
    def _hash_embedding(self, texts: List[str]) -> List[List[float]]:
        """간단한 해시 기반 임베딩 (폴백용)"""
        embeddings = []
        for text in texts:
            vec = np.zeros(self.embed_dim, dtype=np.float32)
            for i, char in enumerate(text.encode('utf-8')):
                vec[i % self.embed_dim] += (char % 13) / 13.0
            
            # 정규화
            norm = np.linalg.norm(vec) + 1e-9
            embeddings.append((vec / norm).tolist())
        
        return embeddings

class RAGManager(QObject):
    """RAG 시스템 메인 관리자"""
    
    # Signals
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.client: Optional[QdrantClient] = None
        self.embedding_manager = EmbeddingManager()
        self.collection_name = config.storage.COLLECTION_NAME
        self._id_counter = 1
        self.available = False
        
        self.initialize()
    
    def initialize(self):
        """RAG 시스템 초기화"""
        if not QDRANT_AVAILABLE:
            self.status_update.emit("Qdrant 미설치 - RAG 기능 비활성화")
            return
        
        try:
            # In-memory 클라이언트 먼저 시도
            self.client = QdrantClient(":memory:")
            self.available = True
            self.status_update.emit("Qdrant in-memory 모드 초기화")
        except Exception:
            try:
                # 로컬 파일 저장소 시도
                config.storage.QDRANT_PATH.mkdir(exist_ok=True)
                self.client = QdrantClient(path=str(config.storage.QDRANT_PATH))
                self.available = True
                self.status_update.emit(f"Qdrant 로컬 저장소 초기화: {config.storage.QDRANT_PATH}")
            except Exception as e:
                self.status_update.emit(f"Qdrant 초기화 실패: {e}")
                self.available = False
                return
        
        self._setup_collection()
    
    def _setup_collection(self):
        """컬렉션 설정"""
        if not self.available:
            return
        
        try:
            # 기존 컬렉션 확인
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # 새 컬렉션 생성
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_manager.embed_dim,
                        distance=Distance.COSINE
                    )
                )
                self.status_update.emit(f"새 컬렉션 생성: {self.collection_name}")
            else:
                self.status_update.emit(f"기존 컬렉션 사용: {self.collection_name}")
                
        except Exception as e:
            self.status_update.emit(f"컬렉션 설정 실패: {e}")
            self.available = False
    
    def add_segments(self, segments: List[Segment], session_id: str = ""):
        """세그먼트들을 RAG에 추가"""
        if not self.available or not segments:
            return False
        
        try:
            # 텍스트 준비
            texts = []
            payloads = []
            points = []
            
            for segment in segments:
                text = f"[{segment.speaker_name}] {segment.text}"
                texts.append(text)
                
                payload = {
                    "speaker_id": segment.speaker_id,
                    "speaker_name": segment.speaker_name,
                    "text": segment.text,
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "duration": segment.duration,
                    "session_id": session_id,
                    "timestamp": str(uuid.uuid4())  # 임시 타임스탬프
                }
                payloads.append(payload)
            
            # 임베딩 생성
            embeddings = self.embedding_manager.encode(texts)
            
            # Qdrant 포인트 생성
            for i, (embedding, payload) in enumerate(zip(embeddings, payloads)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # 업서트
            self.client.upsert(collection_name=self.collection_name, points=points)
            self.status_update.emit(f"RAG에 {len(points)}개 세그먼트 추가됨")
            return True
            
        except Exception as e:
            self.status_update.emit(f"RAG 추가 실패: {e}")
            return False
    
    def add_conversation_entry(self, entry: ConversationEntry):
        """단일 대화 엔트리를 RAG에 추가"""
        if not self.available:
            return False
        
        try:
            # 임베딩 생성
            text = f"[{entry.speaker}] {entry.text}"
            embeddings = self.embedding_manager.encode([text])
            
            # 포인트 생성
            point = PointStruct(
                id=entry.id,
                vector=embeddings[0],
                payload={
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": entry.timestamp,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "session_id": entry.session_id
                }
            )
            
            # 업서트
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return True
            
        except Exception as e:
            self.status_update.emit(f"대화 엔트리 추가 실패: {e}")
            return False
    
    def search(self, query: str, limit: int = 5, session_id: Optional[str] = None) -> List[SearchResult]:
        """유사한 대화 내용 검색"""
        if not self.available:
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embeddings = self.embedding_manager.encode([query])
            
            # 검색 필터 (세션 ID 기반)
            search_filter = None
            if session_id:
                search_filter = {"session_id": session_id}
            
            # 검색 실행
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embeddings[0],
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            )
            
            # SearchResult 객체로 변환
            search_results = []
            for result in results:
                payload = result.payload or {}
                search_result = SearchResult(
                    id=str(result.id),
                    score=result.score,
                    speaker=payload.get("speaker", payload.get("speaker_name", "Unknown")),
                    text=payload.get("text", ""),
                    timestamp=payload.get("timestamp", ""),
                    session_id=payload.get("session_id", ""),
                    start_time=payload.get("start_time", 0.0),
                    end_time=payload.get("end_time", 0.0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.status_update.emit(f"검색 실패: {e}")
            return []
    
    def get_conversation_context(self, query: str, limit: int = 5) -> str:
        """검색 결과를 문맥 문자열로 반환"""
        results = self.search(query, limit)
        
        if not results:
            return "관련 대화 내용을 찾을 수 없습니다."
        
        context_lines = []
        for result in results:
            context_lines.append(f"- [{result.speaker}] {result.text} (유사도: {result.score:.3f})")
        
        return "\n".join(context_lines)
    
    def clear_collection(self):
        """컬렉션 내용 삭제"""
        if not self.available:
            return False
        
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection()
            self.status_update.emit("RAG 컬렉션 초기화됨")
            return True
        except Exception as e:
            self.status_update.emit(f"컬렉션 삭제 실패: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """RAG 통계 정보 반환"""
        if not self.available:
            return {"available": False}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "available": True,
                "total_points": collection_info.points_count,
                "collection_name": self.collection_name,
                "embedding_dim": self.embedding_manager.embed_dim,
                "embedding_model": config.model.EMBEDDING_MODEL if self.embedding_manager.model else "Hash-based"
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

class ConversationManager:
    """대화 내용 관리 및 분석"""
    
    def __init__(self, rag_manager: RAGManager):
        self.rag_manager = rag_manager
        self.conversation_log: List[ConversationEntry] = []
        self.current_session_id = str(uuid.uuid4())
    
    def add_segment(self, segment: Segment):
        """세그먼트를 대화 로그에 추가하고 RAG에 인덱싱"""
        entry = ConversationEntry.from_segment(segment, self.current_session_id)
        self.conversation_log.append(entry)
        
        # RAG에 실시간 추가
        self.rag_manager.add_conversation_entry(entry)
        
        return entry
    
    def get_recent_conversation(self, limit: int = 10) -> List[ConversationEntry]:
        """최근 대화 내용 반환"""
        return self.conversation_log[-limit:] if self.conversation_log else []
    
    def search_conversation_history(self, query: str, limit: int = 5) -> List[SearchResult]:
        """대화 기록에서 검색"""
        return self.rag_manager.search(query, limit, self.current_session_id)
    
    def export_conversation(self) -> Dict[str, Any]:
        """대화 내용 내보내기"""
        return {
            "session_id": self.current_session_id,
            "total_entries": len(self.conversation_log),
            "entries": [
                {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time
                }
                for entry in self.conversation_log
            ]
        }
    
    def new_session(self):
        """새 세션 시작"""
        self.current_session_id = str(uuid.uuid4())
        self.conversation_log.clear()

# 독립 실행 테스트
if __name__ == "__main__":
    import sys
    import tempfile
    from datetime import datetime
    
    print("=" * 50)
    print("RAG Manager Module Test")
    print("=" * 50)
    
    # 의존성 체크
    print("📦 Dependency Check:")
    print(f"  - Qdrant Client: {'✅ Available' if QDRANT_AVAILABLE else '❌ Not available'}")
    print(f"  - SentenceTransformers: {'✅ Available' if EMBEDDING_AVAILABLE else '❌ Not available'}")
    
    # EmbeddingManager 테스트
    print("\n🧠 EmbeddingManager Test:")
    try:
        embedding_manager = EmbeddingManager()
        print(f"  - Model available: {embedding_manager.model is not None}")
        print(f"  - Embedding dimension: {embedding_manager.embed_dim}")
        
        # 샘플 텍스트 임베딩 테스트
        sample_texts = [
            "안녕하세요, 회의를 시작하겠습니다",
            "프로젝트 진행 상황을 보고드리겠습니다",
            "다음 주까지 문서를 준비해야 합니다"
        ]
        
        print("  - Testing embedding generation...")
        embeddings = embedding_manager.encode(sample_texts)
        print(f"  ✅ Generated {len(embeddings)} embeddings")
        print(f"  - Embedding shape: {len(embeddings[0])} dimensions")
        
        # 임베딩 품질 간단 체크
        if len(embeddings) >= 2:
            import numpy as np
            similarity = np.dot(embeddings[0], embeddings[1])
            print(f"  - Sample similarity score: {similarity:.3f}")
        
    except Exception as e:
        print(f"  ❌ EmbeddingManager test failed: {e}")
    
    # RAGManager 초기화 테스트
    print("\n🔍 RAGManager Initialization Test:")
    try:
        from PyQt6.QtCore import QCoreApplication
        
        # Qt 애플리케이션 필요
        app = QCoreApplication(sys.argv) if not QCoreApplication.instance() else QCoreApplication.instance()
        
        rag_manager = RAGManager()
        print(f"  ✅ RAGManager created")
        print(f"  - Available: {rag_manager.available}")
        print(f"  - Collection: {rag_manager.collection_name}")
        
        if rag_manager.available:
            stats = rag_manager.get_stats()
            print(f"  - Stats: {stats}")
        
    except ImportError:
        print("  ⚠️ PyQt6 not available - skipping RAGManager test")
    except Exception as e:
        print(f"  ❌ RAGManager test failed: {e}")
    
    # 데이터 모델로 테스트 데이터 생성
    print("\n📝 Test Data Generation:")
    try:
        from models import Segment, ConversationEntry
        
        # 샘플 세그먼트 생성
        test_segments = [
            Segment(0, 5, "안녕하세요, 오늘 회의를 시작하겠습니다", "SPEAKER_00", "김철수"),
            Segment(6, 12, "네, 프로젝트 현황부터 말씀드리겠습니다", "SPEAKER_01", "이영희"),
            Segment(13, 18, "이번 주까지 문서를 완성해야 합니다", "SPEAKER_00", "김철수"),
            Segment(19, 25, "테스트 결과를 확인해 보겠습니다", "SPEAKER_02", "박민수"),
            Segment(26, 32, "다음 주 화요일에 회의를 예약해 주세요", "SPEAKER_01", "이영희"),
        ]
        
        print(f"  ✅ Generated {len(test_segments)} test segments")
        
        # ConversationEntry로 변환
        session_id = "test_session_123"
        conversation_entries = []
        
        for segment in test_segments:
            entry = ConversationEntry.from_segment(segment, session_id)
            conversation_entries.append(entry)
        
        print(f"  ✅ Converted to {len(conversation_entries)} conversation entries")
        
        # 샘플 데이터 출력
        print("  Sample data:")
        for i, entry in enumerate(conversation_entries[:2]):
            print(f"    {i+1}. [{entry.speaker}] {entry.text}")
        
    except Exception as e:
        print(f"  ❌ Test data generation failed: {e}")
        test_segments = []
        conversation_entries = []
    
    # RAG 기능 테스트 (사용 가능한 경우)
    if QDRANT_AVAILABLE and len(test_segments) > 0:
        print("\n🔎 RAG Functionality Test:")
        try:
            from PyQt6.QtCore import QCoreApplication
            app = QCoreApplication(sys.argv) if not QCoreApplication.instance() else QCoreApplication.instance()
            
            rag_manager = RAGManager()
            
            if rag_manager.available:
                # 세그먼트 추가 테스트
                print("  - Adding segments to RAG...")
                success = rag_manager.add_segments(test_segments, session_id)
                print(f"  {'✅' if success else '❌'} Segment addition: {success}")
                
                # 검색 테스트
                if success:
                    test_queries = [
                        "프로젝트 현황",
                        "문서 완성",
                        "다음 주 회의",
                        "테스트 결과"
                    ]
                    
                    print("  - Testing search functionality...")
                    for query in test_queries:
                        results = rag_manager.search(query, limit=3)
                        print(f"    Query: '{query}' -> {len(results)} results")
                        
                        if results:
                            best_result = results[0]
                            print(f"      Best: {best_result.speaker} (score: {best_result.score:.3f})")
                    
                    # 컨텍스트 생성 테스트
                    print("  - Testing context generation...")
                    context = rag_manager.get_conversation_context("프로젝트", limit=2)
                    print(f"    Context length: {len(context)} characters")
                    print(f"    Sample: {context[:100]}...")
                
                # 통계 확인
                stats = rag_manager.get_stats()
                print(f"  - Final stats: {stats}")
                
            else:
                print("  ⚠️ RAG not available for testing")
            
        except Exception as e:
            print(f"  ❌ RAG functionality test failed: {e}")
    else:
        print("\n⚠️ RAG Functionality Test:")
        print("  Skipped - requires Qdrant and test data")
    
    # ConversationManager 테스트
    print("\n💬 ConversationManager Test:")
    try:
        # Mock RAG manager
        class MockRAGManager:
            def add_conversation_entry(self, entry):
                return True
            
            def search(self, query, limit=5, session_id=None):
                from models import SearchResult
                return [SearchResult(
                    id="mock_001",
                    score=0.85,
                    speaker="김철수",
                    text=f"Mock result for: {query}",
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id or "mock_session"
                )]
        
        mock_rag = MockRAGManager()
        conv_manager = ConversationManager(mock_rag)
        
        print(f"  ConversationManager created")
        print(f"  - Session ID: {conv_manager.current_session_id}")
        print(f"  - Conversation log size: {len(conv_manager.conversation_log)}")
        
        # 세그먼트 추가 테스트
        if len(test_segments) > 0:
            for segment in test_segments[:3]:  # 처음 3개만
                entry = conv_manager.add_segment(segment)
                print(f"  - Added: [{entry.speaker}] {entry.text[:30]}...")
            
            print(f"  - Final log size: {len(conv_manager.conversation_log)}")
        
        # 최근 대화 가져오기
        recent = conv_manager.get_recent_conversation(2)
        print(f"  - Recent conversations: {len(recent)}")
        
        # 검색 테스트
        search_results = conv_manager.search_conversation_history("프로젝트")
        print(f"  - Search results: {len(search_results)}")
        
        # 대화 내용 내보내기
        export_data = conv_manager.export_conversation()
        print(f"  - Export data entries: {export_data.get('total_entries', 0)}")
        
    except Exception as e:
        print(f"  ConversationManager test failed: {e}")
    
    # 파일 저장/로드 테스트
    print("\n File I/O Test:")
    try:
        import json
        import tempfile
        
        # 테스트 데이터 생성
        test_data = {
            "session_id": "test_session_123",
            "timestamp": datetime.now().isoformat(),
            "entries": [
                {
                    "speaker": "김철수",
                    "text": "회의를 시작하겠습니다",
                    "start_time": 0.0,
                    "end_time": 3.0
                },
                {
                    "speaker": "이영희", 
                    "text": "프로젝트 현황을 보고드리겠습니다",
                    "start_time": 4.0,
                    "end_time": 8.0
                }
            ]
        }
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_file_path = f.name
        
        print(f"  Test data saved to: {temp_file_path}")
        
        # 파일에서 읽기
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print(f"  Data loaded successfully: {loaded_data['session_id']}")
        print(f"  Entries loaded: {len(loaded_data['entries'])}")
        
        # 정리
        import os
        os.unlink(temp_file_path)
        print(f"  Temporary file cleaned up")
        
    except Exception as e:
        print(f"  File I/O test failed: {e}")
    
    # 성능 테스트
    print("\n Performance Test:")
    try:
        embedding_manager = EmbeddingManager()
        
        # 대량 텍스트 임베딩 성능 측정
        large_texts = [f"테스트 문장 번호 {i}입니다" for i in range(100)]
        
        import time
        start_time = time.time()
        embeddings = embedding_manager.encode(large_texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"  - 100 texts embedded in {processing_time:.3f} seconds")
        print(f"  - Average: {processing_time/100*1000:.1f} ms per text")
        print(f"  - Embedding dimensions: {len(embeddings[0])}")
        
    except Exception as e:
        print(f"  Performance test failed: {e}")
    
    # 메모리 사용량 체크 (간단한 버전)
    print("\n Memory Usage Check:")
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"  - RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"  - VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB")
        
    except ImportError:
        print("  psutil not available - memory check skipped")
    except Exception as e:
        print(f"  Memory check failed: {e}")
    
    print("\n" + "=" * 50)
    print("RAG Manager Module Test Complete!")
    
    if '--interactive' in sys.argv:
        print("\nInteractive mode - press Enter to exit")
        input()# rag_manager.py
