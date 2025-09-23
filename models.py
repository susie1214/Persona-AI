# models.py
# 데이터 모델 정의

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

@dataclass
class Segment:
    """음성 세그먼트"""
    start: float
    end: float
    text: str
    speaker_id: str = "Unknown"  # pyannote speaker label
    speaker_name: str = "Unknown"  # mapped human name
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass
class SpeakerProfile:
    """화자 프로필"""
    speaker_id: str
    embeddings: List[List[float]] = field(default_factory=list)
    last_seen: float = 0.0
    total_duration: float = 0.0
    sample_count: int = 0
    display_name: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.speaker_id

@dataclass
class ConversationEntry:
    """대화 엔트리"""
    id: str
    timestamp: str
    speaker: str
    text: str
    start_time: float
    end_time: float
    embedding: Optional[List[float]] = None
    session_id: str = ""
    
    @classmethod
    def from_segment(cls, segment: Segment, session_id: str = "") -> 'ConversationEntry':
        """Segment에서 ConversationEntry 생성"""
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            speaker=segment.speaker_name,
            text=segment.text,
            start_time=segment.start,
            end_time=segment.end,
            session_id=session_id
        )

@dataclass
class MeetingState:
    """회의 상태 관리"""
    live_segments: List[Segment] = field(default_factory=list)
    diar_segments: List[tuple] = field(default_factory=list)
    speaker_map: Dict[str, str] = field(default_factory=dict)
    speaker_profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    
    summary: str = ""
    actions: List[str] = field(default_factory=list)
    schedule_note: str = ""
    
    # 설정
    diarization_enabled: bool = False
    forced_speaker_name: Optional[str] = None
    use_gpu: bool = True
    asr_model: str = "medium"
    
    # 세션 정보
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    raw_audio_path: str = ""
    audio_time_elapsed: float = 0.0
    
    def add_segment(self, segment: Segment):
        """세그먼트 추가"""
        self.live_segments.append(segment)
        
    def get_speakers(self) -> List[str]:
        """현재 세션의 화자 목록 반환"""
        speakers = set()
        for segment in self.live_segments:
            if segment.speaker_name != "Unknown":
                speakers.add(segment.speaker_name)
        return sorted(list(speakers))
    
    def get_total_duration(self) -> float:
        """전체 회의 시간 반환"""
        if not self.live_segments:
            return 0.0
        return max(seg.end for seg in self.live_segments)

@dataclass
class SearchResult:
    """검색 결과"""
    id: str
    score: float
    speaker: str
    text: str
    timestamp: str
    session_id: str
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class ActionItem:
    """액션 아이템"""
    id: str
    text: str
    speaker: str
    deadline: Optional[datetime] = None
    priority: str = "normal"  # low, normal, high
    status: str = "pending"  # pending, in_progress, completed
    created_at: datetime = field(default_factory=datetime.now)

# 독립 실행 테스트
if __name__ == "__main__":
    import json
    from datetime import timedelta
    
    print("=" * 50)
    print("Data Models Test")
    print("=" * 50)
    
    # Segment 테스트
    print("🎤 Segment Model Test:")
    segment1 = Segment(
        start=10.5,
        end=15.2,
        text="안녕하세요, 회의를 시작하겠습니다.",
        speaker_id="SPEAKER_00",
        speaker_name="김철수",
        confidence=0.95
    )
    print(f"  - Duration: {segment1.duration:.1f}s")
    print(f"  - Text: '{segment1.text[:30]}...'")
    print(f"  - Speaker: {segment1.speaker_name} (ID: {segment1.speaker_id})")
    
    # SpeakerProfile 테스트
    print("\n👤 SpeakerProfile Model Test:")
    profile = SpeakerProfile(
        speaker_id="Person_A",
        embeddings=[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        last_seen=1234567890.0,
        total_duration=120.5,
        sample_count=15,
        display_name="김철수 팀장"
    )
    print(f"  - Speaker: {profile.display_name}")
    print(f"  - Embeddings count: {len(profile.embeddings)}")
    print(f"  - Total duration: {profile.total_duration}s")
    print(f"  - Sample count: {profile.sample_count}")
    
    # ConversationEntry 테스트
    print("\n💬 ConversationEntry Model Test:")
    entry = ConversationEntry.from_segment(segment1, "session_123")
    print(f"  - ID: {entry.id}")
    print(f"  - Session: {entry.session_id}")
    print(f"  - Speaker: {entry.speaker}")
    print(f"  - Text: '{entry.text[:40]}...'")
    print(f"  - Duration: {entry.end_time - entry.start_time:.1f}s")
    
    # MeetingState 테스트
    print("\n📅 MeetingState Model Test:")
    meeting = MeetingState()
    
    # 여러 세그먼트 추가
    segments = [
        Segment(0, 5, "프로젝트 진행 상황을 보고드리겠습니다.", "SPEAKER_00", "김철수"),
        Segment(6, 10, "네, 잘 들었습니다. 질문이 있습니다.", "SPEAKER_01", "이영희"),
        Segment(11, 15, "다음 주까지 문서를 준비해야 합니다.", "SPEAKER_00", "김철수"),
    ]
    
    for segment in segments:
        meeting.add_segment(segment)
    
    print(f"  - Session ID: {meeting.session_id}")
    print(f"  - Total segments: {len(meeting.live_segments)}")
    print(f"  - Speakers: {meeting.get_speakers()}")
    print(f"  - Total duration: {meeting.get_total_duration():.1f}s")
    print(f"  - Diarization enabled: {meeting.diarization_enabled}")
    
    # ActionItem 테스트
    print("\n✅ ActionItem Model Test:")
    action = ActionItem(
        id="action_001",
        text="다음 주까지 프로젝트 문서를 완성해야 합니다",
        speaker="김철수",
        deadline=datetime.now() + timedelta(days=7),
        priority="high",
        status="pending"
    )
    print(f"  - ID: {action.id}")
    print(f"  - Text: '{action.text}'")
    print(f"  - Speaker: {action.speaker}")
    print(f"  - Deadline: {action.deadline.strftime('%Y-%m-%d %H:%M') if action.deadline else 'None'}")
    print(f"  - Priority: {action.priority}")
    print(f"  - Status: {action.status}")
    
    # SearchResult 테스트
    print("\n🔍 SearchResult Model Test:")
    search_result = SearchResult(
        id="result_001",
        score=0.85,
        speaker="김철수",
        text="프로젝트 진행 상황을 보고드리겠습니다.",
        timestamp=datetime.now().isoformat(),
        session_id="session_123",
        start_time=0.0,
        end_time=5.0
    )
    print(f"  - Score: {search_result.score:.3f}")
    print(f"  - Speaker: {search_result.speaker}")
    print(f"  - Text: '{search_result.text}'")
    print(f"  - Session: {search_result.session_id}")
    
    # JSON 직렬화 테스트
    print("\n📝 JSON Serialization Test:")
    try:
        # ConversationEntry를 dict로 변환
        entry_dict = {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "speaker": entry.speaker,
            "text": entry.text,
            "start_time": entry.start_time,
            "end_time": entry.end_time,
            "session_id": entry.session_id
        }
        
        # JSON 문자열로 변환
        json_str = json.dumps(entry_dict, indent=2, ensure_ascii=False)
        print(f"  ✅ JSON serialization successful")
        print(f"  Sample: {json_str[:100]}...")
        
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
    
    # 데이터 검증 테스트
    print("\n🔍 Data Validation Test:")
    
    # 잘못된 데이터로 테스트
    try:
        invalid_segment = Segment(
            start=10.0,
            end=5.0,  # 종료 시간이 시작 시간보다 빠름
            text="",
            speaker_id="",
            speaker_name=""
        )
        print(f"  - Invalid duration: {invalid_segment.duration}")  # 음수값
        if invalid_segment.duration < 0:
            print("  ⚠️ Warning: Negative duration detected")
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
    
    print("\n" + "=" * 50)
    print("Data Models Test Complete!")