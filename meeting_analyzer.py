# meeting_analyzer.py
# 회의 내용 분석 및 요약 모듈

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import dateparser
from dataclasses import dataclass

from models import Segment, ActionItem, MeetingState

# 액션 키워드 정의
ACTION_KEYWORDS = {
    'ko': [
        '해야', '해주세요', '진행', '확인', '정리', '검토', '공유', '작성', 
        '업로드', '보고', '회의', '예약', '훈련', '배포', '테스트', '구매', '설치',
        '연락', '준비', '완료', '수정', '개선', '계획', '실행', '점검'
    ],
    'en': [
        'should', 'need to', 'will do', 'action', 'todo', 'task', 
        'assign', 'complete', 'review', 'update', 'prepare', 'follow up'
    ]
}

PRIORITY_KEYWORDS = {
    'high': ['긴급', '중요', '우선', 'urgent', 'critical', 'asap'],
    'normal': ['보통', '일반', 'normal', 'regular'],
    'low': ['나중', '여유', '천천히', 'low', 'later', 'when possible']
}

@dataclass
class SpeakerStats:
    """화자별 통계"""
    name: str
    total_duration: float
    segment_count: int
    avg_segment_length: float
    word_count: int
    
    @property
    def speaking_percentage(self) -> float:
        return 0.0  # 전체 대비 비율은 외부에서 계산

class MeetingAnalyzer:
    """회의 내용 분석 클래스"""
    
    def __init__(self):
        self.action_patterns = self._compile_action_patterns()
        self.time_patterns = self._compile_time_patterns()
    
    def _compile_action_patterns(self) -> List[re.Pattern]:
        """액션 아이템 감지를 위한 정규식 패턴 컴파일"""
        patterns = []
        
        # 한국어 패턴
        korean_patterns = [
            r'(.+?)\s*(해야|해주세요|진행해|확인해|정리해|검토해|준비해|완료해).*',
            r'(.+?)\s*(까지|전에|이후에)\s*(해야|진행|완료)',
            r'(.*?)(다음\s*주|내일|오늘|이번\s*주)\s*(.*?)(해야|진행)',
            r'(.+?)\s*담당.*',
            r'(.+?)\s*맡아.*'
        ]
        
        # 영어 패턴  
        english_patterns = [
            r'(.+?)\s*(should|need to|will|must)\s*(.+)',
            r'(.+?)\s*(by|before|after)\s*(.+)',
            r'(.+?)\s*(assigned to|responsible for)\s*(.+)'
        ]
        
        all_patterns = korean_patterns + english_patterns
        
        for pattern in all_patterns:
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                print(f"정규식 컴파일 오류: {pattern} - {e}")
        
        return patterns
    
    def _compile_time_patterns(self) -> List[re.Pattern]:
        """시간 관련 패턴 컴파일"""
        patterns = []
        
        time_patterns = [
            r'(\d{1,2})[:/](\d{2})',  # 시:분
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # 년-월-일
            r'(내일|오늘|모레|다음\s*주|이번\s*주|다음\s*달)',
            r'(\d+)(일|주|개월)\s*(후|전|뒤)',
            r'(월|화|수|목|금|토|일)요일'
        ]
        
        for pattern in time_patterns:
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                print(f"시간 패턴 컴파일 오류: {pattern} - {e}")
        
        return patterns
    
    def analyze_meeting(self, segments: List[Segment]) -> Dict[str, any]:
        """회의 전체 분석"""
        if not segments:
            return {
                'summary': '분석할 내용이 없습니다.',
                'action_items': [],
                'speaker_stats': {},
                'total_duration': 0.0,
                'segment_count': 0
            }
        
        # 기본 통계
        total_duration = max(seg.end for seg in segments) if segments else 0.0
        segment_count = len(segments)
        
        # 화자별 통계
        speaker_stats = self._calculate_speaker_stats(segments, total_duration)
        
        # 요약 생성
        summary = self._generate_summary(segments)
        
        # 액션 아이템 추출
        action_items = self._extract_action_items(segments)
        
        # 주요 토픽 추출
        topics = self._extract_topics(segments)
        
        # 회의 품질 점수
        quality_score = self._calculate_quality_score(segments, speaker_stats)
        
        return {
            'summary': summary,
            'action_items': action_items,
            'speaker_stats': speaker_stats,
            'topics': topics,
            'total_duration': total_duration,
            'segment_count': segment_count,
            'quality_score': quality_score,
            'analysis_time': datetime.now().isoformat()
        }
    
    def _calculate_speaker_stats(self, segments: List[Segment], total_duration: float) -> Dict[str, SpeakerStats]:
        """화자별 통계 계산"""
        speaker_data = {}
        
        for segment in segments:
            speaker = segment.speaker_name
            if speaker == "Unknown":
                continue
                
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'word_count': 0
                }
            
            speaker_data[speaker]['total_duration'] += segment.duration
            speaker_data[speaker]['segment_count'] += 1
            speaker_data[speaker]['word_count'] += len(segment.text.split())
        
        # SpeakerStats 객체로 변환
        stats = {}
        for speaker, data in speaker_data.items():
            avg_length = data['total_duration'] / data['segment_count'] if data['segment_count'] > 0 else 0.0
            
            stats[speaker] = SpeakerStats(
                name=speaker,
                total_duration=data['total_duration'],
                segment_count=data['segment_count'],
                avg_segment_length=avg_length,
                word_count=data['word_count']
            )
        
        return stats
    
    def _generate_summary(self, segments: List[Segment], max_sentences: int = 10) -> str:
        """회의 요약 생성"""
        if not segments:
            return "요약할 내용이 없습니다."
        
        # 최근 세그먼트들을 우선적으로 포함
        recent_segments = segments[-max_sentences:] if len(segments) > max_sentences else segments
        
        summary_lines = []
        current_speaker = None
        
        for segment in recent_segments:
            if not segment.text.strip():
                continue
                
            # 화자가 바뀌면 구분
            if current_speaker != segment.speaker_name:
                if summary_lines:  # 첫 번째가 아닌 경우만 구분선 추가
                    summary_lines.append("")
                current_speaker = segment.speaker_name
            
            # 시간 정보와 함께 내용 추가
            time_str = f"{int(segment.start//60):02d}:{int(segment.start%60):02d}"
            summary_lines.append(f"[{time_str}] {segment.speaker_name}: {segment.text}")
        
        if not summary_lines:
            return "요약할 내용이 없습니다."
        
        return "\n".join(summary_lines)
    
    def _extract_action_items(self, segments: List[Segment]) -> List[ActionItem]:
        """액션 아이템 추출"""
        action_items = []
        
        for segment in segments:
            text = segment.text
            
            # 액션 키워드 포함 여부 확인
            has_action_keyword = any(
                keyword in text.lower() 
                for lang_keywords in ACTION_KEYWORDS.values()
                for keyword in lang_keywords
            )
            
            if not has_action_keyword:
                continue
            
            # 정규식 패턴 매칭
            action_text = text
            matched = False
            
            for pattern in self.action_patterns:
                match = pattern.search(text)
                if match:
                    action_text = text  # 전체 텍스트를 액션으로 사용
                    matched = True
                    break
            
            if has_action_keyword or matched:
                # 우선순위 결정
                priority = self._determine_priority(text)
                
                # 마감일 추출
                deadline = self._extract_deadline(text)
                
                action_item = ActionItem(
                    id=f"action_{len(action_items)+1}",
                    text=action_text.strip(),
                    speaker=segment.speaker_name,
                    deadline=deadline,
                    priority=priority,
                    status="pending"
                )
                
                action_items.append(action_item)
        
        # 중복 제거
        unique_actions = []
        seen_texts = set()
        
        for action in action_items:
            action_key = f"{action.text.lower().strip()}{action.speaker}"
            if action_key not in seen_texts:
                unique_actions.append(action)
                seen_texts.add(action_key)
        
        return unique_actions
    
    def _determine_priority(self, text: str) -> str:
        """텍스트에서 우선순위 결정"""
        text_lower = text.lower()
        
        for priority, keywords in PRIORITY_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority
        
        return "normal"
    
    def _extract_deadline(self, text: str) -> Optional[datetime]:
        """텍스트에서 마감일 추출"""
        try:
            # dateparser를 사용하여 자연어 날짜 파싱
            parsed_date = dateparser.parse(text, languages=['ko', 'en'])
            
            if parsed_date:
                # 과거 날짜인 경우 미래로 조정
                if parsed_date < datetime.now():
                    if "주" in text or "week" in text.lower():
                        parsed_date += timedelta(weeks=1)
                    elif "달" in text or "month" in text.lower():
                        parsed_date += timedelta(days=30)
                    else:
                        parsed_date += timedelta(days=1)
                
                return parsed_date
        except Exception as e:
            print(f"날짜 파싱 오류: {e}")
        
        return None
    
    def _extract_topics(self, segments: List[Segment], top_n: int = 5) -> List[Dict[str, any]]:
        """주요 토픽 추출 (간단한 키워드 기반)"""
        if not segments:
            return []
        
        # 모든 텍스트 결합
        all_text = " ".join(segment.text for segment in segments if segment.text)
        
        # 간단한 단어 빈도 계산
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        
        # 불용어 제거 (간단한 버전)
        stop_words = {
            '그', '이', '저', '것', '수', '등', '및', '또는', '하지만', '그러나',
            '그리고', '또한', '하나', '둘', '셋', 'the', 'and', 'or', 'but', 'in', 'on', 'at'
        }
        
        for word in words:
            if len(word) > 1 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도수 기준 상위 토픽 선택
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        topics = []
        for word, freq in sorted_words[:top_n]:
            topics.append({
                'topic': word,
                'frequency': freq,
                'relevance_score': freq / len(words) if words else 0
            })
        
        return topics
    
    def _calculate_quality_score(self, segments: List[Segment], speaker_stats: Dict[str, SpeakerStats]) -> Dict[str, float]:
        """회의 품질 점수 계산"""
        if not segments or not speaker_stats:
            return {
                'overall': 0.0,
                'participation': 0.0,
                'engagement': 0.0,
                'action_oriented': 0.0
            }
        
        # 참여도 점수 (화자 수와 균등성)
        num_speakers = len(speaker_stats)
        participation_score = min(num_speakers / 5.0, 1.0) * 100  # 최대 5명까지 고려
        
        # 발언 균등성
        if num_speakers > 1:
            durations = [stats.total_duration for stats in speaker_stats.values()]
            avg_duration = sum(durations) / len(durations)
            variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
            evenness = 1.0 / (1.0 + variance / (avg_duration ** 2)) if avg_duration > 0 else 0
            participation_score *= evenness
        
        # 참여 활발성 (발언 빈도)
        total_segments = len(segments)
        total_duration = max(seg.end for seg in segments) if segments else 1
        engagement_score = min(total_segments / (total_duration / 60), 10) * 10  # 분당 발언 수
        
        # 액션 지향성 (액션 아이템 비율)
        action_segments = sum(1 for seg in segments if any(
            keyword in seg.text.lower() 
            for lang_keywords in ACTION_KEYWORDS.values()
            for keyword in lang_keywords
        ))
        action_score = (action_segments / total_segments) * 100 if total_segments > 0 else 0
        
        # 전체 점수
        overall_score = (participation_score + engagement_score + action_score) / 3
        
        return {
            'overall': round(overall_score, 1),
            'participation': round(participation_score, 1),
            'engagement': round(engagement_score, 1),
            'action_oriented': round(action_score, 1)
        }
    
    def generate_meeting_report(self, analysis_result: Dict[str, any]) -> str:
        """회의 보고서 생성"""
        report_lines = []
        
        # 헤더
        report_lines.append("=" * 60)
        report_lines.append("회의 분석 보고서")
        report_lines.append("=" * 60)
        report_lines.append(f"분석 일시: {analysis_result.get('analysis_time', 'N/A')}")
        report_lines.append(f"전체 시간: {analysis_result.get('total_duration', 0):.1f}초")
        report_lines.append(f"총 발언 수: {analysis_result.get('segment_count', 0)}개")
        report_lines.append("")
        
        # 품질 점수
        quality = analysis_result.get('quality_score', {})
        if quality:
            report_lines.append("📊 회의 품질 점수")
            report_lines.append("-" * 30)
            report_lines.append(f"전체 점수: {quality.get('overall', 0)}/100")
            report_lines.append(f"참여도: {quality.get('participation', 0)}/100")
            report_lines.append(f"활발성: {quality.get('engagement', 0)}/100")
            report_lines.append(f"액션 지향성: {quality.get('action_oriented', 0)}/100")
            report_lines.append("")
        
        # 화자별 통계
        speaker_stats = analysis_result.get('speaker_stats', {})
        if speaker_stats:
            report_lines.append("👥 참여자별 통계")
            report_lines.append("-" * 30)
            total_duration = analysis_result.get('total_duration', 1)
            
            for speaker, stats in speaker_stats.items():
                percentage = (stats.total_duration / total_duration) * 100
                report_lines.append(f"• {speaker}")
                report_lines.append(f"  - 발언 시간: {stats.total_duration:.1f}초 ({percentage:.1f}%)")
                report_lines.append(f"  - 발언 횟수: {stats.segment_count}회")
                report_lines.append(f"  - 평균 발언 길이: {stats.avg_segment_length:.1f}초")
                report_lines.append(f"  - 단어 수: {stats.word_count}개")
                report_lines.append("")
        
        # 주요 토픽
        topics = analysis_result.get('topics', [])
        if topics:
            report_lines.append("🔍 주요 토픽")
            report_lines.append("-" * 30)
            for i, topic in enumerate(topics[:5], 1):
                report_lines.append(f"{i}. {topic['topic']} (빈도: {topic['frequency']})")
            report_lines.append("")
        
        # 액션 아이템
        actions = analysis_result.get('action_items', [])
        if actions:
            report_lines.append("✅ 액션 아이템")
            report_lines.append("-" * 30)
            for i, action in enumerate(actions, 1):
                deadline_str = ""
                if action.deadline:
                    deadline_str = f" (마감: {action.deadline.strftime('%Y-%m-%d')})"
                
                priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(action.priority, "🟡")
                report_lines.append(f"{i}. {priority_icon} [{action.speaker}] {action.text}{deadline_str}")
            report_lines.append("")
        
        # 요약
        summary = analysis_result.get('summary', '')
        if summary:
            report_lines.append("📝 회의 요약")
            report_lines.append("-" * 30)
            report_lines.append(summary)
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

class ScheduleManager:
    """일정 및 스케줄 관리"""
    
    def __init__(self):
        self.schedules = []
        self.reminders = []
    
    def create_schedule_memo(self, start_time: datetime, end_time: datetime, 
                           participants: List[str], action_items: List[ActionItem]) -> str:
        """다음 회의 스케줄 메모 생성"""
        memo_lines = []
        
        memo_lines.append("📅 다음 회의 일정")
        memo_lines.append("=" * 40)
        memo_lines.append(f"일시: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")
        memo_lines.append(f"소요 시간: {int((end_time - start_time).total_seconds() / 60)}분")
        
        if participants:
            memo_lines.append(f"참석자: {', '.join(participants)}")
        
        memo_lines.append("")
        memo_lines.append("📋 확인 사항")
        memo_lines.append("-" * 20)
        
        if action_items:
            memo_lines.append("이전 회의 액션 아이템:")
            for i, action in enumerate(action_items[:5], 1):  # 최대 5개
                status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(action.status, "⏳")
                memo_lines.append(f"  {i}. {status_icon} [{action.speaker}] {action.text}")
        
        memo_lines.append("")
        memo_lines.append("💡 준비 사항")
        memo_lines.append("-" * 20)
        memo_lines.append("• 이전 회의록 검토")
        memo_lines.append("• 액션 아이템 진행 상황 점검")
        memo_lines.append("• 새로운 안건 준비")
        memo_lines.append("• 필요한 자료 및 문서 준비")
        
        return "\n".join(memo_lines)
    
    def get_upcoming_deadlines(self, action_items: List[ActionItem], days_ahead: int = 7) -> List[ActionItem]:
        """다가오는 마감일 반환"""
        upcoming = []
        now = datetime.now()
        future_date = now + timedelta(days=days_ahead)
        
        for action in action_items:
            if action.deadline and now <= action.deadline <= future_date:
                upcoming.append(action)
        
        # 마감일 순으로 정렬
        upcoming.sort(key=lambda x: x.deadline or datetime.max)
        return upcoming

# 독립 실행 테스트
if __name__ == "__main__":
    import sys
    from datetime import timedelta
    
    print("=" * 60)
    print("Meeting Analyzer Module Test")
    print("=" * 60)
    
    # 테스트 데이터 생성
    print("📝 Test Data Generation:")
    try:
        from models import Segment, ActionItem
        
        # 샘플 회의 세그먼트 생성
        test_segments = [
            Segment(0, 8, "안녕하세요, 오늘 프로젝트 회의를 시작하겠습니다", "SPEAKER_00", "김철수"),
            Segment(9, 15, "네, 먼저 진행 상황을 보고드리겠습니다", "SPEAKER_01", "이영희"),
            Segment(16, 25, "데이터베이스 설계가 90% 완료되었습니다", "SPEAKER_01", "이영희"),
            Segment(26, 32, "좋습니다. UI 부분은 어떤가요?", "SPEAKER_00", "김철수"),
            Segment(33, 42, "UI 목업을 내일까지 완성해야 합니다", "SPEAKER_02", "박민수"),
            Segment(43, 50, "테스트 계획도 이번 주에 정리해 주세요", "SPEAKER_00", "김철수"),
            Segment(51, 58, "네, 금요일까지 준비하겠습니다", "SPEAKER_02", "박민수"),
            Segment(59, 68, "다음 주 월요일에 고객 미팅이 있습니다", "SPEAKER_01", "이영희"),
            Segment(69, 78, "그럼 주말에 최종 점검을 진행해야겠네요", "SPEAKER_00", "김철수"),
            Segment(79, 85, "중요한 프레젠테이션이니까 꼼꼼히 확인해 주세요", "SPEAKER_00", "김철수"),
        ]
        
        print(f"  ✅ Generated {len(test_segments)} test segments")
        print(f"  - Total duration: {max(seg.end for seg in test_segments):.1f} seconds")
        print(f"  - Speakers: {set(seg.speaker_name for seg in test_segments)}")
        
    except Exception as e:
        print(f"  ❌ Test data generation failed: {e}")
        sys.exit(1)
    
    # MeetingAnalyzer 초기화
    print("\n🧠 MeetingAnalyzer Initialization:")
    try:
        analyzer = MeetingAnalyzer()
        print("  ✅ MeetingAnalyzer created successfully")
        print(f"  - Action patterns compiled: {len(analyzer.action_patterns)}")
        print(f"  - Time patterns compiled: {len(analyzer.time_patterns)}")
        
    except Exception as e:
        print(f"  ❌ MeetingAnalyzer initialization failed: {e}")
        sys.exit(1)
    
    # 화자별 통계 계산 테스트
    print("\n👥 Speaker Statistics Test:")
    try:
        total_duration = max(seg.end for seg in test_segments)
        speaker_stats = analyzer._calculate_speaker_stats(test_segments, total_duration)
        
        print(f"  ✅ Calculated statistics for {len(speaker_stats)} speakers:")
        for speaker, stats in speaker_stats.items():
            print(f"    - {speaker}:")
            print(f"      Duration: {stats.total_duration:.1f}s ({stats.total_duration/total_duration*100:.1f}%)")
            print(f"      Segments: {stats.segment_count}")
            print(f"      Avg length: {stats.avg_segment_length:.1f}s")
            print(f"      Word count: {stats.word_count}")
        
    except Exception as e:
        print(f"  ❌ Speaker statistics test failed: {e}")
    
    # 요약 생성 테스트
    print("\n📄 Summary Generation Test:")
    try:
        summary = analyzer._generate_summary(test_segments, max_sentences=5)
        print("  ✅ Summary generated:")
        print("    " + "\n    ".join(summary.split('\n')[:3]))  # 첫 3줄만 표시
        print(f"    ... (total {len(summary.split())} words)")
        
    except Exception as e:
        print(f"  ❌ Summary generation test failed: {e}")
    
    # 액션 아이템 추출 테스트
    print("\n✅ Action Items Extraction Test:")
    try:
        action_items = analyzer._extract_action_items(test_segments)
        print(f"  ✅ Extracted {len(action_items)} action items:")
        
        for i, action in enumerate(action_items, 1):
            priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(action.priority, "🟡")
            deadline_str = f" (Deadline: {action.deadline.strftime('%Y-%m-%d')})" if action.deadline else ""
            print(f"    {i}. {priority_icon} [{action.speaker}] {action.text[:50]}...{deadline_str}")
            print(f"       Priority: {action.priority}, Status: {action.status}")
        
    except Exception as e:
        print(f"  ❌ Action items extraction test failed: {e}")
    
    # 주요 토픽 추출 테스트
    print("\n🔍 Topics Extraction Test:")
    try:
        topics = analyzer._extract_topics(test_segments, top_n=5)
        print(f"  ✅ Extracted {len(topics)} main topics:")
        
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. '{topic['topic']}' (frequency: {topic['frequency']}, relevance: {topic['relevance_score']:.3f})")
        
    except Exception as e:
        print(f"  ❌ Topics extraction test failed: {e}")
    
    # 회의 품질 점수 계산 테스트
    print("\n📊 Quality Score Calculation Test:")
    try:
        # 먼저 화자 통계가 필요
        total_duration = max(seg.end for seg in test_segments)
        speaker_stats = analyzer._calculate_speaker_stats(test_segments, total_duration)
        
        quality_scores = analyzer._calculate_quality_score(test_segments, speaker_stats)
        print("  ✅ Quality scores calculated:")
        
        for metric, score in quality_scores.items():
            meter = "█" * int(score/10) + "░" * (10 - int(score/10))
            print(f"    - {metric.capitalize()}: {score}/100 [{meter}]")
        
    except Exception as e:
        print(f"  ❌ Quality score calculation test failed: {e}")
    
    # 전체 회의 분석 테스트
    print("\n🎯 Complete Meeting Analysis Test:")
    try:
        analysis_result = analyzer.analyze_meeting(test_segments)
        
        print("  ✅ Complete analysis performed:")
        print(f"    - Total duration: {analysis_result['total_duration']:.1f}s")
        print(f"    - Segment count: {analysis_result['segment_count']}")
        print(f"    - Speakers analyzed: {len(analysis_result['speaker_stats'])}")
        print(f"    - Action items found: {len(analysis_result['action_items'])}")
        print(f"    - Topics identified: {len(analysis_result['topics'])}")
        print(f"    - Overall quality: {analysis_result['quality_score']['overall']}/100")
        
    except Exception as e:
        print(f"  ❌ Complete analysis test failed: {e}")
        analysis_result = {}
    
    # 보고서 생성 테스트
    print("\n📋 Report Generation Test:")
    try:
        if analysis_result:
            report = analyzer.generate_meeting_report(analysis_result)
            print("  ✅ Meeting report generated:")
            
            # 보고서의 첫 몇 줄만 표시
            report_lines = report.split('\n')
            for line in report_lines[:10]:
                print(f"    {line}")
            print(f"    ... (total {len(report_lines)} lines)")
            
            # 보고서를 파일로 저장
            try:
                from config import config
                report_file = config.storage.OUTPUT_DIR / "test_meeting_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"  💾 Report saved to: {report_file}")
            except Exception as e:
                print(f"  ⚠️ Report save failed: {e}")
        
    except Exception as e:
        print(f"  ❌ Report generation test failed: {e}")
    
    # 날짜/시간 파싱 테스트
    print("\n⏰ Date/Time Parsing Test:")
    try:
        test_texts = [
            "내일까지 완성해 주세요",
            "다음 주 월요일에 회의가 있습니다", 
            "2024년 12월 25일까지 제출",
            "오후 3시에 미팅이 예정되어 있습니다",
            "이번 주 금요일까지 준비"
        ]
        
        print("  ✅ Testing date/time extraction:")
        for text in test_texts:
            deadline = analyzer._extract_deadline(text)
            if deadline:
                print(f"    '{text}' -> {deadline.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"    '{text}' -> No date found")
        
    except Exception as e:
        print(f"  ❌ Date/time parsing test failed: {e}")
    
    # ScheduleManager 테스트
    print("\n📅 ScheduleManager Test:")
    try:
        schedule_manager = ScheduleManager()
        
        # 샘플 액션 아이템 생성
        sample_actions = [
            ActionItem("1", "UI 목업 완성", "박민수", datetime.now() + timedelta(days=1), "high"),
            ActionItem("2", "테스트 계획 정리", "김철수", datetime.now() + timedelta(days=3), "normal"),
            ActionItem("3", "데이터베이스 최적화", "이영희", datetime.now() + timedelta(days=7), "low")
        ]
        
        # 일정 메모 생성
        start_time = datetime.now() + timedelta(days=7)
        end_time = start_time + timedelta(hours=2)
        participants = ["김철수", "이영희", "박민수"]
        
        schedule_memo = schedule_manager.create_schedule_memo(
            start_time, end_time, participants, sample_actions
        )
        
        print("  ✅ Schedule memo generated:")
        memo_lines = schedule_memo.split('\n')
        for line in memo_lines[:8]:
            print(f"    {line}")
        print(f"    ... (total {len(memo_lines)} lines)")
        
        # 다가오는 마감일 확인
        upcoming = schedule_manager.get_upcoming_deadlines(sample_actions, days_ahead=7)
        print(f"  ✅ Upcoming deadlines: {len(upcoming)} items")
        
        for action in upcoming:
            days_left = (action.deadline - datetime.now()).days
            print(f"    - {action.text[:40]}... ({days_left} days left)")
        
    except Exception as e:
        print(f"  ❌ ScheduleManager test failed: {e}")
    
    # 성능 테스트
    print("\n⚡ Performance Test:")
    try:
        import time
        
        # 대량 세그먼트로 성능 테스트
        large_segments = []
        for i in range(100):
            segment = Segment(
                start=i*2, 
                end=i*2+1.5, 
                text=f"테스트 발언 번호 {i}입니다. 프로젝트 진행상황을 확인해야 합니다.",
                speaker_id=f"SPEAKER_{i%3:02d}",
                speaker_name=f"참여자{i%3+1}"
            )
            large_segments.append(segment)
        
        start_time = time.time()
        large_analysis = analyzer.analyze_meeting(large_segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"  ✅ Analyzed {len(large_segments)} segments in {processing_time:.3f} seconds")
        print(f"  - Average: {processing_time/len(large_segments)*1000:.2f} ms per segment")
        print(f"  - Found {len(large_analysis['action_items'])} action items")
        print(f"  - Identified {len(large_analysis['topics'])} topics")
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Meeting Analyzer Module Test Complete!")
    
    if '--interactive' in sys.argv:
        print("\nInteractive mode:")
        print("You can now test specific features:")
        print("1. Enter text to extract action items")
        print("2. Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    # 사용자 입력으로 간단한 액션 추출 테스트
                    test_segment = Segment(0, 5, user_input, "USER", "사용자")
                    actions = analyzer._extract_action_items([test_segment])
                    
                    if actions:
                        print(f"Found {len(actions)} action item(s):")
                        for action in actions:
                            print(f"  - {action.text} (Priority: {action.priority})")
                    else:
                        print("No action items detected")
            
            except KeyboardInterrupt:
                break
        
        print("Interactive mode ended.")