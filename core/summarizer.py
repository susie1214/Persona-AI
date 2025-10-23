# ---- 아래를 파일 맨 아래쪽에 추가하세요 ----
from html import escape
from collections import Counter
import re
import datetime
from typing import List, Optional, Dict
from .audio import Segment

# RAG Store import (옵셔널)
try:
    from .rag_store import RagStore  # noqa: F401
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from .llm_router import LLMRouter
except ImportError:
    LLMRouter = None

# simple_summarize 함수 추가 (하위 호환성)
def simple_summarize(segments, max_len=15):
    """기본 텍스트 요약 (LLM 실패 시 fallback)"""
    lines = []
    for s in segments[-max_len:]:
        if getattr(s, "text", "").strip():
            spk = s.speaker_name if s.speaker_name != "Unknown" else "speaker_00"
            lines.append(f"[{spk}] {s.text}")
    return "\n".join(lines) if lines else "요약할 내용이 없습니다."

# ✅ 액션아이템 트리거 키워드(간단 버전) - 없어서 경고났던 부분 보완
ACTION_VERBS = [
    "해야", "해주세요", "진행", "확인", "정리", "검토", "공유", "작성", "업로드",
    "보고", "회의", "예약", "훈련", "배포", "테스트", "구매", "수정", "수정하기",
    "리팩터링", "설정", "정비", "점검", "연동", "연결", "변경", "개선", "등록",
    "적용", "정의", "설계", "구현", "분석", "수립", "정책", "문의", "정리하기",
]

# ✅ dateparser가 없어도 동작하도록 안전하게 처리
try:
    import dateparser  # type: ignore
except Exception:  # 사용자가 설치 안 한 경우를 대비
    dateparser = None

_DATE_RX = re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})")

def _parse_due_fallback_ko(text: str):
    """
    dateparser가 없을 때의 매우 단순한 한국어 날짜 파서(YYYY-MM-DD 우선).
    '내일/모레/다음주' 같은 표현의 최소 대응도 포함.
    """
    # 1) 명시적 날짜
    m = _DATE_RX.search(text)
    if m:
        y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        try:
            dt = datetime.datetime(y, mo, d, 9, 0)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    # 2) 상대 표현 일부 처리
    now = datetime.datetime.now()
    low = text.lower()
    if "내일" in low:
        dt = now + datetime.timedelta(days=1)
        return dt.strftime("%Y-%m-%d 09:00")
    if "모레" in low:
        dt = now + datetime.timedelta(days=2)
        return dt.strftime("%Y-%m-%d 09:00")
    if "다음주" in low or "다음 주" in low:
        dt = now + datetime.timedelta(days=7)
        # 다음주 월 9시 정렬
        dt = dt - datetime.timedelta(days=dt.weekday())  # 월요일
        return dt.strftime("%Y-%m-%d 09:00")

    # 못 찾으면 빈 문자열
    return ""

def _badge(text, bg="#e8f3ff", fg="#0b74de"):
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{bg};color:{fg};font-weight:700;font-size:12px;margin-left:6px">{escape(text)}</span>'

def _li(items):
    return "".join(f"<li>{escape(x)}</li>" for x in items)

def _table(headers, rows):
    th = "".join(
        f'<th style="text-align:left;padding:8px 10px;border-bottom:1px solid #e9ecef">{escape(h)}</th>'
        for h in headers
    )
    trs = []
    for r in rows:
        tds = "".join(
            f'<td style="padding:8px 10px;border-bottom:1px solid #f1f3f5">{escape(str(c))}</td>'
            for c in r
        )
        trs.append(f"<tr>{tds}</tr>")
    return f'''
    <table style="border-collapse:collapse;width:100%;margin-top:6px;border-radius:12px;overflow:hidden">
      <thead style="background:#f8f9fa"><tr>{th}</tr></thead>
      <tbody>{"".join(trs)}</tbody>
    </table>
    '''

def _merge_short_lines(texts, max_len=2):
    """짧은 발화들을 2~3개씩 묶어서 읽기 쉽게."""
    merged, buf = [], []
    for t in texts:
        if len(" ".join(buf + [t])) < 60 and len(buf) < max_len:
            buf.append(t)
        else:
            if buf: merged.append(" ".join(buf))
            buf = [t]
    if buf: merged.append(" ".join(buf))
    return merged

def render_summary_html_from_segments(segments, max_len=12,
                                      meeting_title="회의 요약", date_str=None, participants=None):
    """
    기존 simple_summarize를 대체하는 '스타일 포함' HTML 버전.
    - 마지막 max_len개의 발화를 화자태그 포함으로 카드형 요약으로 보여줌
    - 디자인/레이아웃은 그대로(TextEdit)에 setHtml로만 교체
    """
    # 최근 대화 추출
    lines = []
    for s in segments:
        if getattr(s, "text", "").strip():
            spk = s.speaker_name if s.speaker_name != "Unknown" else "speaker_00"
            # 노이즈성 공백 정리
            txt = re.sub(r"\s+", " ", s.text).strip()
            lines.append(f"[{spk}] {txt}")
    if not lines:
        return "<div style='color:#868e96'>요약할 내용이 없습니다.</div>"

    # 짧은 줄 합치기 → 마지막 N개 뽑기
    merged = _merge_short_lines(lines, max_len=3)
    view = merged[-max_len:]

    # 헤더
    title = escape(meeting_title or "회의 요약")
    date  = f'<span style="color:#868e96;margin-left:8px">{escape(date_str)}</span>' if date_str else ""
    ppl   = ", ".join(participants) if participants else "-"

    header = f'''
    <div style="border-radius:14px;padding:14px 16px;margin-bottom:12px;
                background:linear-gradient(90deg,#e6fcf5,#d0ebff);
                display:flex;align-items:center;justify-content:space-between">
      <div style="font-size:18px;font-weight:800;color:#0b7285">
        🤖 {title}{_badge("AI Summary")}
        {date}
      </div>
      <div style="font-size:12px;color:#495057">참석자: {escape(ppl)}</div>
    </div>
    '''
    # 리스트(최근 발화 요약 카드)
    ul = "<ul style='margin:6px 0 0 0;padding-left:18px;line-height:1.55'>" + _li(view) + "</ul>"

    return f'''
    <div style="font-family:'Pretendard',Segoe UI,Apple SD Gothic Neo,system-ui; font-size:14px; color:#212529">
      {header}
      <div style="margin:4px 0 8px;font-weight:700;color:#1c7ed6">📌 최근 논의 요약</div>
      {ul}
    </div>
    '''

def actions_from_segments(segments):
    """
    기존 extract_actions는 문자열 리스트를 반환.
    UI 표를 위해 owner/due 등을 갖춘 dict 리스트를 함께 쓰기 위해 새로 추가.
    """
    items = []
    seen = set()
    for s in segments:
        text = getattr(s, "text", "").strip()
        if not text:
            continue
        if any(v in text for v in ACTION_VERBS):
            owner = s.speaker_name if s.speaker_name != "Unknown" else "speaker_00"
            # 날짜 파싱
            if dateparser:
                try:
                    deadline = dateparser.parse(text, languages=["ko"])
                except Exception:
                    deadline = None
                due = deadline.strftime("%Y-%m-%d %H:%M") if deadline else _parse_due_fallback_ko(text)
            else:
                due = _parse_due_fallback_ko(text)

            title = text
            key = (owner, title, due)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "title": title,
                "owner": owner,
                "due": due,
                "priority": "M",
                "status": "todo",
                "notes": ""
            })
    return items

def render_actions_table_html(items):
    headers = ["Title","Owner","Due","Priority","Status","Notes"]
    rows = [[i["title"], i["owner"], i["due"], i["priority"], i["status"], i["notes"]] for i in items] if items else []
    table = _table(headers, rows) if rows else "<div style='color:#868e96'>등록된 Action Item이 없습니다.</div>"
    title = '''
    <div style="margin:10px 0 6px;font-weight:700;color:#1c7ed6">📝 Action Items</div>
    '''
    return f'''
    <div style="font-family:'Pretendard',Segoe UI,Apple SD Gothic Neo,system-ui; font-size:14px; color:#212529">
      {title}
      {table}
    </div>
    '''

# --------- Agenda extraction ---------
_AGENDA_HINTS = [
    "논의", "검토", "결정", "의견", "이슈", "리스크", "할 일", "계획",
    "배포", "데모", "기능", "버그", "요약", "성능", "데이터", "모델",
    "화자 분리", "UI", "일정", "스케줄", "리팩터링", "테스트", "배치",
]

def _normalize(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)   # [TAG] (주석) 제거
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _score_sentence(sent: str) -> int:
    s = sent.lower()
    return sum(1 for h in _AGENDA_HINTS if h.lower() in s)

def extract_agenda(segments, max_items: int = 5):
    """
    최근 대화에서 '안건 후보' 문장을 뽑아 상위 N개 반환.
    - 힌트 키워드 매칭 + 간단한 빈도 가중치
    - 겹치는 내용/중복 화자는 제거
    """
    candidates = []
    for seg in segments:
        text = getattr(seg, "text", "").strip()
        if not text:
            continue
        text = _normalize(text)
        # 문장 분할(., ?, !, ~, 끝 조사 기준 간단 분할)
        parts = re.split(r"[\.?!~]\s+|(?<=다)\s+|(?<=요)\s+", text)
        for p in parts:
            p = p.strip()
            if len(p) < 6:
                continue
            sc = _score_sentence(p)
            if sc > 0:
                # 너무 일반적인 말 줄이기
                if re.search(r"(했습니다|하겠습니다|좋겠습니다|같습니다)$", p):
                    sc -= 1
                candidates.append((sc, p))

    # 스코어 → 길이 보정(너무 긴 문장 불이익), 상위 N
    ranked = sorted(
        ((sc - 0.001*len(p), p) for sc, p in candidates if sc > 0),
        key=lambda x: x[0],
        reverse=True
    )

    # 중복/유사 제거
    out, seen = [], set()
    for _, p in ranked:
        k = re.sub(r"\s+", " ", p)[:40]
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) >= max_items:
            break

    # 후보가 없으면, 빈도 상위 키워드로 fallback
    if not out:
        bag = Counter()
        for seg in segments:
            for h in _AGENDA_HINTS:
                if h in getattr(seg, "text", ""):
                    bag[h] += 1
        out = [kw for kw, _ in bag.most_common(max_items)] or ["일반 진행 사항"]

    return out

def llm_summarize(
    segments: List[Segment],
    backend: Optional[str] = None,
    max_tokens: int = 3000
) -> str:
    """
    LLM을 사용하여 회의록 요약 생성

    Args:
        segments: 회의 세그먼트 리스트
        backend: LLM 백엔드 (예: "openai:gpt-4o-mini", None이면 기본값)
        max_tokens: 요약에 사용할 최대 토큰 수

    Returns:
        str: LLM이 생성한 회의록 요약
    """
    if not segments:
        return "요약할 내용이 없습니다."

    # 전사 내용을 텍스트로 변환 (토큰 제한 고려)
    transcript_lines = []
    for s in segments:
        if s.text.strip():
            speaker_display = s.speaker_name if s.speaker_name != "Unknown" else "speaker_00"
            transcript_lines.append(f"[{speaker_display}] {s.text}")

    transcript = "\n".join(transcript_lines)

    # 토큰 제한을 위해 텍스트 길이 제한 (대략 1토큰 = 2-3자)
    max_chars = max_tokens * 2
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n... (내용 생략)"

    # 프롬프트 구성
    prompt = f"""당신은 기업 회의의 내용을 명확하고 간결하게 정리하는 비서입니다.

다음 회의 전사를 읽고 아래 형식으로 요약하세요.
추측하거나 꾸미지 말고, 실제 내용만 반영하세요.

회의 전사:
{transcript}

다음 형식으로 요약해주세요:

📋 회의 주요 내용
(3-5개의 핵심 논의 사항을 bullet point로 정리)

🎯 주요 결정 사항
(회의에서 결정된 사항들을 정리, 없으면 "없음"으로 표시)

📌 액션 아이템
(각 참석자별 할 일과 기한을 정리, 없으면 "없음"으로 표시)

💡 기타 특이사항
(주목할 만한 내용이나 추가 논의가 필요한 사항, 없으면 생략)

요약은 간결하고 명확하게 작성해주세요."""

    try:
        if LLMRouter is None:
            return f"⚠️ LLM 모듈을 사용할 수 없습니다.\n\n기본 요약:\n{simple_summarize(segments, max_len=15)}"

        router = LLMRouter(default_backend=backend or "openai:gpt-4o-mini")
        summary = router.complete(backend, prompt, temperature=0.3)
        return summary
    except Exception as e:
        return f"⚠️ LLM 요약 생성 실패: {str(e)}\n\n기본 요약으로 대체합니다.\n\n{simple_summarize(segments, max_len=15)}"


def llm_summarize_with_rag(
    query: str,
    rag_store: 'RagStore',
    speaker_id: Optional[str] = None,
    backend: Optional[str] = None,
    topk: int = 5
) -> str:
    """
    RAG를 활용한 컨텍스트 기반 LLM 요약 생성

    Args:
        query: 검색 쿼리 (예: "데이터베이스 최적화 방안")
        rag_store: RagStore 인스턴스
        speaker_id: 특정 화자로 필터링 (선택)
        backend: LLM 백엔드
        topk: 검색할 컨텍스트 수

    Returns:
        str: RAG 컨텍스트 기반 요약
    """
    if not RAG_AVAILABLE:
        return "⚠️ RAG 기능을 사용할 수 없습니다. qdrant-client와 sentence-transformers를 설치하세요."

    if not rag_store or not rag_store.ok:
        return "⚠️ RAG Store가 초기화되지 않았습니다."

    # RAG에서 관련 컨텍스트 검색
    try:
        results = rag_store.search(
            query=query,
            topk=topk,
            speaker_id=speaker_id
        )

        if not results:
            return f"'{query}'에 대한 관련 정보를 찾을 수 없습니다."

        # 컨텍스트 포맷팅
        context_lines = []
        for i, r in enumerate(results, 1):
            speaker = r.get('speaker_name', 'Unknown')
            text = r.get('text', '')
            score = r.get('_score', 0.0)
            context_lines.append(f"{i}. [{speaker}] {text} (관련도: {score:.2f})")

        context_block = "\n".join(context_lines)

        # LLM 프롬프트 구성
        prompt = f"""다음은 회의록 데이터베이스에서 검색한 관련 발언들입니다.
이 내용을 바탕으로 '{query}'에 대한 종합적인 답변을 작성해주세요.

[검색된 관련 발언]
{context_block}

요구사항:
1. 위 발언들의 핵심 내용을 종합하여 설명
2. 각 화자의 주요 의견이나 제안 사항 정리
3. 실행 가능한 액션 아이템이 있다면 포함
4. 간결하고 명확하게 작성 (3-5 문단)

답변:"""

        # LLM 호출
        if LLMRouter is None:
            return f"⚠️ LLM 모듈을 사용할 수 없습니다.\n\n[검색된 컨텍스트]\n{context_block}"

        router = LLMRouter(default_backend=backend or "openai:gpt-4o-mini")
        response = router.complete(backend, prompt, temperature=0.3)

        # 응답에 출처 추가
        sources = "\n\n[참조한 발언]\n" + "\n".join([
            f"- [{r.get('speaker_name', 'Unknown')}] {r.get('text', '')[:80]}..."
            for r in results[:3]
        ])

        return response + sources

    except Exception as e:
        return f"⚠️ RAG 검색 중 오류 발생: {str(e)}"


def get_speaker_context_summary(
    rag_store: 'RagStore',
    speaker_id: str,
    topic: Optional[str] = None,
    backend: Optional[str] = None
) -> Dict:
    """
    특정 화자의 발언 패턴과 전문성 요약

    Args:
        rag_store: RagStore 인스턴스
        speaker_id: 화자 ID
        topic: 특정 주제로 필터링 (선택)
        backend: LLM 백엔드

    Returns:
        Dict: 화자 통계 및 요약 정보
    """
    if not RAG_AVAILABLE or not rag_store or not rag_store.ok:
        return {"error": "RAG Store를 사용할 수 없습니다."}

    try:
        # 화자 통계
        stats = rag_store.get_speaker_stats(speaker_id)

        if not stats or stats.get('total_utterances', 0) == 0:
            return {"error": f"화자 '{speaker_id}'의 발언을 찾을 수 없습니다."}

        # 화자의 최근 발언 검색
        query = topic if topic else speaker_id
        utterances = rag_store.search_by_speaker(speaker_id, query=query, topk=10)

        # 발언 내용 종합
        texts = [u.get('text', '') for u in utterances]
        combined_text = "\n".join([f"- {t}" for t in texts[:5]])

        # LLM으로 화자 특성 분석
        prompt = f"""다음은 '{stats.get('speaker_name', speaker_id)}' 화자의 최근 발언들입니다:

{combined_text}

이 화자의 특징을 다음 관점에서 간단히 분석해주세요:
1. 주요 관심 분야 또는 전문성
2. 대화 스타일 (기술적/전략적/실무적 등)
3. 자주 언급하는 키워드나 주제

3-4문장으로 요약해주세요."""

        try:
            if LLMRouter is not None:
                router = LLMRouter(default_backend=backend or "openai:gpt-4o-mini")
                analysis = router.complete(backend, prompt, temperature=0.3)
            else:
                analysis = "LLM 모듈을 사용할 수 없어 분석을 생략합니다."
        except Exception:
            analysis = "분석 정보를 생성할 수 없습니다."

        return {
            "speaker_id": speaker_id,
            "speaker_name": stats.get('speaker_name', speaker_id),
            "total_utterances": stats.get('total_utterances', 0),
            "avg_length": f"{stats.get('avg_length', 0):.1f}자",
            "total_duration": f"{stats.get('total_duration', 0):.1f}초",
            "analysis": analysis,
            "recent_topics": [u.get('text', '')[:50] + "..." for u in utterances[:3]]
        }

    except Exception as e:
        return {"error": f"화자 분석 중 오류: {str(e)}"}


def extract_schedules_from_summary(
    summary_text: str,
    segments: List[Segment],
    backend: Optional[str] = None
) -> List[Dict]:
    """
    LLM을 사용하여 회의 요약에서 일정, 마감일, TODO를 추출

    Args:
        summary_text: 회의 요약 텍스트
        segments: 원본 회의 세그먼트 (추가 컨텍스트용)
        backend: LLM 백엔드 (예: "openai:gpt-4o-mini")

    Returns:
        List[Dict]: 추출된 일정 목록
        [
            {
                "title": "API 개발 완료",
                "date": "2025-01-20",
                "time": "14:00",  # 선택적
                "type": "project",  # "meeting", "project", "todo"
                "description": "RESTful API 개발 및 문서화",
                "assignee": "김개발"  # 선택적
            },
            ...
        ]
    """
    if not summary_text:
        return []

    # 현재 날짜 정보
    today = datetime.datetime.now()
    current_date = today.strftime("%Y-%m-%d")
    current_weekday = ["월", "화", "수", "목", "금", "토", "일"][today.weekday()]

    # 프롬프트 구성
    prompt = f"""당신은 회의록에서 일정과 할 일을 추출하는 전문가입니다.

오늘 날짜: {current_date} ({current_weekday}요일)

다음 회의 요약에서 **일정, 마감일, TODO**를 모두 찾아서 JSON 배열로 추출해주세요.

회의 요약:
{summary_text}

추출 규칙:
1. 명확한 날짜나 기한이 있는 항목만 추출
2. "다음주 월요일", "내일", "이번주 금요일" 같은 상대 날짜는 구체적인 날짜로 변환
3. 시간이 명시되어 있으면 포함, 없으면 null
4. 담당자가 명시되어 있으면 포함, 없으면 null
5. 타입은 다음 중 하나로 분류:
   - "meeting": 회의, 미팅
   - "project": 프로젝트, 개발, 작업
   - "todo": 일반 할 일
   - "deadline": 마감, 제출

응답 형식 (JSON 배열만 출력, 다른 설명 없이):
[
  {{
    "title": "작업 제목",
    "date": "YYYY-MM-DD",
    "time": "HH:MM" 또는 null,
    "type": "meeting|project|todo|deadline",
    "description": "상세 설명",
    "assignee": "담당자 이름" 또는 null
  }}
]

중요:
- 반드시 유효한 JSON 배열만 출력하세요
- 일정이 없으면 빈 배열 [] 을 반환하세요
- 추측하지 말고 명확한 일정만 추출하세요"""

    try:
        if LLMRouter is None:
            print("[WARN] LLMRouter를 사용할 수 없어 일정 추출을 건너뜁니다.")
            return []

        router = LLMRouter(default_backend=backend or "openai:gpt-4o-mini")
        response = router.complete(backend, prompt, temperature=0.1)

        # JSON 파싱 (응답에서 코드 블록 제거)
        response = response.strip()
        if response.startswith("```"):
            # 코드 블록 제거
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            response = response.replace("```json", "").replace("```", "").strip()

        # JSON 파싱
        import json
        schedules = json.loads(response)

        if not isinstance(schedules, list):
            print(f"[WARN] LLM 응답이 리스트가 아닙니다: {type(schedules)}")
            return []

        # 유효성 검증
        valid_schedules = []
        for sch in schedules:
            if not isinstance(sch, dict):
                continue

            # 필수 필드 확인
            if "title" not in sch or "date" not in sch:
                continue

            # 날짜 형식 검증
            try:
                datetime.datetime.strptime(sch["date"], "%Y-%m-%d")
            except ValueError:
                print(f"[WARN] 잘못된 날짜 형식: {sch.get('date')}")
                continue

            # 타입 기본값
            if "type" not in sch or sch["type"] not in ["meeting", "project", "todo", "deadline"]:
                sch["type"] = "todo"

            valid_schedules.append(sch)

        print(f"[INFO] LLM이 {len(valid_schedules)}개의 일정을 추출했습니다.")
        return valid_schedules

    except json.JSONDecodeError as e:
        print(f"[ERROR] LLM 응답 JSON 파싱 실패: {e}")
        print(f"[DEBUG] LLM 응답: {response[:500]}")
        return []
    except Exception as e:
        print(f"[ERROR] 일정 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return []
