# ---- 아래를 파일 맨 아래쪽에 추가하세요 ----
from html import escape
from collections import Counter
import re
import datetime

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

def simple_summarize(segments: List[Segment], max_len=10) -> str:
    lines = []
    for s in segments:
        if s.text.strip():
            # speaker_XX 형태 그대로 표시
            speaker_display = s.speaker_name
            if speaker_display == "Unknown":
                speaker_display = "speaker_00"

            lines.append(f"[{speaker_display}] {s.text}")

    return "\n".join(lines[-max_len:]) if lines else "요약할 내용이 없습니다."


def extract_actions(segments: List[Segment]) -> List[str]:
    acts = []
    for s in segments:
        if any(v in s.text for v in ACTION_VERBS):
            deadline = dateparser.parse(s.text, languages=["ko"])
            dstr = f" (기한: {deadline.strftime('%Y-%m-%d %H:%M')})" if deadline else ""
            acts.append(f"- [{s.speaker_name}] {s.text}{dstr}")
    uniq, seen = [], set()
    for a in acts:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq

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
    prompt = f"""다음은 회의 전사 내용입니다. 이 회의 내용을 분석하여 전문적인 회의록 요약을 작성해주세요.

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
        router = LLMRouter(default_backend=backend or "openai:gpt-4o-mini")
        summary = router.complete(backend, prompt, temperature=0.3)
        return summary
    except Exception as e:
        return f"⚠️ LLM 요약 생성 실패: {str(e)}\n\n기본 요약으로 대체합니다.\n\n{simple_summarize(segments, max_len=15)}"