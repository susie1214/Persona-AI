# core/summarizer.py
import dateparser
from typing import List
from .audio import Segment

ACTION_VERBS = [
    "해야",
    "해주세요",
    "진행",
    "확인",
    "정리",
    "검토",
    "공유",
    "작성",
    "업로드",
    "보고",
    "회의",
    "예약",
    "훈련",
    "배포",
    "테스트",
    "구매",
    "설치",
]


def simple_summarize(segments: List[Segment], max_len=10) -> str:
    lines = [f"[{s.speaker_name}] {s.text}" for s in segments if s.text]
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
