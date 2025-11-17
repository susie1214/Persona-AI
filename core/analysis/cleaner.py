# core/text_clean.py
import re

FILLERS = [
    r"\b음+\b", r"\b어+\b", r"\b아+\b", r"\b네\b", r"\b그+렇+죠\b", r"\b맞죠\b"
]
TEST_MARKERS = [
    r"\b테스트\b", r"\b쉬었다가\b", r"\b스피커\s?\d+\b", r"\b영어로 한국어로\b",
    r"\b다음 영상에서 만나요\b", r"\bstatus\b", r"\b프리뷰\b"
]

def clean_utterances(utterances):
    """
    utterances: List[dict] = [{speaker, text, start, end}, ...]
    return: List[dict]
    """
    out = []
    for u in utterances:
        t = u["text"]
        # 괄호식 타임스탬프/태그 제거
        t = re.sub(r"\([^)]*\)", " ", t)
        t = re.sub(r"\[[^\]]*\]", " ", t)
        # 반복 구두점/공백 정리
        t = re.sub(r"\s+", " ", t).strip()
        # 채우기말/테스트 멘트 필터
        if any(re.search(p, t, re.IGNORECASE) for p in TEST_MARKERS): 
            continue
        for f in FILLERS:
            t = re.sub(f, "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) >= 3:
            out.append({**u, "text": t})
    return out

def merge_short(utterances, threshold=18):
    """짧은 문장들 붙여서 의미 단위로 만듦"""
    merged, buf = [], []
    total_len = 0
    for u in utterances:
        if total_len + len(u["text"]) <= threshold and (not buf or buf[-1]["speaker"]==u["speaker"]):
            buf.append(u); total_len += len(u["text"])
        else:
            if buf:
                merged.append({
                    "speaker": buf[0]["speaker"],
                    "start": buf[0]["start"],
                    "end": buf[-1]["end"],
                    "text": " ".join(x["text"] for x in buf)
                })
            buf = [u]; total_len = len(u["text"])
    if buf:
        merged.append({
            "speaker": buf[0]["speaker"], "start": buf[0]["start"], "end": buf[-1]["end"],
            "text": " ".join(x["text"] for x in buf)
        })
    return merged
