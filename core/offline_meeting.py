# -*- coding: utf-8 -*-
# core/offline_meeting.py
"""
오프라인 파일 처리 유틸:
- process_audio_file(path, ...) : 오디오/비디오 파일을 받아 STT(+옵션으로 화자분리) 후
  마크다운 회의록과 JSON을 반환.
"""

import os, json, time, datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

# ----- Optional deps guard -----
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

# 프로젝트 내 요약/액션 도우미 재사용 (있으면 사용, 없으면 내부 간이 버전)
try:
    from core.summarizer import (
        extract_actions as _extract_actions,
        simple_summarize as _simple_summarize,
    )
except Exception:

    def _extract_actions(segments: List[Dict]) -> List[str]:
        verbs = [
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
        out = []
        for s in segments:
            t = s.get("text", "")
            if any(v in t for v in verbs):
                out.append(f"- [{s.get('speaker','Unknown')}] {t}")
        return list(dict.fromkeys(out))

    def _simple_summarize(segments: List[Dict], max_len=12) -> str:
        lines = [
            f"[{s.get('speaker','?')}] {s.get('text','')}"
            for s in segments
            if s.get("text")
        ]
        return "\n".join(lines[-max_len:]) if lines else "요약할 내용이 없습니다."


# ---------- Helper ----------
def _whisper_device(use_gpu: bool) -> str:
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda" if use_gpu else "cpu"


def _ensure_dirs():
    os.makedirs("output/meetings", exist_ok=True)


def _md_from_segments(title: str, segs: List[Dict]) -> Dict[str, str]:
    summary = _simple_summarize(segs, max_len=14)
    actions = _extract_actions(segs)
    md_lines = []
    md_lines.append(f"# {title}")
    md_lines.append("")
    md_lines.append(f"- 생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"- 총 발언 수: {len(segs)}")
    md_lines.append("")
    md_lines.append("## 요약(최근순)")
    md_lines.append("")
    md_lines.append("```\n" + summary + "\n```")
    md_lines.append("")
    if actions:
        md_lines.append("## Action Items")
        md_lines.extend(actions)
        md_lines.append("")
    md_lines.append("## 전체 대화 로그")
    for s in segs:
        st = s.get("start", 0.0)
        m, sec = divmod(int(st), 60)
        tstr = f"{m:02d}:{sec:02d}"
        md_lines.append(
            f"- **[{s.get('speaker','Unknown')}]({tstr})**: {s.get('text','')}"
        )
    return {
        "markdown": "\n".join(md_lines),
        "summary": summary,
        "actions_text": "\n".join(actions) if actions else "",
    }


def _match_speakers_by_overlap(whisper_segments, diar_annotation) -> List[Dict]:
    """
    whisper_segments: iterable of objects with .start, .end, .text (또는 dict)
    diar_annotation: pyannote Annotation
    return list of dicts: {speaker, text, start, end}
    """
    out = []
    # diar.itertracks(yield_label=True) → (turn, track, label)
    diar_turns = (
        list(diar_annotation.itertracks(yield_label=True)) if diar_annotation else []
    )
    for ws in whisper_segments:
        # dict와 object 모두 지원
        if isinstance(ws, dict):
            w_start = float(ws.get("start", 0.0))
            w_end = float(ws.get("end", 0.0))
            w_text = ws.get("text", "").strip()
        else:
            w_start = float(ws.start or 0.0)
            w_end = float(ws.end or 0.0)
            w_text = (ws.text or "").strip()

        speaker = "Unknown"
        # 가장 많이 겹치는 turn을 선택
        best_overlap = 0.0
        best_spk = None
        for turn, _, spk in diar_turns:
            s, e = float(turn.start), float(turn.end)
            overlap = max(0.0, min(w_end, e) - max(w_start, s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = spk
        if best_spk:
            speaker = str(best_spk)
        out.append(
            {
                "speaker": speaker,
                "text": w_text,
                "start": w_start,
                "end": w_end,
            }
        )
    return out


# ---------- Public API ----------
def process_audio_file(
    path: str,
    asr_model: str = "small",
    use_gpu: bool = True,
    diarize: bool = True,
    settings: Optional[Dict] = None,
) -> Dict:
    """
    오디오/비디오 파일 경로를 받아 STT(+옵션: 화자분리) 수행 후
    - segments(list[dict]), markdown(str), title(str), json_path(str) 를 반환

    requirements:
      - faster-whisper (필수)
      - ffmpeg (파일 디코딩에 필요, 시스템에 설치되어 있어야 함)
      - pyannote.audio (선택) + HF_TOKEN 환경변수 (diarize=True일 때)
    """
    _ensure_dirs()
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper 가 설치되어 있지 않습니다. (pip install faster-whisper)"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {path}")

    device = _whisper_device(use_gpu)
    compute = "int8" if device == "cpu" else "float16"

    # 1) Whisper STT
    model = WhisperModel(asr_model, device=device, compute_type=compute)
    # faster-whisper는 ffmpeg를 사용해 비디오/오디오 파일 모두 인식 가능
    # language 추정 자동, 한국어 고정하고 싶으면 language="ko"
    segments, info = model.transcribe(path, vad_filter=True, language="ko")

    seg_list = list(segments)  # generator → list

    # 디버그: 첫 번째 segment 타입 확인
    if seg_list:
        print(f"[DEBUG] First segment type: {type(seg_list[0])}")
        print(f"[DEBUG] First segment: {seg_list[0]}")
        if isinstance(seg_list[0], dict):
            print(f"[DEBUG] Segment is dict with keys: {seg_list[0].keys()}")
        else:
            print(f"[DEBUG] Segment attributes: {dir(seg_list[0])}")

    # 2) (optional) Diarization
    diar_annotation = None
    if diarize:
        if PyannotePipeline is None:
            # 화자분리 미설치 시, Unknown으로 처리
            diar_annotation = None
        else:
            hf_token = os.getenv("HF_TOKEN", "").strip()
            try:
                pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token or None
                )
                # pyannote도 ffmpeg로 처리 가능한 경로 입력을 허용함
                diar_annotation = pipeline(path)
            except Exception as e:
                # 실패 시 diarization 없이 진행
                diar_annotation = None

    # 3) 스피커 매칭(있으면), 없으면 Unknown
    if diar_annotation is not None:
        merged = _match_speakers_by_overlap(seg_list, diar_annotation)
    else:
        merged = []
        for s in seg_list:
            # dict와 object 모두 지원
            if isinstance(s, dict):
                merged.append({
                    "speaker": "Unknown",
                    "text": s.get("text", "").strip(),
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                })
            else:
                merged.append({
                    "speaker": "Unknown",
                    "text": (s.text or "").strip(),
                    "start": float(s.start or 0.0),
                    "end": float(s.end or 0.0),
                })

    # 4) 마크다운/요약/액션
    title = os.path.splitext(os.path.basename(path))[0] or "회의록"
    md_bundle = _md_from_segments(title, merged)

    # 5) JSON 저장
    ts = int(time.time())
    json_obj = {
        "title": title,
        "source_path": os.path.abspath(path),
        "created_at": datetime.datetime.now().isoformat(),
        "segments": merged,
        "summary": md_bundle.get("summary", ""),
        "actions": md_bundle.get("actions_text", ""),
        "asr_model": asr_model,
        "device": device,
        "diarization": bool(diar_annotation is not None),
    }
    json_path = os.path.join("output", "meetings", f"meeting_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    # UI에서 사용하기 편하도록 반환
    return {
        "title": title,
        "markdown": md_bundle["markdown"],
        "json_path": json_path,
        "segments": merged,
    }
