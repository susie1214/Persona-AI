# -*- coding: utf-8 -*-
# core/offline_meeting.py
"""
오프라인 파일 처리 유틸:
- process_audio_file(path, ...) : 오디오/비디오 파일을 받아 STT(+옵션으로 화자분리) 후
  마크다운 회의록과 JSON을 반환.
"""

import os
import json
import time
import datetime
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


# ---------- Helper Functions ----------

def _whisper_device(use_gpu: bool) -> str:
    """GPU 사용 여부에 따라 디바이스 결정"""
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"

    # macOS는 CUDA 지원 안함
    import platform
    if platform.system() == "Darwin":
        return "cpu"

    return "cuda" if use_gpu else "cpu"


def _ensure_dirs():
    """출력 디렉터리 생성"""
    os.makedirs("output/meetings", exist_ok=True)


def _extract_segment_data(segment) -> Dict:
    """
    Whisper segment에서 데이터 안전하게 추출
    segment는 dict 또는 NamedTuple/object 가능
    """
    try:
        # dict인 경우
        if isinstance(segment, dict):
            return {
                "text": segment.get("text", "").strip(),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
            }

        # object/NamedTuple인 경우
        text = ""
        start = 0.0
        end = 0.0

        # text 속성 찾기
        if hasattr(segment, "text"):
            text = getattr(segment, "text", "")
        elif hasattr(segment, "get"):
            text = segment.get("text", "")

        # start 속성 찾기
        if hasattr(segment, "start"):
            start = getattr(segment, "start", 0.0)
        elif hasattr(segment, "get"):
            start = segment.get("start", 0.0)

        # end 속성 찾기
        if hasattr(segment, "end"):
            end = getattr(segment, "end", 0.0)
        elif hasattr(segment, "get"):
            end = segment.get("end", 0.0)

        return {
            "text": (text or "").strip(),
            "start": float(start or 0.0),
            "end": float(end or 0.0),
        }

    except Exception as e:
        print(f"[WARNING] Failed to extract segment data: {e}")
        print(f"[WARNING] Segment type: {type(segment)}")
        print(f"[WARNING] Segment: {segment}")
        return {
            "text": "",
            "start": 0.0,
            "end": 0.0,
        }


def _simple_summarize(segments: List[Dict], max_len: int = 12) -> str:
    """간단한 요약 생성"""
    lines = []
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            speaker = s.get("speaker", "?")
            lines.append(f"[{speaker}] {text}")

    return "\n".join(lines[-max_len:]) if lines else "요약할 내용이 없습니다."


def _extract_actions(segments: List[Dict]) -> List[str]:
    """액션 아이템 추출"""
    action_verbs = [
        "해야", "해주세요", "진행", "확인", "정리", "검토",
        "공유", "작성", "업로드", "보고", "회의", "예약",
        "훈련", "배포", "테스트", "구매", "설치"
    ]

    actions = []
    for s in segments:
        text = s.get("text", "")
        if any(verb in text for verb in action_verbs):
            speaker = s.get("speaker", "Unknown")
            actions.append(f"- [{speaker}] {text}")

    # 중복 제거
    return list(dict.fromkeys(actions))


def _md_from_segments(title: str, segs: List[Dict]) -> Dict[str, str]:
    """세그먼트에서 마크다운 문서 생성"""
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
    md_lines.append("```")
    md_lines.append(summary)
    md_lines.append("```")
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
        speaker = s.get("speaker", "Unknown")
        text = s.get("text", "")
        md_lines.append(f"- **[{speaker}]({tstr})**: {text}")

    return {
        "markdown": "\n".join(md_lines),
        "summary": summary,
        "actions_text": "\n".join(actions) if actions else "",
    }


def _match_speakers_by_overlap(whisper_segments: List[Dict], diar_annotation) -> List[Dict]:
    """
    Whisper segments와 Diarization 결과를 매칭
    whisper_segments: List[Dict] with keys: text, start, end
    diar_annotation: pyannote Annotation
    return: List[Dict] with keys: speaker, text, start, end
    """
    out = []

    # Diarization turns 추출
    diar_turns = []
    if diar_annotation:
        try:
            diar_turns = list(diar_annotation.itertracks(yield_label=True))
        except Exception as e:
            print(f"[WARNING] Failed to extract diarization turns: {e}")
            diar_turns = []

    for seg in whisper_segments:
        w_start = seg["start"]
        w_end = seg["end"]
        w_text = seg["text"]

        speaker = "Unknown"

        # 가장 많이 겹치는 화자 찾기
        best_overlap = 0.0
        best_spk = None

        for turn, _, spk in diar_turns:
            s = float(turn.start)
            e = float(turn.end)
            overlap = max(0.0, min(w_end, e) - max(w_start, s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = spk

        if best_spk:
            speaker = str(best_spk)

        out.append({
            "speaker": speaker,
            "text": w_text,
            "start": w_start,
            "end": w_end,
        })

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
    오디오 파일 경로를 받아 STT(+옵션: 화자분리) 수행 후
    - segments(list[dict]), markdown(str), title(str), json_path(str) 를 반환

    Args:
        path: 입력 파일 경로 (오디오/비디오)
        asr_model: Whisper 모델명 (tiny/base/small/medium/large)
        use_gpu: GPU 사용 여부 (macOS는 무시됨)
        diarize: 화자 분리 사용 여부
        settings: 추가 설정 (선택)

    Returns:
        Dict with keys: title, markdown, json_path, segments

    Raises:
        RuntimeError: faster-whisper 미설치
        FileNotFoundError: 입력 파일 없음
    """
    _ensure_dirs()

    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper가 설치되어 있지 않습니다. (pip install faster-whisper)"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {path}")

    # 디바이스 설정
    device = _whisper_device(use_gpu)
    compute = "int8" if device == "cpu" else "float16"

    print(f"[INFO] Processing: {path}")
    print(f"[INFO] Device: {device}, Compute: {compute}")
    print(f"[INFO] Model: {asr_model}")

    # 1) Whisper STT
    try:
        model = WhisperModel(asr_model, device=device, compute_type=compute)
    except Exception as e:
        print(f"[WARNING] Failed to load model with {device}, falling back to CPU")
        print(f"[WARNING] Error: {e}")
        model = WhisperModel(asr_model, device="cpu", compute_type="int8")

    print("[INFO] Transcribing audio...")
    segments_generator, info = model.transcribe(
        path,
        vad_filter=True,
        language="ko"
    )

    # Generator를 list로 변환하면서 안전하게 데이터 추출
    whisper_segments = []
    for seg in segments_generator:
        seg_data = _extract_segment_data(seg)
        if seg_data["text"]:  # 빈 텍스트는 제외
            whisper_segments.append(seg_data)

    print(f"[INFO] Extracted {len(whisper_segments)} segments")

    # 2) Diarization (optional)
    diar_annotation = None
    if diarize and PyannotePipeline is not None:
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if hf_token:
            try:
                print("[INFO] Running speaker diarization...")
                pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                diar_annotation = pipeline(path)
                print("[INFO] Diarization completed")
            except Exception as e:
                print(f"[WARNING] Diarization failed: {e}")
                diar_annotation = None
        else:
            print("[WARNING] HF_TOKEN not set, skipping diarization")

    # 3) 화자 매칭
    if diar_annotation is not None:
        print("[INFO] Matching speakers...")
        merged = _match_speakers_by_overlap(whisper_segments, diar_annotation)
    else:
        print("[INFO] No diarization, marking all as Unknown")
        merged = []
        for seg in whisper_segments:
            merged.append({
                "speaker": "Unknown",
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            })

    # 4) 마크다운 생성
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

    print(f"[INFO] Saved to: {json_path}")

    # UI에서 사용하기 편하도록 반환
    return {
        "title": title,
        "markdown": md_bundle["markdown"],
        "json_path": json_path,
        "segments": merged,
    }
