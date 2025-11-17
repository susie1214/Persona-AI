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
import numpy as np
from typing import List, Dict, Optional
from core.audio import Segment

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

def _dict_to_segment(seg_dict: Dict) -> Segment:
    """
    Dict 형식의 세그먼트를 Segment 객체로 변환

    Args:
        seg_dict: {"speaker": str, "text": str, "start": float, "end": float}

    Returns:
        Segment 객체
    """
    return Segment(
        text=seg_dict.get("text", ""),
        start=seg_dict.get("start", 0.0),
        end=seg_dict.get("end", 0.0),
        speaker_name=seg_dict.get("speaker", "Unknown")
    )


def _whisper_device(use_gpu: bool) -> str:
    """
    GPU 사용 여부에 따라 디바이스 결정
    - macOS (Apple Silicon): mps (Metal Performance Shaders)
    - Linux/Windows (NVIDIA): cuda
    - CPU fallback
    """
    # if os.getenv("FORCE_CPU", "0") == "1":
    #     return "cpu"

    # if not use_gpu:
    #     return "cpu"

    import platform
    import torch

    # macOS에서 Apple Silicon (M1/M2/M3) MPS 지원
    if platform.system() == "Darwin":
        # MPS 사용 가능 여부 확인 (PyTorch 1.12+ 필요)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[INFO] Using Apple Silicon MPS for acceleration")
            return "mps"
        else:
            print("[WARNING] MPS not available, falling back to CPU")
            return "cpu"

    # CUDA 지원 (Linux/Windows)
    if torch.cuda.is_available():
        return "cuda"
    elif not use_gpu or os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"


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
    """간단한 요약 생성 (LLM 없이)"""
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


def _md_from_segments(title: str, segs: List[Dict], use_llm_summary: bool = False, llm_backend: Optional[str] = None) -> Dict[str, str]:
    """세그먼트에서 마크다운 문서 생성"""

    md_lines = []
    md_lines.append(f"# {title}")
    md_lines.append("")
    # md_lines.append(f"- 생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"- 총 발언 수: {len(segs)}")
    # md_lines.append(f"- 요약 방식: {'AI 요약 (LLM)' if use_llm_summary else '기본 요약'}")
    md_lines.append("")
    
    summary = f"# {title}\n\n"
    
    # LLM 요약 또는 간단한 요약 사용
    if use_llm_summary:
        print("[INFO] Generating LLM-based summary...")
        # Dict를 Segment 객체로 변환
        from core.analysis import llm_summarize
        segment_objects = [_dict_to_segment(s) for s in segs]
        summary += llm_summarize(segment_objects, backend=llm_backend)
    else:
        summary += _simple_summarize(segs, max_len=14)

    actions = _extract_actions(segs)

    # md_lines.append("## 요약")
    # md_lines.append("")
    
    # print(f"[DEBUG - offline_meeting-_md_from_segments] summary : {summary}")
    
    # if use_llm_summary:
    #     # LLM 요약은 이미 마크다운 형식으로 되어 있음
    #     md_lines.append(summary)
    # else:
    #     # 기본 요약은 코드 블록으로 감싸기
    #     md_lines.append("```")
    #     md_lines.append(summary)
    #     md_lines.append("```")
    # md_lines.append("")

    # if actions:
    #     md_lines.append("## Action Items (자동 추출)")
    #     md_lines.extend(actions)
    #     md_lines.append("")

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


def _match_speakers_by_overlap(
    whisper_segments: List[Dict],
    diar_annotation,
    pipeline=None,
    speaker_manager=None,
    audio_path: Optional[str] = None
) -> List[Dict]:
    """
    Whisper segments와 Diarization 결과를 매칭
    whisper_segments: List[Dict] with keys: text, start, end
    diar_annotation: pyannote Annotation
    pipeline: pyannote Pipeline (임베딩 추출용)
    speaker_manager: SpeakerManager 인스턴스
    audio_path: 오디오 파일 경로 (임베딩 추출용)
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

    # 임베딩 추출 (가능한 경우)
    embeddings = None
    if pipeline and audio_path and speaker_manager and speaker_manager.voice_store.ok:
        try:
            # pyannote pipeline에서 임베딩 모델 추출 (다양한 접근 방법 시도)
            embedding_model = None

            # 방법 1: _models 속성에서 임베딩 모델 가져오기
            if hasattr(pipeline, '_models') and 'embedding' in pipeline._models:
                embedding_model = pipeline._models['embedding']
                print("[INFO] Found embedding model in pipeline._models")
            # 방법 2: embedding 속성 직접 접근
            elif hasattr(pipeline, 'embedding'):
                embedding_model = pipeline.embedding
                print("[INFO] Found embedding model in pipeline.embedding")
            # 방법 3: embedding_model 속성 (이전 버전 호환)
            elif hasattr(pipeline, 'embedding_model'):
                embedding_model = pipeline.embedding_model
                print("[INFO] Found embedding model in pipeline.embedding_model")

            if embedding_model:
                embeddings = embedding_model(audio_path)
                print("[INFO] Successfully extracted embeddings for speaker identification")
            else:
                print("[WARNING] Could not find embedding model in pipeline")
        except Exception as e:
            print(f"[WARNING] Failed to extract embeddings: {e}")
            import traceback
            traceback.print_exc()

    for seg in whisper_segments:
        w_start = seg["start"]
        w_end = seg["end"]
        w_text = seg["text"]

        speaker = "Unknown"

        # 가장 많이 겹치는 화자 찾기
        best_overlap = 0.0
        best_turn = None
        best_pyannote_spk = None

        for turn, _, spk in diar_turns:
            s = float(turn.start)
            e = float(turn.end)
            overlap = max(0.0, min(w_end, e) - max(w_start, s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
                best_pyannote_spk = spk

        # 임베딩 기반 화자 식별 시도
        if best_turn and embeddings is not None and speaker_manager:
            try:
                # 해당 시간 구간의 임베딩 추출
                segment_embedding = embeddings.crop(best_turn)
                segment_embedding = np.mean(segment_embedding, axis=0)

                # SpeakerManager로 화자 식별
                speaker_id, confidence = speaker_manager.identify_speaker(segment_embedding, threshold=0.58)
                # 화자 ID를 표시 이름으로 변환
                speaker = speaker_manager.get_speaker_display_name(speaker_id)
                print(f"[INFO] Identified speaker: {speaker_id} -> {speaker} (confidence: {confidence:.3f})")

            except Exception as e:
                print(f"[WARNING] Speaker identification failed, using pyannote label: {e}")
                speaker = str(best_pyannote_spk) if best_pyannote_spk else "Unknown"
        elif best_pyannote_spk:
            # 임베딩을 사용할 수 없는 경우: pyannote 레이블을 화자 ID로 변환
            # SPEAKER_00 -> speaker_00 형식으로 통일
            pyannote_label = str(best_pyannote_spk)

            # pyannote 레이블에서 숫자 추출 (SPEAKER_00 -> 0)
            import re
            match = re.search(r'(\d+)', pyannote_label)
            if match and speaker_manager:
                speaker_num = int(match.group(1))
                speaker = f"speaker_{speaker_num:02d}"
                print(f"[INFO] Mapped pyannote label '{pyannote_label}' to '{speaker}'")

                # SpeakerManager에 임베딩 없이 등록 (나중에 라이브에서 추가 가능)
                # 이미 존재하는지 확인
                if speaker not in speaker_manager.speakers:
                    from core.speaker import Speaker
                    # 더미 임베딩 생성 (512차원, 모두 0)
                    # VoiceStore에 저장하려면 임베딩이 필요함
                    dummy_embedding = np.zeros(512, dtype=np.float32)

                    # VoiceStore에 저장
                    if speaker_manager.voice_store.ok:
                        speaker_manager.voice_store.upsert_speaker_embedding(
                            speaker_id=speaker,
                            display_name=speaker,
                            embedding=dummy_embedding
                        )
                        print(f"[INFO] Saved speaker '{speaker}' to VoiceStore (with placeholder embedding)")

                    # 메모리 캐시에도 추가
                    new_speaker = Speaker(
                        speaker_id=speaker,
                        display_name=speaker,
                        embedding=dummy_embedding,
                        embedding_count=0  # 실제 임베딩이 아니므로 0
                    )
                    speaker_manager.speakers[speaker] = new_speaker
                    print(f"[INFO] Created placeholder speaker entry: {speaker}")
            else:
                speaker = pyannote_label

        # speaker_id를 표시 이름으로 변환 (speaker_manager가 있는 경우)
        display_name = speaker
        if speaker_manager and speaker != "Unknown":
            display_name = speaker_manager.get_speaker_display_name(speaker)

        out.append({
            "speaker_id": speaker,  # speaker_id 저장 (persona lookup 시 사용)
            "speaker": display_name,  # display_name (UI 표시용)
            "text": w_text,
            "start": w_start,
            "end": w_end,
        })

    return out


# ---------- Audio Processing Steps ----------

def _convert_m4a_to_wav(path: str) -> tuple[str, Optional[str]]:
    """
    m4a 파일을 임시 wav로 변환

    Args:
        path: 입력 파일 경로

    Returns:
        (처리할 파일 경로, 임시 파일 경로 또는 None)
    """
    file_ext = os.path.splitext(path)[1].lower()

    if file_ext != ".m4a":
        return path, None

    print(f"[INFO] Detected .m4a file, converting to temporary WAV for better compatibility...")
    temp_wav_path = path.replace(".m4a", "_temp.wav")

    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-i", path,
                "-ar", "16000",  # 16kHz sampling rate (Whisper 최적)
                "-ac", "1",       # mono
                "-y",             # overwrite
                temp_wav_path
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"[INFO] Converted to: {temp_wav_path}")
            return temp_wav_path, temp_wav_path
        else:
            print(f"[WARNING] FFmpeg conversion failed: {result.stderr}")
            print(f"[INFO] Attempting to process original .m4a file directly...")
    except FileNotFoundError:
        print("[WARNING] FFmpeg not found in PATH. Attempting to process .m4a directly...")
    except Exception as e:
        print(f"[WARNING] Conversion error: {e}. Processing original file...")

    return path, None


def _transcribe_with_whisper(path: str, asr_model: str, use_gpu: bool) -> List[Dict]:
    """
    Whisper 모델을 사용하여 음성을 텍스트로 변환

    Args:
        path: 오디오 파일 경로
        asr_model: Whisper 모델명
        use_gpu: GPU 사용 여부

    Returns:
        세그먼트 리스트

    Raises:
        RuntimeError: WhisperModel이 설치되지 않았거나 로드 실패
    """
    if WhisperModel is None:
        raise RuntimeError("faster-whisper가 설치되지 않았습니다. (pip install faster-whisper)")

    device = _whisper_device(use_gpu)

    # compute_type 설정
    # - CPU: int8 (양자화로 빠른 속도)
    # - MPS: float32 (Metal에서 안정적)
    # - CUDA: float16 (GPU에서 빠른 속도)
    if device == "cpu":
        compute = "int8"
    elif device == "mps":
        compute = "float32"  # MPS는 float32가 안정적
    else:  # cuda
        compute = "float16"

    print(f"[INFO] Processing: {path}")
    print(f"[INFO] Device: {device}, Compute: {compute}")
    print(f"[INFO] Model: {asr_model}")

    # Whisper 모델 로드
    try:
        model = WhisperModel(asr_model, device=device, compute_type=compute)
    except Exception as e:
        print(f"[WARNING] Failed to load model with {device}/{compute}")
        print(f"[WARNING] Error: {e}")
        print(f"[INFO] Falling back to CPU with int8")
        try:
            model = WhisperModel(asr_model, device="cpu", compute_type="int8")
        except Exception as fallback_error:
            print(f"[ERROR] CPU fallback also failed: {fallback_error}")
            raise RuntimeError(f"Failed to load Whisper model: {fallback_error}")

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
    return whisper_segments


def _perform_diarization(path: str):
    """
    화자 분리(Diarization) 수행

    Args:
        path: 오디오 파일 경로

    Returns:
        (diarization annotation, pipeline) 튜플
    """
    if PyannotePipeline is None:
        return None, None

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        print("[WARNING] HF_TOKEN not set, skipping diarization")
        return None, None

    try:
        print("[INFO] Running speaker diarization...")
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        diar_annotation = pipeline(path)
        print("[INFO] Diarization completed")
        return diar_annotation, pipeline
    except Exception as e:
        print(f"[WARNING] Diarization failed: {e}")
        return None, None


def _merge_with_speakers(
    whisper_segments: List[Dict],
    diar_annotation,
    pipeline,
    speaker_manager,
    audio_path: str
) -> List[Dict]:
    """
    Whisper 세그먼트와 화자 정보를 병합

    Args:
        whisper_segments: Whisper STT 결과
        diar_annotation: Diarization 결과
        pipeline: Diarization 파이프라인
        speaker_manager: 화자 관리자
        audio_path: 오디오 파일 경로

    Returns:
        화자 정보가 포함된 세그먼트 리스트
    """
    if diar_annotation is not None:
        print("[INFO] Matching speakers...")
        return _match_speakers_by_overlap(
            whisper_segments,
            diar_annotation,
            pipeline=pipeline,
            speaker_manager=speaker_manager,
            audio_path=audio_path
        )
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
        return merged


def _save_meeting_json(
    title: str,
    original_path: str,
    merged_segments: List[Dict],
    md_bundle: Dict,
    asr_model: str,
    device: str,
    has_diarization: bool,
    use_llm_summary: bool,
    llm_backend: Optional[str]
) -> str:
    """
    회의 결과를 JSON 파일로 저장

    Args:
        title: 회의 제목
        original_path: 원본 파일 경로
        merged_segments: 병합된 세그먼트
        md_bundle: 마크다운 번들
        asr_model: 사용한 ASR 모델
        device: 사용한 디바이스
        has_diarization: 화자 분리 사용 여부
        use_llm_summary: LLM 요약 사용 여부
        llm_backend: LLM 백엔드

    Returns:
        저장된 JSON 파일 경로
    """
    json_obj = {
        "title": title,
        "source_path": os.path.abspath(original_path),
        "created_at": datetime.datetime.now().isoformat(),
        "segments": merged_segments,
        "summary": md_bundle.get("summary", ""),
        "actions": md_bundle.get("actions_text", ""),
        "asr_model": asr_model,
        "device": device,
        "diarization": has_diarization,
        "use_llm_summary": use_llm_summary,
        "llm_backend": llm_backend if use_llm_summary else None,
    }

    json_path = os.path.join("output", "meetings", f"{title}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved to: {json_path}")
    return json_path


def _cleanup_temp_file(temp_path: Optional[str]):
    """임시 파일 정리"""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"[INFO] Cleaned up temporary file: {temp_path}")
        except Exception as e:
            print(f"[WARNING] Failed to remove temporary file: {e}")


def _populate_digital_personas(
    persona_manager,
    speaker_manager,
    segments: List[Dict],
    llm_backend: Optional[str] = None
):
    """
    회의 세그먼트에서 디지털 페르소나를 자동 생성/업데이트

    Phase 1 구현:
    - 각 화자에 대해 페르소나 존재 여부 확인
    - 없으면 음성 임베딩으로 새 페르소나 생성
    - 발언을 RAG 컬렉션에 추가

    Args:
        persona_manager: DigitalPersonaManager 인스턴스
        speaker_manager: SpeakerManager 인스턴스
        segments: 화자가 식별된 세그먼트 리스트
        llm_backend: LLM 백엔드 (선택)
    """
    if not persona_manager or not speaker_manager or not segments:
        return

    # 화자별 발언 그룹화
    speaker_utterances = {}
    for seg in segments:
        # speaker_id가 있으면 사용, 없으면 speaker(display_name) 시도
        speaker_id = seg.get("speaker_id") or seg.get("speaker", "Unknown")
        if speaker_id == "Unknown":
            continue

        if speaker_id not in speaker_utterances:
            speaker_utterances[speaker_id] = []

        speaker_utterances[speaker_id].append({
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0)
        })

    # 각 화자에 대해 페르소나 처리
    for speaker_id, utterances in speaker_utterances.items():
        try:
            # 1. 페르소나 존재 여부 확인
            persona = persona_manager.get_persona(speaker_id)

            if not persona:
                # 2. 페르소나가 없으면 생성
                speaker = speaker_manager.get_speaker_by_id(speaker_id)
                if speaker and speaker.embedding is not None:
                    # 음성 임베딩이 있으면 페르소나 생성
                    display_name = speaker_manager.get_speaker_display_name(speaker_id)
                    persona_manager.create_persona(
                        speaker_id=speaker_id,
                        display_name=display_name,
                        voice_embedding=speaker.embedding,
                        llm_backend=llm_backend or "openai:gpt-4o-mini"
                    )
                    print(f"[INFO] Created digital persona for {speaker_id}")

            # 3. 발언을 RAG에 추가
            for utt in utterances:
                if utt["text"].strip():
                    persona_manager.add_utterance(
                        speaker_id=speaker_id,
                        text=utt["text"],
                        start=utt["start"],
                        end=utt["end"]
                    )

            print(f"[INFO] Added {len(utterances)} utterances for {speaker_id}")

        except Exception as e:
            print(f"[WARN] Failed to populate persona for {speaker_id}: {e}")


# ---------- Public API ----------

def process_audio_file(
    path: str,
    asr_model: str,
    use_gpu: bool = True,
    diarize: bool = True,
    use_llm_summary: bool = False,
    llm_backend: Optional[str] = None,
    settings: Optional[Dict] = None,
    speaker_manager=None,
    persona_manager=None,
) -> Dict:
    """
    오디오 파일 경로를 받아 STT(+옵션: 화자분리) 수행 후
    - segments(list[dict]), markdown(str), title(str), json_path(str) 를 반환
    - 디지털 페르소나 자동 생성/업데이트 (persona_manager 제공 시)

    Args:
        path: 입력 파일 경로 (오디오/비디오)
        asr_model: Whisper 모델명 (tiny/base/small/medium/large)
        use_gpu: GPU 사용 여부 (macOS는 무시됨)
        diarize: 화자 분리 사용 여부
        use_llm_summary: LLM 기반 요약 사용 여부
        llm_backend: LLM 백엔드 (예: "openai:gpt-4o-mini")
        settings: 추가 설정 (선택)
        speaker_manager: 화자 관리자 (선택)
        persona_manager: 디지털 페르소나 관리자 (선택)

    Returns:
        Dict with keys: title, markdown, json_path, segments

    Raises:
        RuntimeError: faster-whisper 미설치
        FileNotFoundError: 입력 파일 없음
    """
    _ensure_dirs()

    # 입력 검증
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper가 설치되어 있지 않습니다. (pip install faster-whisper)"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {path}")

    original_path = path

    try:
        # 1) m4a → wav 변환 (필요시)
        path, temp_wav_path = _convert_m4a_to_wav(path)

        # 2) Whisper STT
        whisper_segments = _transcribe_with_whisper(path, asr_model, use_gpu)

        # 3) Diarization (선택)
        diar_annotation, pipeline = None, None
        if diarize:
            diar_annotation, pipeline = _perform_diarization(path)

        # 4) 화자 매칭
        merged = _merge_with_speakers(
            whisper_segments,
            diar_annotation,
            pipeline,
            speaker_manager,
            path
        )

        # 5) 제목 정리
        title = os.path.splitext(os.path.basename(original_path))[0] or "회의록"
        if title.endswith("_temp"):
            title = title[:-5]

        # 6) 마크다운 생성
        md_bundle = _md_from_segments(
            title, merged,
            use_llm_summary=use_llm_summary,
            llm_backend=llm_backend
        )

        # 7) JSON 저장
        device = _whisper_device(use_gpu)
        json_path = _save_meeting_json(
            title=title,
            original_path=original_path,
            merged_segments=merged,
            md_bundle=md_bundle,
            asr_model=asr_model,
            device=device,
            has_diarization=bool(diar_annotation is not None),
            use_llm_summary=use_llm_summary,
            llm_backend=llm_backend
        )

        # 8) 디지털 페르소나 자동 생성/업데이트 (Phase 1)
        if persona_manager and speaker_manager and merged:
            _populate_digital_personas(
                persona_manager=persona_manager,
                speaker_manager=speaker_manager,
                segments=merged,
                llm_backend=llm_backend
            )

            # 파일 전사 완료 = 1회 회의 완료 처리
            speaker_ids = list(set(seg.get('speaker_id') or seg.get('speaker') for seg in merged if seg.get('speaker_id') or seg.get('speaker')))
            if speaker_ids:
                persona_manager.on_meeting_ended(speaker_ids)
                print(f"[INFO] File transcription completed: meeting count updated for {len(speaker_ids)} speakers")

        # 9) 임시 파일 정리
        _cleanup_temp_file(temp_wav_path)

        # UI에서 사용하기 편하도록 반환
        return {
            "title": title,
            "markdown": md_bundle.get("markdown", ""),
            "summary": md_bundle.get("summary", ""),
            "json_path": json_path,
            "segments": merged,
        }

    except Exception as e:
        # 에러 발생 시 임시 파일 정리
        if 'temp_wav_path' in locals():
            _cleanup_temp_file(temp_wav_path)
        raise
