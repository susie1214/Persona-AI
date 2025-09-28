# file_processor.py
import os
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio

from config import Config
from diarization import RealTimeDiarization
from summary import SummaryService

class FileProcessor:
    """오디오 파일 처리 클래스"""
    
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    
    def __init__(self, config: Config):
        self.config = config
        self.diar_pipeline = None
        self.whisper_model = None
        self.models_loaded = False
        
        # 콜백 함수들
        self.on_progress: Optional[Callable[[int], None]] = None  # 진행률 (0-100)
        self.on_transcription: Optional[Callable[[str, str, str], None]] = None
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_processing_complete: Optional[Callable[[], None]] = None
        
        # 처리 결과 저장
        self.transcription_results: List[Tuple[str, str, str]] = []  # (timestamp, speaker, text)
        self.speaker_texts: Dict[str, List[str]] = {}
        
        self.summary_service = SummaryService(self.config);
    
    def _emit_progress(self, progress: int):
        """진행률 발생"""
        if self.on_progress:
            self.on_progress(progress)
    
    def _emit_status(self, message: str):
        """상태 메시지 발생"""
        if self.on_status_change:
            self.on_status_change(message)
        else:
            print("[STATUS]", message)
    
    def _emit_error(self, message: str):
        """오류 메시지 발생"""
        if self.on_error:
            self.on_error(message)
        else:
            print("[ERROR]", message)
    
    def _emit_transcription(self, timestamp: str, speaker: str, text: str):
        """전사 결과 발생"""
        # 결과 저장
        self.transcription_results.append((timestamp, speaker, text))
        
        if speaker not in self.speaker_texts:
            self.speaker_texts[speaker] = []
        self.speaker_texts[speaker].append(text)
        
        # 콜백 호출
        if self.on_transcription:
            self.on_transcription(timestamp, speaker, text)
        else:
            print(f"[{timestamp}] {speaker}: {text}")
    
    def is_supported_file(self, file_path: str) -> bool:
        """지원되는 파일 형식인지 확인"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_AUDIO_FORMATS
    
    def get_supported_formats(self) -> List[str]:
        """지원되는 파일 형식 반환"""
        return list(self.SUPPORTED_AUDIO_FORMATS)
    
    def load_models_from_diarization(self, rt_diarization: RealTimeDiarization) -> bool:
        """RealTimeDiarization에서 로드된 모델 가져오기"""
        if not rt_diarization.models_loaded:
            return False
        
        self.diar_pipeline = rt_diarization.diar_pipeline
        self.whisper_model = rt_diarization.whisper_model
        self.models_loaded = True
        return True
    
    # def preprocess_audio_file(self, file_path: str) -> str:
    #     """오디오 파일 전처리 (리샘플링, 모노로 변환)"""
    #     self._emit_status("오디오 파일 전처리 중...")
        
    #     try:
    #         # librosa로 오디오 로드 (자동으로 모노로 변환하고 리샘플링)

    #         audio_data, sr = librosa.load(
    #             file_path, 
    #             sr = self.config.SAMPLE_RATE, 
    #             mono = True
    #         )
                
    #         print(f"[DEBUG] file_processor - audio file preprocess file path: {file_path}")
            
    #         # 임시 WAV 파일로 저장
    #         temp_wav = tempfile.NamedTemporaryFile(suffix = ".wav", delete = False)
    #         temp_wav_path = temp_wav.name
    #         print(f"[DEBUG] file_processor - temp audio file preprocess file path: {temp_wav_path}")
            
    #         # WAV 형식으로 저장
    #         sf.write(temp_wav_path, audio_data, self.config.SAMPLE_RATE)
            
    #         temp_wav.close()
            
    #         return temp_wav_path
    #     except Exception as e:
    #         self._emit_error(f"오디오 전처리 실패: {e}")
    #         raise
    
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """torchaudio로 오디오 로드, 모노 변환, 리샘플링"""
        
        print(f"[DEBUG] file_processor - audio file preprocess file path: {file_path}")
        waveform, sr = torchaudio.load(file_path)  # shape: [channels, samples]
        
        # 모노 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링
        if sr != self.config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # shape [samples]
        return waveform.squeeze(0)
    
    def preprocess_audio_file(self, file_path: str) -> str:
        self._emit_status("오디오 파일 전처리 중...")
        try:
            waveform = self._load_audio(file_path)
            
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            torchaudio.save(temp_wav_path, waveform.unsqueeze(0), self.config.SAMPLE_RATE)
            return temp_wav_path
        except Exception as e:
            self._emit_error(f"오디오 전처리 실패: {e}")
            raise
    
    def process_file(self, file_path: str) -> bool:
        """파일 처리 메인 함수"""
        if not self.models_loaded:
            self._emit_error("모델이 로드되지 않았습니다.")
            return False
        
        if not self.is_supported_file(file_path):
            self._emit_error("지원되지 않는 파일 형식입니다.")
            return False
        
        # 결과 초기화
        self.transcription_results.clear()
        self.speaker_texts.clear()
        
        try:
            self._emit_progress(10)
            
            # 오디오 파일 전처리
            self._emit_status("오디오 파일 전처리 중...")
            processed_audio_path = self.preprocess_audio_file(file_path)
            print(f'[DEBUG] preprocessed file path : {processed_audio_path}')
            
            self._emit_progress(30)
            
            # 화자분리 수행
            self._emit_status("화자분리 수행 중...")
            diar_result = self.diar_pipeline(processed_audio_path)
            
            self._emit_progress(50)
            
            # 전사 수행
            self._emit_status("음성 전사 수행 중...")
            self._process_diarization_result(processed_audio_path, diar_result)
            
            self._emit_progress(100)
            self._emit_status("파일 처리 완료")
            
            # 임시 파일 정리
            if processed_audio_path != file_path:
                os.unlink(processed_audio_path)
            
            if self.on_processing_complete:
                self.on_processing_complete()
            
            return True
            
        except Exception as e:
            self._emit_error(f"파일 처리 중 오류: {e}")
            # 임시 파일 정리
            if 'processed_audio_path' in locals() and processed_audio_path != file_path:
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass
            return False
    
    # def _process_diarization_result(self, audio_path: str, diar_result):
    #     """화자분리 결과를 전사로 변환"""
    #     # 오디오 데이터 로드
    #     audio_data, _ = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, mono=True)
        
    #     # 각 화자 세그먼트 처리
    #     segments = list(diar_result.itertracks(yield_label=True))
    #     total_segments = len(segments)
        
    #     for idx, (turn, _, speaker) in enumerate(segments):
    #         # 최소 길이 체크
    #         if (turn.end - turn.start) < self.config.MIN_SEG_DUR:
    #             continue
            
    #         # 진행률 업데이트
    #         progress = 50 + int((idx / total_segments) * 40)
    #         self._emit_progress(progress)
            
    #         # 오디오 세그먼트 추출
    #         start_sample = max(0, int(turn.start * self.config.SAMPLE_RATE))
    #         end_sample = min(len(audio_data), int(turn.end * self.config.SAMPLE_RATE))
            
    #         segment_audio = audio_data[start_sample:end_sample]
    #         if segment_audio.size == 0:
    #             continue
            
    #         # Whisper로 전사
    #         try:
    #             segments_result, _ = self.whisper_model.transcribe(
    #                 segment_audio,
    #                 language=self.config.WHISPER_LANG,
    #                 vad_filter=True,
    #                 beam_size=1,
    #                 word_timestamps=False
    #             )
    #             text = " ".join([s.text.strip() for s in segments_result if s.text])
    #         except Exception as e:
    #             self._emit_error(f"전사 중 오류: {e}")
    #             text = ""
            
    #         # 결과 발생
    #         if text.strip():
    #             timestamp = str(timedelta(seconds=round(turn.start)))
    #             self._emit_transcription(timestamp, speaker, text)
    
    def _process_diarization_result(self, audio_path: str, diar_result):
        waveform = self._load_audio(audio_path)
        segments = list(diar_result.itertracks(yield_label=True))
        total_segments = len(segments)
        
        for idx, (turn, _, speaker) in enumerate(segments):
            if (turn.end - turn.start) < self.config.MIN_SEG_DUR:
                continue
            progress = 50 + int((idx / total_segments) * 40)
            self._emit_progress(progress)
            
            start_sample = max(0, int(turn.start * self.config.SAMPLE_RATE))
            end_sample = min(waveform.shape[0], int(turn.end * self.config.SAMPLE_RATE))
            segment_waveform = waveform[start_sample:end_sample]
            if segment_waveform.numel() == 0:
                continue
            
            try:
                segments_result, _ = self.whisper_model.transcribe(
                    segment_waveform.cpu().numpy(),  # Whisper 입력으로 numpy 사용 가능
                    language=self.config.WHISPER_LANG,
                    vad_filter=True,
                    beam_size=1,
                    word_timestamps=False
                )
                text = " ".join([s.text.strip() for s in segments_result if s.text])
            except Exception as e:
                self._emit_error(f"전사 중 오류: {e}")
                text = ""
            
            if text.strip():
                timestamp = str(timedelta(seconds=round(turn.start)))
                self._emit_transcription(timestamp, speaker, text)
    
    def get_processing_results(self) -> Dict:
        """처리 결과 반환"""
        return {
            'transcriptions': self.transcription_results.copy(),
            'speaker_texts': self.speaker_texts.copy(),
            'total_speakers': len(self.speaker_texts),
            'total_segments': len(self.transcription_results)
        }
    
    def export_results_to_file(self, output_path: str, format: str = 'txt') -> bool:
        """결과를 파일로 내보내기"""
        try:
            if format.lower() == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("=== 화자분리 및 전사 결과 ===\n\n")
                    for timestamp, speaker, text in self.transcription_results:
                        f.write(f"[{timestamp}] {speaker}: {text}\n")
                                        
                    f.write("\n\n=== 화자별 요약 ===\n\n")
                    for speaker, texts in self.speaker_texts.items():
                        f.write(f"{speaker}:\n")
                        summary = self.summary_service.summarize_speaker_text(speaker, texts)
                        f.write(f"  - {summary}")
                        f.write("\n")
                        
            
            elif format.lower() == 'json':
                import json
                results = self.get_processing_results()
                results['export_time'] = datetime.now().isoformat()
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            self._emit_error(f"파일 내보내기 실패: {e}")
            return False