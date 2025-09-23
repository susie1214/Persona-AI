# audio_processor.py
# 오디오 처리 및 음성 인식 모듈

import os
import io
import time
import tempfile
import threading
import queue
from collections import deque
from typing import Optional, List, Dict, Callable
import numpy as np
import soundfile as sf
import pickle
import wave
from sklearn.metrics.pairwise import cosine_similarity
from PyQt6.QtCore import QObject, pyqtSignal

from config import config
from models import Segment, SpeakerProfile, ConversationEntry

# Runtime imports with error handling
try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SpeakerManager:
    """화자 관리 및 식별 클래스"""
    
    def __init__(self):
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.session_speaker_mapping: Dict[str, str] = {}
        self.load_speaker_profiles()
    
    def load_speaker_profiles(self):
        """저장된 화자 프로필 로드"""
        try:
            if config.storage.SPEAKER_PROFILES_PATH.exists():
                with open(config.storage.SPEAKER_PROFILES_PATH, 'rb') as f:
                    self.speaker_profiles = pickle.load(f)
                
                # 오래된 화자 정보 정리
                current_time = time.time()
                expired_speakers = []
                
                for speaker_id, profile in self.speaker_profiles.items():
                    if current_time - profile.last_seen > config.speaker.SPEAKER_TIMEOUT:
                        expired_speakers.append(speaker_id)
                
                for speaker_id in expired_speakers:
                    del self.speaker_profiles[speaker_id]
                
                print(f"화자 프로필 {len(self.speaker_profiles)}개 로드 완료")
                if expired_speakers:
                    print(f"만료된 화자 {len(expired_speakers)}개 정리")
                    
        except Exception as e:
            print(f"화자 프로필 로드 실패: {e}")
            self.speaker_profiles = {}
    
    def save_speaker_profiles(self):
        """화자 프로필 저장"""
        try:
            with open(config.storage.SPEAKER_PROFILES_PATH, 'wb') as f:
                pickle.dump(self.speaker_profiles, f)
            print(f"화자 프로필 {len(self.speaker_profiles)}개 저장 완료")
        except Exception as e:
            print(f"화자 프로필 저장 실패: {e}")
    
    def get_speaker_embedding(self, audio_segment: np.ndarray, diar_pipeline) -> Optional[List[float]]:
        """오디오 세그먼트에서 화자 임베딩 추출"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_segment, config.audio.SAMPLE_RATE)
                
                # pyannote 임베딩 추출
                embedding_result = diar_pipeline._embedding(tmp_file.name)
                if hasattr(embedding_result, 'data') and len(embedding_result.data) > 0:
                    speaker_embedding = np.mean(embedding_result.data, axis=0)
                    os.unlink(tmp_file.name)
                    return speaker_embedding.tolist()
                
                os.unlink(tmp_file.name)
                return None
                
        except Exception as e:
            print(f"화자 임베딩 추출 실패: {e}")
            return None
    
    def identify_speaker(self, pyannote_speaker: str, audio_segment: np.ndarray, 
                        segment_duration: float, diar_pipeline) -> str:
        """pyannote 화자 ID를 실제 화자 ID로 매핑"""
        current_time = time.time()
        
        # 이미 매핑된 화자인 경우
        if pyannote_speaker in self.session_speaker_mapping:
            mapped_speaker = self.session_speaker_mapping[pyannote_speaker]
            
            # 화자 프로필 업데이트
            if mapped_speaker in self.speaker_profiles:
                profile = self.speaker_profiles[mapped_speaker]
                profile.last_seen = current_time
                profile.total_duration += segment_duration
                profile.sample_count += 1
                
                # 새로운 임베딩 추가 (가끔씩)
                if profile.sample_count % 5 == 0:
                    speaker_embedding = self.get_speaker_embedding(audio_segment, diar_pipeline)
                    if speaker_embedding:
                        profile.embeddings.append(speaker_embedding)
                        if len(profile.embeddings) > config.speaker.MAX_SPEAKER_EMBEDDINGS:
                            profile.embeddings.pop(0)
            
            return mapped_speaker
        
        # 새로운 pyannote 화자 - 기존 화자와 매칭 시도
        speaker_embedding = self.get_speaker_embedding(audio_segment, diar_pipeline)
        if not speaker_embedding:
            # 임베딩 추출 실패 시 임시 ID 할당
            temp_speaker_id = f"Speaker_{len(self.session_speaker_mapping) + 1}"
            self.session_speaker_mapping[pyannote_speaker] = temp_speaker_id
            return temp_speaker_id
        
        best_match_speaker = None
        best_similarity = 0.0
        
        # 기존 화자들과 유사도 비교
        for speaker_id, profile in self.speaker_profiles.items():
            if not profile.embeddings:
                continue
            
            similarities = []
            for stored_embedding in profile.embeddings:
                similarity = cosine_similarity([speaker_embedding], [stored_embedding])[0][0]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_speaker = speaker_id
        
        # 임계값 이상의 유사도를 가진 화자가 있는 경우
        if best_match_speaker and best_similarity >= config.speaker.SIMILARITY_THRESHOLD:
            real_speaker_id = best_match_speaker
            print(f"화자 연결: {pyannote_speaker} -> {real_speaker_id} (유사도: {best_similarity:.3f})")
        else:
            # 새로운 화자 생성
            existing_persons = [s for s in self.speaker_profiles.keys() if s.startswith('Person_')]
            real_speaker_id = f"Person_{chr(65 + len(existing_persons))}"
            print(f"새 화자 등록: {pyannote_speaker} -> {real_speaker_id}")
        
        # 매핑 저장
        self.session_speaker_mapping[pyannote_speaker] = real_speaker_id
        
        # 화자 프로필 생성/업데이트
        if real_speaker_id not in self.speaker_profiles:
            self.speaker_profiles[real_speaker_id] = SpeakerProfile(
                speaker_id=real_speaker_id,
                embeddings=[speaker_embedding],
                last_seen=current_time,
                total_duration=segment_duration,
                sample_count=1,
                display_name=real_speaker_id
            )
        else:
            profile = self.speaker_profiles[real_speaker_id]
            profile.embeddings.append(speaker_embedding)
            if len(profile.embeddings) > config.speaker.MAX_SPEAKER_EMBEDDINGS:
                profile.embeddings.pop(0)
            profile.last_seen = current_time
            profile.total_duration += segment_duration
            profile.sample_count += 1
        
        return real_speaker_id

class AudioProcessor(QObject):
    """오디오 처리 및 실시간 STT 클래스"""
    
    # Signals
    segment_ready = pyqtSignal(object)  # Segment
    status_update = pyqtSignal(str)
    diarization_update = pyqtSignal(list)  # List[tuple]
    
    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.audio.BUFFER_DURATION * config.audio.SAMPLE_RATE))
        self.is_recording = False
        self.diar_segments: List[tuple] = []
        
        # PyAudio
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Models
        self.whisper_model: Optional[WhisperModel] = None
        self.diar_pipeline = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Speaker management
        self.speaker_manager = SpeakerManager()
        
        # Processing
        self.processing_thread = None
        self.diarization_thread = None
        self._stop_event = threading.Event()
        self._file_lock = threading.Lock()  # 파일 접근 동기화를 위한 락
        
        # Audio file management
        self.raw_audio_path: Optional[str] = None
        self.wave_file: Optional[wave.Wave_write] = None
        self._frames_elapsed = 0

        # 시그널 연결
        self.diarization_update.connect(self.on_diarization_update)

    @pyqtSlot(list)
    def on_diarization_update(self, segments: list):
        """화자분리 결과 업데이트"""
        self.diar_segments = segments
        
    def initialize_models(self):
        """모델 초기화"""
        if WhisperModel is None:
            raise RuntimeError("faster-whisper 미설치")
        
        self.status_update.emit("모델 로딩 중...")
        
        # Whisper 모델 로드
        try:
            self.whisper_model = WhisperModel(
                config.model.WHISPER_MODEL,
                device=config.model.WHISPER_DEVICE,
                compute_type=config.model.WHISPER_COMPUTE_TYPE
            )
            self.status_update.emit(f"Whisper 모델 로드 완료: {config.model.WHISPER_MODEL}")
        except Exception as e:
            self.status_update.emit(f"Whisper GPU 실패 -> CPU 재시도: {e}")
            self.whisper_model = WhisperModel(
                config.model.WHISPER_MODEL,
                device="cpu",
                compute_type="int8"
            )
        
        # Diarization 파이프라인 로드
        if PyannotePipeline and os.getenv(config.model.HF_TOKEN_ENV):
            try:
                self.diar_pipeline = PyannotePipeline.from_pretrained(
                    config.model.PYANNOTE_PIPELINE_NAME,
                    use_auth_token=os.getenv(config.model.HF_TOKEN_ENV)
                )
                self.status_update.emit("화자분리 모델 로드 완료")
            except Exception as e:
                self.status_update.emit(f"화자분리 모델 로드 실패: {e}")
        
        # Embedding 모델 로드
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(config.model.EMBEDDING_MODEL)
                self.status_update.emit("임베딩 모델 로드 완료")
            except Exception as e:
                self.status_update.emit(f"임베딩 모델 로드 실패: {e}")
    
    def start_recording(self):
        """녹음 시작"""
        if pyaudio is None:
            raise RuntimeError("PyAudio 미설치")
        
        self.initialize_models()
        self._stop_event.clear()
        
        # 오디오 파일 준비
        fd, self.raw_audio_path = tempfile.mkstemp(suffix=".wav", prefix="meeting_")
        os.close(fd)
        
        self.wave_file = wave.open(self.raw_audio_path, 'wb')
        self.wave_file.setnchannels(config.audio.CHANNELS)
        self.wave_file.setsampwidth(config.audio.SAMPLE_WIDTH)
        self.wave_file.setframerate(config.audio.SAMPLE_RATE)
        
        # PyAudio 초기화
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(config.audio.SAMPLE_WIDTH),
            channels=config.audio.CHANNELS,
            rate=config.audio.SAMPLE_RATE,
            input=True,
            frames_per_buffer=int(config.audio.SAMPLE_RATE * 0.2),
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        self.audio_stream.start_stream()
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # 화자분리 스레드 시작
        if self.diar_pipeline:
            self.diarization_thread = threading.Thread(target=self._diarization_loop, daemon=True)
            self.diarization_thread.start()
        
        self.status_update.emit("녹음 시작됨")
    
    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        self._stop_event.set()
        
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            # 웨이브 파일 닫기 (락 사용)
            with self._file_lock:
                if self.wave_file:
                    self.wave_file.close()
                    self.wave_file = None
                
        except Exception as e:
            print(f"오디오 정리 중 오류: {e}")
        
        # 화자 프로필 저장
        self.speaker_manager.save_speaker_profiles()
        
        self.status_update.emit("녹음 중지됨")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 콜백"""
        if status:
            print(f"오디오 콜백 상태: {status}")
        
        # 원본 데이터 파일에 쓰기 (락 사용)
        with self._file_lock:
            if self.wave_file:
                self.wave_file.writeframes(in_data)
            
        # numpy array로 변환 (분석용)
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 버퍼에 추가
        self.audio_buffer.extend(audio_data)
        self.audio_queue.put(audio_data.copy())
        self._frames_elapsed += frame_count
        
        return (None, pyaudio.paContinue)
    
    def _processing_loop(self):
        """오디오 처리 메인 루프"""
        process_buffer = []
        last_process_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # 오디오 데이터 수집
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    process_buffer.extend(audio_chunk)
                except queue.Empty:
                    continue
                
                current_time = time.time()
                buffer_duration = len(process_buffer) / config.audio.SAMPLE_RATE
                
                # 충분한 데이터가 모이면 처리
                if (buffer_duration >= config.audio.PROCESS_INTERVAL or
                    current_time - last_process_time >= config.audio.PROCESS_INTERVAL):
                    
                    if len(process_buffer) > 0:
                        audio_array = np.array(process_buffer, dtype=np.float32)
                        base_time = current_time - buffer_duration
                        
                        # STT 처리
                        self._process_stt(audio_array, base_time)
                        
                        # 겹침 유지
                        overlap_samples = int(config.audio.OVERLAP_DURATION * config.audio.SAMPLE_RATE)
                        if len(process_buffer) > overlap_samples:
                            process_buffer = process_buffer[-overlap_samples:]
                        else:
                            process_buffer = []
                        
                        last_process_time = current_time
                
            except Exception as e:
                print(f"처리 루프 오류: {e}")
                time.sleep(0.1)
    
    def _process_stt(self, audio_data: np.ndarray, base_time: float):
        """STT 처리"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                sf.write(tmp_file.name, audio_data, config.audio.SAMPLE_RATE)
                
                # Whisper STT
                segments, info = self.whisper_model.transcribe(
                    tmp_file.name,
                    language=config.model.WHISPER_LANG,
                    vad_filter=True
                )
                
                for whisper_seg in segments:
                    if not whisper_seg.text.strip():
                        continue
                    
                    segment = Segment(
                        start=base_time + whisper_seg.start,
                        end=base_time + whisper_seg.end,
                        text=whisper_seg.text.strip(),
                        speaker_id="Unknown",
                        speaker_name="Unknown"
                    )
                    
                    # 화자 식별 (화자분리가 활성화된 경우)
                    if self.diar_pipeline and self.diar_segments:
                        speaker = self._find_speaker_for_segment(segment)
                        if speaker:
                            segment.speaker_name = speaker
                            segment.speaker_id = speaker
                    
                    self.segment_ready.emit(segment)
                    
        except Exception as e:
            print(f"STT 처리 오류: {e}")

    def _find_speaker_for_segment(self, segment: Segment) -> Optional[str]:
        """세그먼트의 시간과 가장 많이 겹치는 화자를 찾습니다."""
        max_overlap = 0
        best_speaker = None
        
        for diar_start, diar_end, speaker in self.diar_segments:
            overlap_start = max(segment.start, diar_start)
            overlap_end = min(segment.end, diar_end)
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker
        
        # 최소 겹침 시간을 만족하는 경우에만 화자 할당 (옵션)
        segment_duration = segment.end - segment.start
        if best_speaker and max_overlap / segment_duration > 0.5: # 50% 이상 겹칠 때
            return best_speaker
            
        return None
    
    def _diarization_loop(self):
        """화자분리 처리 루프 (주기적)"""
        while not self._stop_event.is_set():
            time.sleep(30)  # 30초마다 실행
            
            if not self.raw_audio_path or not self.is_recording:
                continue

            diar_path = None
            try:
                with self._file_lock:
                    if self.wave_file:
                        # 파일을 닫아 읽기 가능 상태로 만듦
                        self.wave_file.close()
                        diar_path = self.raw_audio_path
                    else:
                        continue

                # 화자분리 실행 (락 외부에서)
                if diar_path and os.path.exists(diar_path):
                    self.status_update.emit("화자분리 실행 중...")
                    diar_result = self.diar_pipeline(diar_path)
                    segments = []
                    for turn, _, speaker in diar_result.itertracks(yield_label=True):
                        segments.append((turn.start, turn.end, speaker))
                    
                    if segments:
                        self.diarization_update.emit(segments)
                        self.status_update.emit(f"화자분리 업데이트: {len(segments)}개 구간")

            except Exception as e:
                print(f"화자분리 처리 오류: {e}")
            finally:
                # 파일을 다시 append 모드로 열기 (락 내부에서)
                with self._file_lock:
                    if self.is_recording and self.raw_audio_path and not self.wave_file:
                        try:
                            self.wave_file = wave.open(self.raw_audio_path, 'ab')
                        except Exception as e:
                            print(f"웨이브 파일 다시 열기 실패: {e}")

# 독립 실행 테스트
if __name__ == "__main__":
    import time
    import sys
    from datetime import datetime
    
    print("=" * 50)
    print("Audio Processor Module Test")
    print("=" * 50)
    
    # 의존성 체크
    print("📦 Dependency Check:")
    print(f"  - PyAudio: {'✅ Available' if pyaudio else '❌ Not available'}")
    print(f"  - Faster-Whisper: {'✅ Available' if WhisperModel else '❌ Not available'}")
    print(f"  - Pyannote: {'✅ Available' if PyannotePipeline else '❌ Not available'}")
    print(f"  - SentenceTransformers: {'✅ Available' if SentenceTransformer else '❌ Not available'}")
    
    # SpeakerManager 테스트
    print("\n👤 SpeakerManager Test:")
    try:
        speaker_manager = SpeakerManager()
        print(f"  - Speaker profiles loaded: {len(speaker_manager.speaker_profiles)}")
        print(f"  - Session mappings: {len(speaker_manager.session_speaker_mapping)}")
        
        # 더미 오디오 데이터로 임베딩 테스트 (실제로는 작동하지 않음)
        print("  - Embedding extraction test: Simulation only")
        print("    (Real test requires actual audio data and diarization pipeline)")
        
    except Exception as e:
        print(f"  ❌ SpeakerManager test failed: {e}")
    
    # AudioProcessor 초기화 테스트
    print("\n🎤 AudioProcessor Initialization Test:")
    try:
        from PyQt6.QtCore import QCoreApplication
        
        # Qt 애플리케이션 필요 (QObject 상속 때문에)
        app = QCoreApplication(sys.argv) if not QCoreApplication.instance() else QCoreApplication.instance()
        
        processor = AudioProcessor()
        print("  ✅ AudioProcessor created successfully")
        print(f"  - Recording status: {processor.is_recording}")
        print(f"  - Audio queue size: {processor.audio_queue.qsize()}")
        print(f"  - Buffer max length: {processor.audio_buffer.maxlen}")
        
        # 시그널 연결 테스트
        def test_signal_handler(message):
            print(f"  📡 Signal received: {message}")
        
        processor.status_update.connect(test_signal_handler)
        processor.status_update.emit("Test signal emission")
        
    except ImportError:
        print("  ⚠️ PyQt6 not available - creating mock processor")
        print("  (Full test requires PyQt6 installation)")
    except Exception as e:
        print(f"  ❌ AudioProcessor test failed: {e}")
    
    # 설정 테스트
    print(f"\n⚙️ Configuration Test:")
    print(f"  - Sample Rate: {config.audio.SAMPLE_RATE} Hz")
    print(f"  - Channels: {config.audio.CHANNELS}")
    print(f"  - Buffer Duration: {config.audio.BUFFER_DURATION}s")
    print(f"  - Process Interval: {config.audio.PROCESS_INTERVAL}s")
    print(f"  - Chunk Size: {config.audio.CHUNK_SIZE}")
    
    # 오디오 장치 목록 테스트 (PyAudio 사용 가능시)
    if pyaudio:
        print(f"\n🔊 Available Audio Devices:")
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"  Total devices: {device_count}")
            
            for i in range(min(5, device_count)):  # 처음 5개만 표시
                info = p.get_device_info_by_index(i)
                print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
            
            p.terminate()
            
        except Exception as e:
            print(f"  ❌ Audio device enumeration failed: {e}")
    
    # 모델 설정 테스트
    print(f"\n🤖 Model Configuration Test:")
    print(f"  - Whisper Model: {config.model.WHISPER_MODEL}")
    print(f"  - Device: {config.model.WHISPER_DEVICE}")
    print(f"  - Compute Type: {config.model.WHISPER_COMPUTE_TYPE}")
    print(f"  - Language: {config.model.WHISPER_LANG}")
    
    # 임시 파일 생성 테스트
    print(f"\n📁 File Operations Test:")
    try:
        import tempfile
        import soundfile as sf
        import numpy as np
        
        # 임시 WAV 파일 생성 테스트
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # 1초간의 더미 오디오 데이터 생성
            sample_rate = config.audio.SAMPLE_RATE
            duration = 1.0
            samples = int(sample_rate * duration)
            
            # 사인파 생성 (440Hz)
            t = np.linspace(0, duration, samples)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            sf.write(tmp_file.name, audio_data, sample_rate)
            print(f"  ✅ Temporary WAV file created: {tmp_file.name}")
            
            # 파일 읽기 테스트
            read_data, read_sr = sf.read(tmp_file.name)
            print(f"  ✅ File read successfully: {len(read_data)} samples at {read_sr}Hz")
            
            # 파일 삭제
            import os
            os.unlink(tmp_file.name)
            print(f"  ✅ Temporary file cleaned up")
            
    except Exception as e:
        print(f"  ❌ File operations test failed: {e}")
    
    # 실제 오디오 처리 데모 (PyAudio와 Whisper가 모두 사용 가능한 경우)
    if pyaudio and WhisperModel:
        print(f"\n🎙️ Audio Processing Demo:")
        response = input("  Start 5-second recording demo? (y/N): ").strip().lower()
        
        if response == 'y':
            try:
                print("  Preparing audio recording...")
                
                # 간단한 녹음 테스트
                p = pyaudio.PyAudio()
                
                stream = p.open(
                    format=p.get_format_from_width(config.audio.SAMPLE_WIDTH),
                    channels=config.audio.CHANNELS,
                    rate=config.audio.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024
                )
                
                print("  🔴 Recording for 5 seconds... Speak now!")
                frames = []
                
                for _ in range(0, int(config.audio.SAMPLE_RATE / 1024 * 5)):
                    data = stream.read(1024)
                    frames.append(data)
                
                print("  ⏹️ Recording finished")
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    with wave.open(tmp_file.name, 'wb') as wf:
                        wf.setnchannels(config.audio.CHANNELS)
                        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(config.audio.SAMPLE_WIDTH)))
                        wf.setframerate(config.audio.SAMPLE_RATE)
                        wf.writeframes(b''.join(frames))
                    
                    print(f"  💾 Audio saved to: {tmp_file.name}")
                    
                    # Whisper 테스트
                    print("  🧠 Loading Whisper model...")
                    model = WhisperModel(
                        config.model.WHISPER_MODEL,
                        device="cpu",  # 안전을 위해 CPU 사용
                        compute_type="int8"
                    )
                    
                    print("  🎯 Transcribing audio...")
                    segments, info = model.transcribe(tmp_file.name, language=config.model.WHISPER_LANG)
                    
                    print("  📝 Transcription results:")
                    for segment in segments:
                        print(f"    [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
                    
                    # 정리
                    import os
                    os.unlink(tmp_file.name)
                    print("  ✅ Demo completed and cleaned up")
                
            except Exception as e:
                print(f"  ❌ Recording demo failed: {e}")
        else:
            print("  Demo skipped")
    else:
        print(f"\n⚠️ Audio Processing Demo:")
        print("  Demo requires both PyAudio and Faster-Whisper")
        missing = []
        if not pyaudio:
            missing.append("PyAudio")
        if not WhisperModel:
            missing.append("Faster-Whisper")
        print(f"  Missing: {', '.join(missing)}")
    
    print(f"\n" + "=" * 50)
    print("Audio Processor Module Test Complete!")
    
    if '--interactive' in sys.argv:
        print("\nInteractive mode - press Enter to exit")
        input()