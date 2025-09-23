import os
import librosa
import numpy as np
from datetime import timedelta, datetime
import yaml
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel
from dataclasses import dataclass
from pathlib import Path
import torch
import pyaudio
import wave
import threading
import queue
import tempfile
from collections import deque
import time
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import atexit
import pickle
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class SpeakerProfile:
    """화자 프로필을 저장하는 데이터 클래스"""
    speaker_id: str  # 실제 화자 ID (Person_A, Person_B 등)
    embeddings: List[List[float]]  # 화자 임베딩 벡터들
    last_seen: float  # 마지막 등장 시간
    total_duration: float  # 총 발화 시간
    sample_count: int  # 샘플 수

@dataclass
class ConversationEntry:
    """대화 엔트리를 나타내는 데이터 클래스"""
    id: str
    timestamp: str
    speaker: str
    text: str
    start_time: float
    end_time: float
    embedding: List[float] = None

@dataclass
class Config:
    """Configuration for real-time diarization and transcription."""
    PYANNOTE_MODEL_PATH: Path = Path("D:/Persona-AI/models/diart_model")
    WHISPER_MODEL_PATH: Path = Path("D:/Persona-AI/models/whisper-small-ct2")
    
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    WHISPER_LANG: str = "ko"
    
    # Audio recording settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    CHANNELS: int = 1
    AUDIO_FORMAT: int = pyaudio.paInt16
    
    # Processing settings
    BUFFER_DURATION: float = 30.0  # 30초 버퍼
    PROCESS_INTERVAL: float = 10.0  # 10초마다 처리
    MIN_SEG_DUR: float = 0.35
    OVERLAP_DURATION: float = 5.0  # 겹치는 구간 (연속성 보장)
    
    # Storage settings
    OUTPUT_DIR: Path = Path("./output")
    QDRANT_PATH: Path = Path("./qdrant_storage")  # 로컬 파일 저장 경로
    SPEAKER_PROFILES_PATH: Path = Path("./speaker_profiles.pkl")  # 화자 프로필 저장
    COLLECTION_NAME: str = "conversation_embeddings"
    EMBEDDING_MODEL: str = "dragonkue/BGE-m3-ko"  # 한국어 특화 BGE 모델
    
    # Speaker continuity settings
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.75  # 화자 유사도 임계값
    MAX_SPEAKER_EMBEDDINGS: int = 10  # 화자당 최대 보관할 임베딩 수
    SPEAKER_TIMEOUT: float = 3600.0  # 화자 정보 유지 시간 (1시간)

class RealTimeDiarization:
    def __init__(self, config: Config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.BUFFER_DURATION * config.SAMPLE_RATE))
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # 화자 연속성 관리
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}  # 실제 화자 프로필
        self.session_speaker_mapping: Dict[str, str] = {}  # pyannote ID -> 실제 화자 ID 매핑
        self.load_speaker_profiles()  # 이전 세션의 화자 정보 로드
        
        # 대화 내용 저장용
        self.conversation_log: List[ConversationEntry] = []
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # 모델 로드
        print("[INFO] Loading models...")
        self.diar_pipeline = self.load_diarization_pipeline()
        self.whisper_model = self.load_whisper_model()
        self.embedding_model = self.load_embedding_model()
        
        # Qdrant 클라이언트 초기화
        self.qdrant_client = self.initialize_qdrant()
        
        # 출력 디렉토리 생성
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # 프로그램 종료 시 자동 저장
        atexit.register(self.save_conversation)
        
    def hhmmss(self, seconds: float) -> str:
        """Converts seconds to HH:MM:SS format."""
        return str(timedelta(seconds=round(seconds)))
    
    def load_diarization_pipeline(self) -> SpeakerDiarization:
        """Loads the pyannote speaker diarization pipeline."""
        print("[INFO] Loading local Pyannote model...")
        segmentation_model = Model.from_pretrained(
            self.config.PYANNOTE_MODEL_PATH / "segmentation-3.0" / "pytorch_model.bin"
        )
        embedding_model = Model.from_pretrained(
            self.config.PYANNOTE_MODEL_PATH / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin"
        )

        with open(self.config.PYANNOTE_MODEL_PATH / "config.yaml", "r") as f:
            pipeline_config = yaml.safe_load(f)

        diar_pipeline = SpeakerDiarization(
            segmentation=segmentation_model,
            embedding=embedding_model,
        )
        diar_pipeline.instantiate(pipeline_config['params'])
        return diar_pipeline

    def load_whisper_model(self) -> WhisperModel:
        """Loads the faster-whisper model."""
        print("[INFO] Loading Whisper model...")
        return WhisperModel(
            str(self.config.WHISPER_MODEL_PATH),
            device=self.config.WHISPER_DEVICE,
            compute_type=self.config.WHISPER_COMPUTE_TYPE
        )
    
    def load_embedding_model(self) -> SentenceTransformer:
        """임베딩 모델 로드"""
        print("[INFO] Loading embedding model...")
        return SentenceTransformer(self.config.EMBEDDING_MODEL)
    
    def initialize_qdrant(self) -> QdrantClient:
        """Qdrant 로컬 클라이언트 초기화 및 컬렉션 생성"""
        print("[INFO] Initializing Qdrant local storage...")
        try:
            # 로컬 저장 디렉토리 생성
            self.config.QDRANT_PATH.mkdir(exist_ok=True)
            
            # 로컬 파일 시스템을 사용하는 Qdrant 클라이언트 생성
            client = QdrantClient(
                path=str(self.config.QDRANT_PATH)
            )
            
            # 컬렉션 존재 여부 확인
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.config.COLLECTION_NAME not in collection_names:
                # 임베딩 차원 확인 (테스트 임베딩 생성)
                test_embedding = self.embedding_model.encode(["test"])
                embedding_dim = len(test_embedding[0])
                
                print(f"[INFO] Creating Qdrant collection with dimension {embedding_dim}")
                client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                print(f"[INFO] Using existing Qdrant collection: {self.config.COLLECTION_NAME}")
            
            print(f"[INFO] Qdrant local storage initialized at: {self.config.QDRANT_PATH}")
            return client
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize Qdrant local storage: {e}")
            print("[WARNING] Continuing without Qdrant. Data will only be saved as JSON.")
            return None
    
    def load_speaker_profiles(self):
        """이전 세션의 화자 프로필 로드"""
        try:
            if self.config.SPEAKER_PROFILES_PATH.exists():
                with open(self.config.SPEAKER_PROFILES_PATH, 'rb') as f:
                    self.speaker_profiles = pickle.load(f)
                
                # 오래된 화자 정보 정리
                current_time = time.time()
                expired_speakers = []
                
                for speaker_id, profile in self.speaker_profiles.items():
                    if current_time - profile.last_seen > self.config.SPEAKER_TIMEOUT:
                        expired_speakers.append(speaker_id)
                
                for speaker_id in expired_speakers:
                    del self.speaker_profiles[speaker_id]
                
                print(f"📋 이전 세션 화자 {len(self.speaker_profiles)}명 로드완료")
                if expired_speakers:
                    print(f"🧹 만료된 화자 {len(expired_speakers)}명 정리완료")
                
        except Exception as e:
            print(f"[WARNING] 화자 프로필 로드 실패: {e}")
            self.speaker_profiles = {}
    
    def save_speaker_profiles(self):
        """화자 프로필을 파일로 저장"""
        try:
            with open(self.config.SPEAKER_PROFILES_PATH, 'wb') as f:
                pickle.dump(self.speaker_profiles, f)
            print(f"💾 화자 프로필 저장완료: {len(self.speaker_profiles)}명")
        except Exception as e:
            print(f"❌ 화자 프로필 저장 실패: {e}")
    
    def get_speaker_embedding(self, audio_segment: np.ndarray) -> Optional[List[float]]:
        """오디오 세그먼트에서 화자 임베딩 추출"""
        try:
            # 임시 파일 생성하여 화자 임베딩 추출
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.config.CHANNELS)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.config.SAMPLE_RATE)
                    audio_int16 = (audio_segment * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # pyannote 임베딩 모델 사용 (이미 로드된 모델 재사용)
                embedding_result = self.diar_pipeline._embedding(tmp_file.name)
                if hasattr(embedding_result, 'data') and len(embedding_result.data) > 0:
                    # 평균 임베딩 계산
                    speaker_embedding = np.mean(embedding_result.data, axis=0)
                    os.unlink(tmp_file.name)
                    return speaker_embedding.tolist()
                
                os.unlink(tmp_file.name)
                return None
                
        except Exception as e:
            print(f"[WARNING] 화자 임베딩 추출 실패: {e}")
            return None
    
    def identify_speaker(self, pyannote_speaker: str, audio_segment: np.ndarray, segment_duration: float) -> str:
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
                
                # 새로운 임베딩 추가 (가끔씩)
                if profile.sample_count % 5 == 0:  # 5번마다 한 번씩 임베딩 업데이트
                    speaker_embedding = self.get_speaker_embedding(audio_segment)
                    if speaker_embedding:
                        profile.embeddings.append(speaker_embedding)
                        if len(profile.embeddings) > self.config.MAX_SPEAKER_EMBEDDINGS:
                            profile.embeddings.pop(0)  # 오래된 것 제거
                
                profile.sample_count += 1
            
            return mapped_speaker
        
        # 새로운 pyannote 화자 - 기존 화자와 매칭 시도
        speaker_embedding = self.get_speaker_embedding(audio_segment)
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
            
            # 평균 유사도 계산
            similarities = []
            for stored_embedding in profile.embeddings:
                similarity = cosine_similarity([speaker_embedding], [stored_embedding])[0][0]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_speaker = speaker_id
        
        # 임계값 이상의 유사도를 가진 화자가 있는 경우
        if best_match_speaker and best_similarity >= self.config.SPEAKER_SIMILARITY_THRESHOLD:
            real_speaker_id = best_match_speaker
            print(f"🔗 화자 연결: {pyannote_speaker} → {real_speaker_id} (유사도: {best_similarity:.3f})")
        else:
            # 새로운 화자 생성
            real_speaker_id = f"Person_{chr(65 + len([s for s in self.speaker_profiles.keys() if s.startswith('Person_')]))}"
            print(f"✨ 새 화자 등록: {pyannote_speaker} → {real_speaker_id}")
        
        # 매핑 저장
        self.session_speaker_mapping[pyannote_speaker] = real_speaker_id
        
        # 화자 프로필 생성/업데이트
        if real_speaker_id not in self.speaker_profiles:
            self.speaker_profiles[real_speaker_id] = SpeakerProfile(
                speaker_id=real_speaker_id,
                embeddings=[speaker_embedding],
                last_seen=current_time,
                total_duration=segment_duration,
                sample_count=1
            )
        else:
            profile = self.speaker_profiles[real_speaker_id]
            profile.embeddings.append(speaker_embedding)
            if len(profile.embeddings) > self.config.MAX_SPEAKER_EMBEDDINGS:
                profile.embeddings.pop(0)
            profile.last_seen = current_time
            profile.total_duration += segment_duration
            profile.sample_count += 1
        
        return real_speaker_id
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio input."""
        if status:
            print(f"[WARNING] Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        # Normalize to float32 (-1.0 to 1.0)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Add to queue for processing
        self.audio_queue.put(audio_data.copy())
        
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording from microphone."""
        self.pyaudio_instance = pyaudio.PyAudio()
        
        print("[INFO] Available audio devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            print(f"  {i}: {info['name']} (channels: {info['maxInputChannels']})")
        
        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=self.config.AUDIO_FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK_SIZE,
                stream_callback=self.audio_callback
            )
            
            self.is_recording = True
            self.audio_stream.start_stream()
            print("🎙️  Recording started. Press Ctrl+C to stop.")
            print("🔊 Listening for speech...")
            print("=" * 80)
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            self.cleanup()
    
    def process_audio_segment(self, audio_data: np.ndarray, base_time: float):
        """Process audio segment for diarization and transcription."""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write audio data to temporary file
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.config.CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.config.SAMPLE_RATE)
                    
                    # Convert float32 back to int16 for wave file
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                tmp_file_path = tmp_file.name
            
            # Perform diarization
            segment_duration = len(audio_data)/self.config.SAMPLE_RATE
            print(f"🔄 Processing audio segment ({segment_duration:.1f}s)...")
            diar_result = self.diar_pipeline(tmp_file_path)
            
            # Transcribe each speaker segment
            current_time = datetime.now()
            for turn, _, pyannote_speaker in diar_result.itertracks(yield_label=True):
                start_time = base_time + turn.start
                end_time = base_time + turn.end
                
                if turn.end - turn.start < self.config.MIN_SEG_DUR:
                    continue
                
                # Extract segment from audio data
                i0 = max(0, int(turn.start * self.config.SAMPLE_RATE))
                i1 = min(len(audio_data), int(turn.end * self.config.SAMPLE_RATE))
                segment_wav = audio_data[i0:i1]
                
                if segment_wav.size == 0:
                    continue
                
                # 실제 화자 ID 식별
                real_speaker = self.identify_speaker(pyannote_speaker, segment_wav, turn.end - turn.start)
                
                # Transcribe segment
                segments, _ = self.whisper_model.transcribe(
                    segment_wav,
                    language=self.config.WHISPER_LANG,
                    vad_filter=True,
                    beam_size=1,
                    word_timestamps=False
                )
                
                text = " ".join([s.text.strip() for s in segments if s.text])
                
                if text.strip():
                    timestamp = current_time.strftime("%H:%M:%S")
                    duration = end_time - start_time
                    
                    # 실시간 STT 결과 터미널 출력 (실제 화자 ID 사용)
                    print("=" * 80)
                    print(f"🎤 실시간 음성 인식 결과")
                    print(f"⏰ 시간: {timestamp}")
                    print(f"👤 화자: {real_speaker}")
                    if pyannote_speaker != real_speaker:
                        print(f"🔗 원본 ID: {pyannote_speaker}")
                    print(f"⏱️  지속시간: {duration:.2f}초")
                    print(f"💬 내용: {text}")
                    
                    # 대화 로그에 추가 (실제 화자 ID 사용)
                    entry = ConversationEntry(
                        id=str(uuid.uuid4()),
                        timestamp=current_time.isoformat(),
                        speaker=real_speaker,
                        text=text.strip(),
                        start_time=start_time,
                        end_time=end_time
                    )
                    self.conversation_log.append(entry)
                    
                    # 즉시 Qdrant에 저장
                    self.save_to_qdrant_realtime(entry)
                    
                    print("=" * 80)
                    print()  # 빈 줄 추가
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
    
    def processing_loop(self):
        """Main processing loop that runs in a separate thread."""
        process_buffer = []
        last_process_time = time.time()
        
        while self.is_recording:
            try:
                # Collect audio data
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    process_buffer.extend(audio_chunk)
                except queue.Empty:
                    continue
                
                current_time = time.time()
                
                # Process when buffer is large enough or enough time has passed
                buffer_duration = len(process_buffer) / self.config.SAMPLE_RATE
                
                if (buffer_duration >= self.config.PROCESS_INTERVAL or 
                    current_time - last_process_time >= self.config.PROCESS_INTERVAL):
                    
                    if len(process_buffer) > 0:
                        audio_array = np.array(process_buffer, dtype=np.float32)
                        
                        # Calculate base time for timestamps
                        base_time = current_time - buffer_duration
                        
                        # Process in separate thread to avoid blocking
                        processing_thread = threading.Thread(
                            target=self.process_audio_segment,
                            args=(audio_array, base_time)
                        )
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                        # Keep some overlap for continuity
                        overlap_samples = int(self.config.OVERLAP_DURATION * self.config.SAMPLE_RATE)
                        if len(process_buffer) > overlap_samples:
                            process_buffer = process_buffer[-overlap_samples:]
                        else:
                            process_buffer = []
                        
                        last_process_time = current_time
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] Processing loop error: {e}")
    
    def save_conversation(self):
        """대화 내용을 JSON으로 저장 (Qdrant는 실시간으로 이미 저장됨)"""
        if not self.conversation_log:
            print("[INFO] No conversation data to save.")
            return
        
        print(f"\n💾 JSON 파일로 대화 저장 중... ({len(self.conversation_log)}개 항목)")
        
        try:
            # JSON 저장을 위한 데이터 준비
            conversation_data = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_entries": len(self.conversation_log),
                "conversation": []
            }
            
            for entry in self.conversation_log:
                # JSON용 데이터 (임베딩은 제외)
                entry_dict = {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "duration": entry.end_time - entry.start_time
                }
                conversation_data["conversation"].append(entry_dict)
            
            # JSON 파일로 저장
            json_filename = self.config.OUTPUT_DIR / f"conversation_{self.session_id[:8]}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ JSON 저장 완료: {json_filename}")
            
            # 세션 종료 통계 출력
            if self.qdrant_client:
                speakers = set(entry.speaker for entry in self.conversation_log)
                total_duration = sum(entry.end_time - entry.start_time for entry in self.conversation_log)
                
                print("\n" + "=" * 60)
                print("📊 세션 통계")
                print("=" * 60)
                print(f"👥 감지된 화자: {', '.join(sorted(speakers))}")
                print(f"⏱️  총 발화 시간: {total_duration:.2f}초")
                print(f"💾 Qdrant에 저장된 항목: {len(self.conversation_log)}개")
                print(f"📁 JSON 파일: {json_filename.name}")
                print(f"👥 화자 프로필: {len(self.speaker_profiles)}명 저장")
                
                # 화자별 통계
                for speaker_id, profile in self.speaker_profiles.items():
                    print(f"   - {speaker_id}: {profile.total_duration:.1f}초 ({profile.sample_count}회 발화)")
                
                print("=" * 60)
            
        except Exception as e:
            print(f"❌ JSON 저장 실패: {e}")
    
    def search_conversation(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Qdrant에서 유사한 대화 내용 검색"""
        if not self.qdrant_client:
            print("[ERROR] Qdrant client not available.")
            return []
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # 검색 실행
            search_results = self.qdrant_client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "speaker": result.payload["speaker"],
                    "text": result.payload["text"],
                    "timestamp": result.payload["timestamp"],
                    "session_id": result.payload["session_id"]
                })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def save_to_qdrant_realtime(self, entry: ConversationEntry):
        """실시간으로 단일 엔트리를 Qdrant에 저장"""
        if not self.qdrant_client:
            return
        
        try:
            # 임베딩 생성
            embedding = self.embedding_model.encode([entry.text])[0].tolist()
            entry.embedding = embedding
            
            # Qdrant 포인트 생성
            point = PointStruct(
                id=entry.id,
                vector=embedding,
                payload={
                    "session_id": self.session_id,
                    "timestamp": entry.timestamp,
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "duration": entry.end_time - entry.start_time
                }
            )
            
            # Qdrant에 즉시 저장
            self.qdrant_client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=[point]
            )
            
            print(f"💾 Qdrant 저장 완료 (ID: {entry.id[:8]}...)")
            
        except Exception as e:
            print(f"❌ Qdrant 실시간 저장 실패: {e}")
    
    def run(self):
        """Main run method."""
        try:
            self.start_recording()
            
            # Start processing thread
            processing_thread = threading.Thread(target=self.processing_loop)
            processing_thread.daemon = True
            processing_thread.start()
            
            # Keep main thread alive
            while self.is_recording:
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n🛑 사용자가 중지를 요청했습니다...")
                    break
                    
        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 리소스 정리 중...")
        self.is_recording = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        # 대화 내용 저장
        print("📝 최종 JSON 파일 저장 중...")
        self.save_conversation()
        
        # 화자 프로필 저장
        self.save_speaker_profiles()
        
        print("✅ 정리 완료!")

def main():
    """Main function."""
    config = Config()
    
    print("🎯 실시간 화자 분리 및 음성 인식 시스템")
    print("=" * 60)
    print("📁 출력 디렉토리:", config.OUTPUT_DIR)
    print("💾 Qdrant 로컬 저장소:", config.QDRANT_PATH)
    print("🤖 임베딩 모델:", config.EMBEDDING_MODEL)
    print("=" * 60)
    print("🔧 모델 초기화 중... 잠시만 기다려주세요...")
    
    try:
        rt_diarization = RealTimeDiarization(config)
        rt_diarization.run()
    except Exception as e:
        print(f"❌ 애플리케이션 오류: {e}")
    
    print("👋 애플리케이션이 종료되었습니다.")

if __name__ == "__main__":
    main()