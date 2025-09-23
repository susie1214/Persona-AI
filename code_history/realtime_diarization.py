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

class RealTimeDiarization:
    def __init__(self, config: Config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.BUFFER_DURATION * config.SAMPLE_RATE))
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        self.last_speakers = {}  # 화자 연속성 추적
        
        # 모델 로드
        self.diar_pipeline = self.load_diarization_pipeline()
        self.whisper_model = self.load_whisper_model()
        
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
            print("[INFO] Recording started. Press Ctrl+C to stop.")
            
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
            print(f"[INFO] Processing audio segment ({len(audio_data)/self.config.SAMPLE_RATE:.1f}s)...")
            diar_result = self.diar_pipeline(tmp_file_path)
            
            # Transcribe each speaker segment
            current_time = datetime.now()
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
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
                    print(f"[{timestamp}] {speaker}: {text}")
            
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
                    print("\n[INFO] Stopping...")
                    break
                    
        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[INFO] Cleaning up...")
        self.is_recording = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        print("[INFO] Cleanup completed.")

def main():
    """Main function."""
    config = Config()
    
    print("[INFO] Starting real-time diarization...")
    print("[INFO] This may take a moment to initialize models...")
    
    try:
        rt_diarization = RealTimeDiarization(config)
        rt_diarization.run()
    except Exception as e:
        print(f"[ERROR] Application error: {e}")
    
    print("[INFO] Application finished.")

if __name__ == "__main__":
    main()