"""
Real-time Audio Event Detection Demo
Captures audio from microphone and detects emergency events in real-time
"""

import pyaudio
import numpy as np
import torch
import librosa
from collections import deque
import time
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.inference import AudioEventDetector


class RealTimeDetector:
    """
    Real-time audio event detector using microphone input
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = "configs/config.yaml",
                 device: str = "cuda"):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to run on
        """
        # Initialize detector
        self.detector = AudioEventDetector(
            model_path=model_path, 
            config_path=config_path, 
            device=device,
            root_path=PROJECT_ROOT  # THÊM DÒNG NÀY
        )
        
        # Audio parameters
        self.sample_rate = 22050
        self.chunk_duration = 1.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.buffer_duration = 4.0  # seconds (model input duration)
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Detection settings
        self.detection_interval = 0.5  # seconds between detections
        self.last_detection_time = 0
        
        print("Real-time detector initialized")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Buffer duration: {self.buffer_duration}s")
    
    def start_stream(self):
        """Start audio stream from microphone"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("\n" + "="*60)
        print("Real-time Audio Event Detection Started")
        print("="*60)
        print("Listening for emergency sounds...")
        print("Press Ctrl+C to stop\n")
        
        self.stream.start_stream()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio stream
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Stream status
            
        Returns:
            Tuple of (None, pyaudio.paContinue)
        """
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Detect if buffer is full and enough time has passed
        current_time = time.time()
        if (len(self.audio_buffer) >= self.buffer_size and 
            current_time - self.last_detection_time >= self.detection_interval):
            
            # Get audio from buffer
            audio_array = np.array(list(self.audio_buffer))
            
            # Run detection
            result = self.detector.predict_real_time(audio_array, self.sample_rate)
            
            # Display result
            if result is not None:
                self.display_detection(result)
            
            self.last_detection_time = current_time
        
        return (None, pyaudio.paContinue)
    
    def display_detection(self, result: dict):
        """
        Display detection result
        
        Args:
            result: Detection result dictionary
        """
        class_name = result['class']
        confidence = result['confidence']
        
        # Color coding for different events
        colors = {
            'gunshot': '\033[91m',      # Red
            'explosion': '\033[91m',     # Red
            'siren': '\033[93m',         # Yellow
            'glass_breaking': '\033[93m', # Yellow
            'scream': '\033[91m',        # Red
            'dog_bark': '\033[93m',      # Yellow
            'fire_crackling': '\033[91m' # Red
        }
        
        reset_color = '\033[0m'
        color = colors.get(class_name, '\033[92m')  # Default green
        
        # Print detection
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] DETECTED: {class_name.upper()} "
              f"(Confidence: {confidence:.2%}){reset_color}")
    
    def stop_stream(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        print("\n" + "="*60)
        print("Real-time detection stopped")
        print("="*60)
    
    def run(self):
        """Run real-time detection"""
        try:
            self.start_stream()
            
            # Keep running until interrupted
            while self.stream.is_active():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop_stream()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Audio Event Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python scripts/train.py")
        return
    
    # Initialize and run detector
    detector = RealTimeDetector(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    detector.run()


if __name__ == "__main__":
    main()
