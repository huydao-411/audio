"""
Inference Script for Audio Event Detection
Supports both batch and real-time inference
"""

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import yaml
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.append('audio_event_detection')
from models.ast_model import AudioSpectrogramTransformer



class AudioEventDetector:
    """
    Audio event detector for inference
    """
    def __init__(self, model_path, config_path, device, root_path):
    # def __init__(self, model_path: str, config_path: str = "configs/config.yaml",
    #              device: str = "cuda", root_path):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.root_path = root_path
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        self.inference_config = self.config['inference']
        self.num_classes = self.config['model']['num_classes']
        
        # Class names
        self.class_names = [c['name'] for c in self.config['target_classes']]
        
        # Audio parameters
        self.target_sr = self.preprocessing_config['target_sample_rate']
        self.duration = self.preprocessing_config['duration']
        self.n_mels = self.preprocessing_config['n_mels']
        self.n_fft = self.preprocessing_config['n_fft']
        self.hop_length = self.preprocessing_config['hop_length']
        
        # Inference parameters
        self.confidence_threshold = self.inference_config['confidence_threshold']
        self.top_k = self.inference_config['top_k']
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        

        
        print(f"Detector initialized on {self.device}")
        print(f"Loaded model from: {model_path}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        config_path = str(self.root_path / "configs" / "config.yaml")
        
        # 1. Load the checkpoint first to see what the brain actually knows
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # 2. Fix the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 3. Detect the correct number of classes from the checkpoint weights
        # The 'head.weight' shape is [num_classes, embedding_dim]
        checkpoint_classes = new_state_dict['head.weight'].shape[0]
        print(f"Overriding config: Checkpoint has {checkpoint_classes} classes.")

        # 4. Create model with the CORRECT number of classes
        model = AudioSpectrogramTransformer(
            config_path=config_path,
            # If your AST class allows passing num_classes directly, do it here:
            # num_classes=checkpoint_classes 
        )
        
        # NOTE: If AudioSpectrogramTransformer ONLY reads from the YAML file, 
        # you MUST manually edit configs/config.yaml to set num_classes: 8
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        
        return model
    # def _load_model(self, model_path: str) -> nn.Module:
    #     """
    #     Load trained model
        
    #     Args:
    #         model_path: Path to model checkpoint
            
    #     Returns:
    #         Loaded model
    #     """
    #     config_path = str(self.root_path / "configs" / "config.yaml")
        
    #     # Create model
    #     model = AudioSpectrogramTransformer(
    #         config_path=config_path
    #     )
        
    #     # Load checkpoint
    #     checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(self.device)
        
    #     return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for inference
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed spectrogram tensor
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Pad or truncate
        target_length = int(self.duration * sr)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.preprocessing_config['fmin'],
            fmax=self.preprocessing_config['fmax']
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        
        return mel_spec_tensor
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict audio event for a single file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        input_tensor = self.preprocess_audio(audio_path).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[::-1][:self.top_k]
        
        predictions = []
        for idx in top_k_indices:
            if probabilities[idx] >= self.confidence_threshold:
                predictions.append({
                    'class': self.class_names[idx],
                    'confidence': float(probabilities[idx]),
                    'label': int(idx)
                })
        
        result = {
            'file': audio_path,
            'predictions': predictions,
            'all_probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(self.num_classes)
            }
        }
        
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Predict for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
                print(f"Processed: {audio_path}")
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                results.append({
                    'file': audio_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_real_time(self, audio_chunk: np.ndarray, sr: int) -> Dict:
        """
        Predict for real-time audio chunk
        
        Args:
            audio_chunk: Audio array
            sr: Sample rate
            
        Returns:
            Prediction dictionary
        """
        # Resample if necessary
        if sr != self.target_sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.target_sr)
        
        # Normalize
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        # Pad or truncate
        target_length = int(self.duration * self.target_sr)
        if len(audio_chunk) > target_length:
            audio_chunk = audio_chunk[:target_length]
        elif len(audio_chunk) < target_length:
            audio_chunk = np.pad(audio_chunk, (0, target_length - len(audio_chunk)), mode='constant')
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx]
        
        if confidence >= self.confidence_threshold:
            return {
                'class': self.class_names[pred_idx],
                'confidence': float(confidence),
                'label': int(pred_idx)
            }
        else:
            return None


def main():
    PROJECT_ROOT = Path.cwd().parent / 'audio_event_detection'
    print(f"{PROJECT_ROOT}")
    """Main inference script"""
    parser = argparse.ArgumentParser(description='Audio Event Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--output', type=str, default='results/predictions.json', help='Output file')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    print(args)
    
    print("="*60)
    print("Audio Event Detection - Inference")
    print("="*60)
    
    # Initialize detector
    detector = AudioEventDetector(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        root_path=PROJECT_ROOT  # Pass it here
    )
    
    print(detector)
    # Get audio files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [str(input_path)]
    elif input_path.is_dir():
        audio_files = [
            str(f) for f in input_path.glob('**/*')
            if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        ]
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Run inference
    results = detector.predict_batch(audio_files)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Inference Summary")
    print("="*60)
    
    for result in results:
        if 'predictions' in result and result['predictions']:
            print(f"\nFile: {Path(result['file']).name}")
            for pred in result['predictions']:
                print(f"  - {pred['class']}: {pred['confidence']:.2%}")


if __name__ == "__main__":
    
    main()
# python3 -m scripts.inference --model /Users/dangchauanh/Downloads/agent-artifacts-zip_4a7a084c-cf0c-43c1-88b8-c7669caa2676_1772094907/audio_event_detection/results/checkpoints/best_model.pth --input /Users/dangchauanh/Downloads/agent-artifacts-zip_4a7a084c-cf0c-43c1-88b8-c7669caa2676_1772094907/audio_event_detection/data/raw/ESC-50/ESC-50-master/audio/1-23222-A-19.wav