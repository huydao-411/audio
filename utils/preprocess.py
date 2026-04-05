"""
Data Preprocessing Module for Audio Event Detection
Handles loading, processing, and augmentation of audio datasets
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Preprocessor for audio event detection datasets
    Supports UrbanSound8K, ESC-50, FSD50K, and custom datasets
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        self.target_sr = self.preprocessing_config['target_sample_rate']
        self.duration = self.preprocessing_config['duration']
        self.n_mels = self.preprocessing_config['n_mels']
        self.n_fft = self.preprocessing_config['n_fft']
        self.hop_length = self.preprocessing_config['hop_length']
        
        # Class mapping
        self.class_mapping = self._create_class_mapping()
        
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create mapping from class names to labels"""
        mapping = {}
        for class_info in self.config['target_classes']:
            mapping[class_info['name']] = class_info['label']
        return mapping
    
    
    def load_audio(self, file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary
        
        Args:
            file_path: Path to audio file
            sr: Target sample rate (uses config default if None)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if sr is None:
            sr = self.target_sr
            
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None, None
    
    def pad_or_truncate(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate audio to target length
        
        Args:
            audio: Audio array
            target_length: Target length in samples
            
        Returns:
            Processed audio array
        """
        if len(audio) > target_length:
            # Truncate from center
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            pad_width = target_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        
        return audio
    
    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence from audio using librosa's robust effects
        """
        if not self.config['preprocessing'].get('remove_silence', False):
            return audio
            
        intervals = librosa.effects.split(audio, top_db=30)
        
        if len(intervals) > 0:
            non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])
            return non_silent_audio
            
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio
        """
        if not self.preprocessing_config['normalize']:
            return audio
            
        # Peak normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract mel-spectrogram from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Mel-spectrogram (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.preprocessing_config['fmin'],
            fmax=self.preprocessing_config['fmax'],
            window=self.preprocessing_config['window']
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features (n_mfcc, time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def process_audio_file(self, file_path: str, extract_features: bool = True) -> Dict:
        """
        Complete preprocessing pipeline for a single audio file
        
        Args:
            file_path: Path to audio file
            extract_features: Whether to extract mel-spectrogram
            
        Returns:
            Dictionary with processed audio and features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        if audio is None:
            return None
        
        # Remove silence
        audio = self.remove_silence(audio)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Pad or truncate to fixed duration
        target_length = int(self.duration * sr)
        audio = self.pad_or_truncate(audio, target_length)
        
        result = {
            'audio': audio,
            'sample_rate': sr,
            'file_path': file_path
        }
        
        # Extract features if requested
        if extract_features:
            mel_spec = self.extract_mel_spectrogram(audio, sr)
            result['mel_spectrogram'] = mel_spec
            result['mfcc'] = self.extract_mfcc(audio, sr)
        
        return result
    
    def load_urbansound8k(self, data_path: str) -> pd.DataFrame:
        """
        Load UrbanSound8K dataset
        
        Args:
            data_path: Path to UrbanSound8K directory
            
        Returns:
            DataFrame with file paths and labels
        """
        
        metadata_path = os.path.join(data_path, 'metadata', 'UrbanSound8K.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        
  
        us8k_to_target = {
            'gun_shot': 'gunshot',
            'siren': 'siren',
            'dog_bark': 'dog_bark',
            'glass_breaking': 'glass_breaking'
        }

        metadata['target_class'] = metadata['class'].map(us8k_to_target).fillna('normal')
        metadata['label'] = metadata['target_class'].map(self.class_mapping)
        
        # Create full file paths
        metadata['file_path'] = metadata.apply(
            lambda row: os.path.join(data_path, 'audio', f"fold{row['fold']}", row['slice_file_name']).replace('\\', '/'),
            axis=1
        )
        return metadata[['file_path', 'target_class', 'label', 'fold']]
    
    def load_esc50(self, data_path: str) -> pd.DataFrame:
        """
        Load ESC-50 dataset
        
        Args:
            data_path: Path to ESC-50 directory
            
        Returns:
            DataFrame with file paths and labels
        """
        metadata_path = os.path.join(data_path, 'meta', 'esc50.csv')
        print(metadata_path)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)

        esc50_to_target = {
            'crying_baby': 'scream',
            'fireworks': 'explosion',
            'crackling_fire': 'fire_crackling',
            'glass_breaking': 'glass_breaking'
        }

        metadata['target_class'] = metadata['category'].map(esc50_to_target).fillna('normal')
        metadata['label'] = metadata['target_class'].map(self.class_mapping)

        # Create full file paths
        metadata['file_path'] = metadata.apply(
            lambda row: os.path.join(data_path, 'audio', row['filename']).replace('\\', '/'),
            axis=1
        )
        
        return metadata[['file_path', 'target_class', 'label', 'fold']]
    
    def merge_datasets(self, output_path: str = "data/processed/merged_dataset.csv"):
        """
        Merge all datasets into a single DataFrame
        
        Args:
            output_path: Path to save merged dataset
            
        Returns:
            Merged DataFrame
        """
        all_data = []
        
        # Load UrbanSound8K
        try:
            us8k_path = self.config['datasets']['urbansound8k']['path']
            us8k_data = self.load_urbansound8k(us8k_path)
            us8k_data['dataset'] = 'urbansound8k'
            all_data.append(us8k_data)
            print(f"Loaded {len(us8k_data)} samples from UrbanSound8K")
        except Exception as e:
            print(f"Error loading UrbanSound8K: {str(e)}")
        
        # Load ESC-50
        try:
            esc50_path = self.config['datasets']['esc50']['path']
            esc50_data = self.load_esc50(esc50_path)
            esc50_data['dataset'] = 'esc50'
            all_data.append(esc50_data)
            print(f"Loaded {len(esc50_data)} samples from ESC-50")
        except Exception as e:
            print(f"Error loading ESC-50: {str(e)}")
        
        # Merge all datasets
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_df.to_csv(output_path, index=False)
            
            print(f"\nMerged dataset saved to {output_path}")
            print(f"Total samples: {len(merged_df)}")
            print("\nClass distribution:")
            print(merged_df['target_class'].value_counts())
            return merged_df
        else:
            print("No datasets loaded successfully")
            return None
    
    def preprocess_dataset(self, 
                          metadata_df: pd.DataFrame, 
                          output_dir: str = "data/processed/spectrograms",
                          save_format: str = "npy") -> None:
        """
        Preprocess entire dataset and save features
        
        Args:
            metadata_df: DataFrame with file paths and labels
            output_dir: Directory to save processed features
            save_format: Format to save features ('npy' or 'h5')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = []
        
        print(f"Processing {len(metadata_df)} audio files...")
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            result = self.process_audio_file(row['file_path'])
            
            if result is not None:
                # Save mel-spectrogram
                output_filename = f"{idx:05d}_{row['target_class']}.npy"
                output_path = os.path.join(output_dir, output_filename).replace('\\', '/')
                
                np.save(output_path, result['mel_spectrogram'])
                
                processed_data.append({
                    'index': idx,
                    'feature_path': output_path,
                    'target_class': row['target_class'],
                    'label': row['label'],
                    'dataset': row['dataset'],
                    'fold': row['fold']
                })
        
        # Save processed metadata
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(os.path.join(output_dir, 'processed_metadata.csv'), index=False)
        
        print(f"\nProcessed {len(processed_data)} files successfully")
        print(f"Features saved to {output_dir}")


def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("Audio Event Detection - Data Preprocessing")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(config_path="../audio_event_detection/configs/config.yaml")
    
    # Merge datasets
    print("\n1. Merging datasets...")
    merged_df = preprocessor.merge_datasets()
    
    if merged_df is not None:
        # Preprocess and extract features
        print("\n2. Extracting features...")
        preprocessor.preprocess_dataset(merged_df)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
