import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import yaml
from typing import Tuple, Optional, List
import random


class AudioEventDataset(Dataset):
    """
    PyTorch Dataset for Audio Event Detection
    """
    
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 config_path: str = "configs/config.yaml",
                 mode: str = "train",
                 transform=None):
        """
        Initialize dataset
        
        Args:
            metadata_df: DataFrame with file paths and labels
            config_path: Path to configuration file
            mode: 'train', 'val', or 'test'
            transform: Optional transform to apply
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.num_classes = self.config['model']['num_classes']
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (spectrogram, label)
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load preprocessed spectrogram
        feature_path = row.get('file_path', row.get('feature_path'))
        mel_spec = np.load(feature_path)
        
        # Pad or truncate to fixed length
        # (models like AST or DataLoader need fixed size for batched tensor stack)
        if 'ast' in self.config.get('model', {}):
            target_frames = self.config['model']['ast']['input_size'][1]
        else:
            target_frames = int(self.config['preprocessing']['duration'] * self.config['preprocessing']['target_sample_rate'] // self.config['preprocessing']['hop_length']) + 1

        if mel_spec.shape[1] > target_frames:
            mel_spec = mel_spec[:, :target_frames]
        elif mel_spec.shape[1] < target_frames:
            pad_width = target_frames - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0) # -80dB is silence
        
        # Get label
        label = row['label']
        
        # Apply transforms
        if self.transform is not None and self.mode == 'train':
            mel_spec = self.transform(mel_spec)
        
        # Convert to tensor (ensure contiguous memory with copy() to avoid DataLoader resize errors)
        mel_spec = torch.FloatTensor(mel_spec.copy()).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([label])
        return mel_spec, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            Tensor of class weights
        """
        class_counts = self.metadata['label'].value_counts().sort_index()
        total_samples = len(self.metadata)
        
        weights = []
        for i in range(self.num_classes):
            if i in class_counts.index:
                weight = total_samples / (self.num_classes * class_counts[i])
            else:
                weight = 0.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)


class RawAudioDataset(Dataset):
    """
    Dataset that loads raw audio files (for real-time processing)
    """
    
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 config_path: str = "configs/config.yaml",
                 mode: str = "train"):
        """
        Initialize dataset
        
        Args:
            metadata_df: DataFrame with file paths and labels
            config_path: Path to configuration file
            mode: 'train', 'val', or 'test'
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        self.target_sr = self.preprocessing_config['target_sample_rate']
        self.duration = self.preprocessing_config['duration']
        self.n_mels = self.preprocessing_config['n_mels']
        self.n_fft = self.preprocessing_config['n_fft']
        self.hop_length = self.preprocessing_config['hop_length']
        self.num_classes = self.config['model']['num_classes']
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (spectrogram, label)
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio, sr = librosa.load(row['file_path'], sr=self.target_sr, mono=True)
        
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
            n_mels=self.n_mels
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        label = torch.LongTensor([row['label']])
        
        return mel_spec, label


def create_data_loaders(config_path: str = "configs/config.yaml",
                       dataset_path: str = "data/final_dataset/final_dataset.csv",
                       batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config_path: Path to configuration file
        dataset_path: Path to unified dataset CSV
        batch_size: Batch size (uses config default if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # Load dataset
    if os.path.exists(dataset_path):
        metadata = pd.read_csv(dataset_path)
    else:
        raise FileNotFoundError(f"Cannot find dataset at {dataset_path}")
            
    
    # Split by fold (using UrbanSound8K fold structure)
    # Folds 1-7: train, Fold 8-9: validation, Fold 10: test
    train_df = metadata[metadata['fold'] <= 7]
    val_df = metadata[metadata['fold'].isin([8, 9])]
    test_df = metadata[metadata['fold'] == 10]
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = AudioEventDataset(train_df, config_path, mode='train')
    val_dataset = AudioEventDataset(val_df, config_path, mode='val')
    test_dataset = AudioEventDataset(test_df, config_path, mode='test')
    
    # Create data loaders
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset loading"""
    print("Testing dataset...")
    
    # Create dummy metadata
    dummy_data = {
        'feature_path': ['data/processed/spectrograms/00000_gunshot.npy'] * 10,
        'target_class': ['gunshot'] * 10,
        'label': [0] * 10,
        'fold': [1] * 10
    }
    
    metadata_df = pd.DataFrame(dummy_data)
    
    # Create dataset
    dataset = AudioEventDataset(
        metadata_df,
        config_path="configs/config.yaml",
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"Class weights: {weights}")
    
    print("Dataset test complete!")

def test_raw_audio_dataset():
    print("\n--- Testing RawAudioDataset ---")
    
    test_audio_path = "data/raw/UrbanSound8K/audio/fold1/7061-6-0-0.wav" 
    
    if not os.path.exists(test_audio_path):
        print(f"⚠️ Can't find file {test_audio_path}. Check the path!")
        return

    dummy_data = {
        'file_path': [test_audio_path],
        'label': [0],
        'fold': [1]
    }
    metadata_df = pd.DataFrame(dummy_data)
    
    try:
        dataset = RawAudioDataset(
            metadata_df,
            config_path="configs/config.yaml",
            mode='train'
        )
        
        mel_spec, label = dataset[0]
        
        print(f"Successfully!")
        print(f"Shape of Spectrogram: {mel_spec.shape}")
        print(f"Label: {label.item()}")
        print(f"Max/Min Valuable in dB: {mel_spec.max():.2f} / {mel_spec.min():.2f}")
        
    except Exception as e:
        print(f"Error in test RawAudioDataset: {e}")

if __name__ == "__main__":
    test_dataset()
    test_raw_audio_dataset()
