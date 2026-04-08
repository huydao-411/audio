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
        feature_path = row['feature_path']
        mel_spec = np.load(feature_path)
        
        # Get label
        label = row['label']
        
        # Apply transforms
        if self.transform is not None and self.mode == 'train':
            mel_spec = self.transform(mel_spec)
        
        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
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
                       processed_metadata_path: str = "data/processed/spectrograms/processed_metadata.csv",
                       augmented_metadata_path: str = "data/metadata/augmented_metadata.csv",
                       batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config_path: Path to configuration file
        processed_metadata_path: Path to processed metadata CSV
        batch_size: Batch size (uses config default if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # Load processed metadata
    df_orig = pd.read_csv(processed_metadata_path)
    if os.path.exists(augmented_metadata_path):
        df_aug = pd.read_csv(augmented_metadata_path)
        metadata = pd.concat([df_orig, df_aug], ignore_index=True)
        print(f"Merged data: Original ({len(df_orig)}) + Augmentation ({len(df_aug)}) = Total ({len(metadata)})")
    else:
        metadata = df_orig
        print(f"Can not find the augmented file. Used original file: ({len(df_orig)})")
            
    
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
