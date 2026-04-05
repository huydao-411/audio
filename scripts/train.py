"""
Training Pipeline for Audio Event Detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import yaml
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import time
from datetime import datetime
from pathlib import Path

# Import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.ast_model import AudioSpectrogramTransformer, count_parameters
from utils.metrics import MetricsCalculator
from models.losses import FocalLoss


class Trainer:
    """
    Trainer class for audio event detection model
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config_path: str = "configs/config.yaml",
                 device: str = "cuda"):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config_path: Path to configuration file
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.num_classes = self.config['model']['num_classes']
        
        # Training parameters
        self.num_epochs = self.training_config['num_epochs']
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(self.num_classes)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.saved_checkpoints = []  # Track saved checkpoint paths
        
        # Setup logging
        self.writer = None
        if self.config['logging']['tensorboard']:
            log_dir = os.path.join(self.config['paths']['logs_dir'], 
                                  f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            
        # Setup Weights & Biases (wandb)
        self.use_wandb = self.config['logging'].get('wandb', {}).get('enabled', False)
        if self.use_wandb:
            wandb_project = self.config['logging']['wandb'].get('project', 'audio-event-detection')
            wandb.init(
                project=wandb_project,
                config=self.config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Mixed precision training
        self.use_amp = self.training_config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        print(f"Trainer initialized")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = self.training_config['optimizer'].lower()
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.training_config['betas']
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.training_config['scheduler'].lower()
        
        if scheduler_name == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.training_config['min_lr']
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function"""
        if self.training_config['focal_loss']['enabled']:
            criterion = FocalLoss(
                alpha=self.training_config['focal_loss']['alpha'],
                gamma=self.training_config['focal_loss']['gamma'],
                num_classes=self.num_classes
            )
        else:
            # Use weighted cross-entropy for class imbalance
            if self.training_config['class_weights'] == 'balanced':
                # Calculate class weights from training data
                # This should be computed from actual data distribution
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).squeeze()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.training_config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.training_config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).squeeze()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Manage max_checkpoints limit
        self.saved_checkpoints.append(checkpoint_path)
        max_checkpoints = self.config['logging']['checkpoint'].get('max_checkpoints', 5)
        if max_checkpoints > 0 and len(self.saved_checkpoints) > max_checkpoints:
            oldest_ckpt = self.saved_checkpoints.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with F1: {metrics['f1_score']:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"LR: {current_lr:.6f}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train F1: {train_metrics['f1_score']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_score']:.4f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/F1', train_metrics['f1_score'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/F1', val_metrics['f1_score'], epoch)
                
            # WandB logging
            if getattr(self, 'use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'Learning_Rate': current_lr,
                    'Train/Loss': train_metrics['loss'],
                    'Train/F1': train_metrics['f1_score'],
                    'Val/Loss': val_metrics['loss'],
                    'Val/F1': val_metrics['f1_score']
                })
            
            # Save checkpoint
            is_best = val_metrics['f1_score'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_score']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['logging']['checkpoint']['save_frequency'] == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.training_config['early_stopping']['enabled']:
                patience = self.training_config['early_stopping']['patience']
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print("="*60)
        
        if self.writer:
            self.writer.close()
            
        if getattr(self, 'use_wandb', False):
            wandb.finish()


def main():
    """Main training script"""
    print("Audio Event Detection - Training")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = AudioSpectrogramTransformer(config_path)
    
    # Load data loaders (this would be implemented)
    # For now, this is a placeholder
    print("\nNote: Data loaders need to be created from preprocessed data")
    print("Run preprocess.py first to prepare the dataset")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config_path, device)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
