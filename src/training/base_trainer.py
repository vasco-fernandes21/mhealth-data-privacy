#!/usr/bin/env python3
"""
Abstract base class for all trainers.

Defines common training interface for all trainer implementations.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import time
from datetime import datetime


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        
        # Initialize state
        self.optimizer = None
        self.criterion = None
        self.best_val_loss = float('inf')
        self.best_val_acc = -1.0
        self.epochs_no_improve = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch': []
        }
    
    @abstractmethod
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            (loss, accuracy) for the epoch
        """
        pass
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            (loss, accuracy) on validation set
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_loss = total_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 10,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            output_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with training results
        """
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss
        self.setup_optimizer_and_loss()
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log
            self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                
                # Save checkpoint
                if output_path is not None:
                    self.save_checkpoint(output_path / 'best_model.pth')
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= patience:
                self._log_early_stopping(epoch, patience)
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        if output_path is not None and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
        
        return {
            'total_epochs': epoch,
            'training_time_seconds': training_time,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history
        }
    
    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch results."""
        print(f"Epoch {epoch:03d}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    
    def _log_early_stopping(self, epoch: int, patience: int) -> None:
        """Log early stopping."""
        print(f"Early stopping triggered after {epoch} epochs (patience={patience})")
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': len(self.history['epoch']),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        if self.optimizer is not None and checkpoint['optimizer_state'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save training results to JSON.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        results['device'] = str(self.device)
        results['model_class'] = self.model.__class__.__name__
        
        # Save results
        results_file = output_path / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {results_file}")