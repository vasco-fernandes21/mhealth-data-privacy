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
from copy import deepcopy
import json
import time
from datetime import datetime


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """Initialize trainer."""
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)

        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    @abstractmethod
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        pass

    @abstractmethod
    def train_epoch(
        self, train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train one epoch. Returns (loss, accuracy)."""
        pass

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model. Returns (loss, accuracy).

        Args:
            val_loader: Validation data loader

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.detach().item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0

        return avg_loss, avg_acc

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

        self._reset_training_state()
        self.setup_optimizer_and_loss()

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Log
            self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            # Early stopping with best model tracking
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0

                # Save best model in memory
                self.best_model_state = deepcopy(self.model.state_dict())

                # Also save to disk if output_dir provided
                if output_path is not None:
                    self.save_checkpoint(output_path / 'best_model.pth')
            else:
                self.epochs_no_improve += 1

            # Step ReduceLROnPlateau scheduler (if applicable)
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_acc)

            # Early stopping check
            if self.epochs_no_improve >= patience:
                self._log_early_stopping(epoch, patience)
                break

        training_time = time.time() - start_time

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        elif output_path is not None and \
             (output_path / 'best_model.pth').exists():
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

    def _reset_training_state(self) -> None:
        """Reset training state."""
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch results."""
        print(f"Epoch {epoch:03d}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    def _log_early_stopping(self, epoch: int, patience: int) -> None:
        """Log early stopping."""
        print(f"Early stopping triggered after {epoch} epochs "
              f"(patience={patience})")

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
            'optimizer_state': (self.optimizer.state_dict()
                               if self.optimizer else None),
            'scheduler_state': (self.scheduler.state_dict()
                               if self.scheduler else None),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
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
        if self.optimizer is not None and \
           checkpoint.get('optimizer_state') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.scheduler is not None and \
           checkpoint.get('scheduler_state') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

    @abstractmethod
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with detailed metrics
        """
        pass