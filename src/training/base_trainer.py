#!/usr/bin/env python3
"""
Simplified BaseTrainer for PyTorch models.
Handles training, validation, checkpointing, and memory cleanup.
Compatible with CUDA and CPU.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cuda'):
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

    def _reset_training_state(self) -> None:
        """Reset training state for a new training run."""
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
        pass

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        pass

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, patience: int = 10,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Standard training loop."""
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            best_ckpt = output_path / 'best_model.pth'
            if best_ckpt.exists():
                best_ckpt.unlink()

        self._reset_training_state()
        self.setup_optimizer_and_loss()
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                if output_path:
                    self.best_model_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                if output_path:
                    self.save_checkpoint(output_path / 'best_model.pth')
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})"
                )
                break

            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_acc)

        training_time = time.time() - start_time

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        elif output_path and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')

        self.cleanup_memory()

        return {
            'total_epochs': epoch,
            'training_time_seconds': training_time,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        try:
            torch.save({
                'model_state': {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                },
                'optimizer_state': (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                'scheduler_state': (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                'best_val_acc': self.best_val_acc,
                'history': self.history,
                'config': self.config
            }, path)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        try:
            # weights_only=False needed for PyTorch 2.6+ compatibility
            # Checkpoints contain numpy arrays in history/config
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint['model_state'])

            if self.optimizer and checkpoint.get('optimizer_state'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            if self.scheduler and checkpoint.get('scheduler_state'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])

            self.history = checkpoint.get('history', self.history)
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    @abstractmethod
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        pass