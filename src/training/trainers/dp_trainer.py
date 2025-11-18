#!/usr/bin/env python3
"""
DP Trainer - Differential Privacy training using Opacus.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

from opacus import PrivacyEngine
from src.training.base_trainer import BaseTrainer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DPConfig:
    """Configuration for Differential Privacy training."""

    def __init__(self, config: Dict[str, Any]):
        dp_cfg = config.get('differential_privacy', {})
        self.enabled = bool(dp_cfg.get('enabled', False))
        self.noise_multiplier = float(dp_cfg.get('noise_multiplier', 0.9))
        self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        self.delta = float(dp_cfg.get('delta', 1e-5))
        self.poisson_sampling = bool(dp_cfg.get('poisson_sampling', True))
        self.grad_sample_mode = str(dp_cfg.get('grad_sample_mode', 'hooks'))

    def __repr__(self) -> str:
        return (
            f"DPConfig(noise_mult={self.noise_multiplier}, "
            f"max_grad={self.max_grad_norm}, delta={self.delta})"
        )


class DPTrainer(BaseTrainer):
    """Differential Privacy Trainer using Opacus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = DPConfig(self.config)
        self.privacy_engine = None
        self.dp_train_loader = None
        self.best_model_state = None

    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss for DP training."""
        cfg = self.config['training']
        lr = float(cfg.get('learning_rate', 1e-3))
        weight_decay = float(cfg.get('weight_decay', 0.0))
        opt_name = cfg.get('optimizer', 'sgd').lower()

        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            logger.info(f"Using AdamW: lr={lr}, weight_decay={weight_decay}")
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            logger.info(f"Using Adam: lr={lr}, weight_decay={weight_decay}")
        else:
            momentum = float(cfg.get('momentum', 0.7))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
            logger.info(
                f"Using SGD: lr={lr}, momentum={momentum}, "
                f"weight_decay={weight_decay}"
            )

        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = self.criterion.to(self.device)
        self._setup_scheduler()

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        cfg = self.config['training']
        epochs = int(cfg.get('epochs', 20))
        scheduler_name = cfg.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5,
                min_lr=1e-6
            )
        else:
            self.scheduler = None

    def setup_privacy_engine(self, train_loader: DataLoader) -> DataLoader:
        """Setup Opacus PrivacyEngine."""
        if not self.dp_config.enabled:
            logger.info("DP disabled")
            return train_loader

        logger.info(f"Setting up PrivacyEngine: {self.dp_config}")

        try:
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dp_train_loader = (
                privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=train_loader,
                    noise_multiplier=self.dp_config.noise_multiplier,
                    max_grad_norm=self.dp_config.max_grad_norm,
                    poisson_sampling=self.dp_config.poisson_sampling,
                    clipping_mode='flat',
                    grad_sample_mode=self.dp_config.grad_sample_mode
                )
            )
            self.privacy_engine = privacy_engine
            logger.info("PrivacyEngine ready")
            return self.dp_train_loader
        except Exception as e:
            logger.error(f"PrivacyEngine setup failed: {e}")
            raise

    def _get_epsilon(self) -> Optional[float]:
        """Get current epsilon."""
        if self.privacy_engine is None:
            return None

        try:
            return float(
                self.privacy_engine.get_epsilon(self.dp_config.delta)
            )
        except Exception:
            return None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loader = (self.dp_train_loader if self.dp_train_loader
                 else train_loader)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item() * batch_x.size(0)

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        if (self.scheduler is not None and
                not isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                )):
            self.scheduler.step()

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 20, patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop with DP."""
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        self._reset_training_state()
        self.setup_optimizer_and_loss()
        self.setup_privacy_engine(train_loader)

        start_time = time.time()

        print(f"\nDP Training")
        print(f"  Noise multiplier: {self.dp_config.noise_multiplier}")
        print(f"  Max grad norm: {self.dp_config.max_grad_norm}")
        print(f"  Delta: {self.dp_config.delta}")
        print(f"  Epochs: {epochs}\n")

        best_epoch = 0
        epoch_num = 0

        for epoch_num in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            epsilon = self._get_epsilon()

            self.history['epoch'].append(epoch_num)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            eps_str = f"ε={epsilon:.4f}" if epsilon else "ε=N/A"

            print(
                f"Epoch {epoch_num:02d}: loss={train_loss:.4f} "
                f"acc={train_acc:.4f} | val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} | {eps_str}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                best_epoch = epoch_num

                # Store in memory always
                self.best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                if output_path:
                    self.save_checkpoint(output_path / 'best_model.pth')
                    logger.info(
                        f"Saved best model at epoch {epoch_num} "
                        f"(val_acc={val_acc:.4f})"
                    )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch_num}")
                    break

            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_acc)

        elapsed = time.time() - start_time

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model from memory")
        elif output_path and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
            logger.info("Loaded best model from disk")

        epsilon = self._get_epsilon()

        return {
            'total_epochs': epoch_num,
            'best_epoch': best_epoch,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history,
            'final_epsilon': epsilon
        }

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation with metrics."""
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)

        (precision_per_class, recall_per_class, f1_per_class, _) = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        epsilon = self._get_epsilon()

        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(
                precision_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'recall': float(
                recall_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'f1_score': float(
                f1_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_names': self.config['dataset'].get('class_names', []),
            'final_epsilon': epsilon
        }