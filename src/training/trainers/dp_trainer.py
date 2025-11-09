#!/usr/bin/env python3
"""
DP Trainer - Version with fair-comparison optimizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
import time
from pathlib import Path
from copy import deepcopy

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

from opacus import PrivacyEngine
from src.training.base_trainer import BaseTrainer
from src.utils.logging_utils import get_logger

try:
    from src.losses.focal_loss import FocalLoss
except ImportError:
    FocalLoss = None

logger = get_logger(__name__)


class DPConfig:
    """Configuration for Differential Privacy training."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize DP config with type coercion."""
        dp_cfg = config.get('differential_privacy', {})

        self.enabled = bool(dp_cfg.get('enabled', False))

        try:
            self.noise_multiplier = float(
                dp_cfg.get('noise_multiplier', 0.9)
            )
        except (TypeError, ValueError):
            self.noise_multiplier = 0.9

        try:
            self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        except (TypeError, ValueError):
            self.max_grad_norm = 1.0

        try:
            self.delta = float(dp_cfg.get('delta', 1e-5))
        except (TypeError, ValueError):
            self.delta = 1e-5

        self.poisson_sampling = bool(
            dp_cfg.get('poisson_sampling', True)
        )
        self.accounting_method = str(
            dp_cfg.get('accounting_method', 'rdp')
        )
        self.grad_sample_mode = str(
            dp_cfg.get('grad_sample_mode', 'hooks')
        )

    def __repr__(self) -> str:
        return (
            f"DPConfig(noise_mult={self.noise_multiplier}, "
            f"max_grad={self.max_grad_norm}, delta={self.delta})"
        )


def check_dp_compatibility(model: nn.Module) -> Tuple[bool, list]:
    """Check if model is DP-compatible with Opacus."""
    incompatible_layers = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                              nn.BatchNorm3d)):
            incompatible_layers.append((name, type(module).__name__))

    is_compatible = len(incompatible_layers) == 0

    if not is_compatible:
        logger.warning(
            f"Model has {len(incompatible_layers)} incompatible layers:"
        )
        for name, layer_type in incompatible_layers:
            logger.warning(f"  - {name}: {layer_type}")
        logger.warning("Replace BatchNorm with GroupNorm or LayerNorm")
    else:
        logger.info("Model is DP-compatible")

    return is_compatible, incompatible_layers


class DPTrainer(BaseTrainer):
    """Differential Privacy Trainer using Opacus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = DPConfig(self.config)
        self.privacy_engine = None
        self.dp_train_loader = None
        
        # Otimizações de I/O transparentes
        torch.backends.cudnn.benchmark = True

    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer, loss, and scheduler."""
        cfg = self.config['training']

        lr = float(cfg.get('learning_rate', 1e-3))
        weight_decay = float(cfg.get('weight_decay', 1e-4))
        opt_name = cfg.get('optimizer', 'adamw').lower()

        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )

        loss_name = cfg.get('loss', 'cross_entropy').lower()
        class_weights = cfg.get('class_weights', None)

        if class_weights is not None:
            weights_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
        else:
            weights_tensor = None

        if loss_name == 'focal_loss' and FocalLoss:
            self.criterion = FocalLoss(
                alpha=float(cfg.get('focal_alpha', 0.25)),
                gamma=float(cfg.get('focal_gamma', 2.0))
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_tensor, reduction='mean'
            )

        self.criterion = self.criterion.to(self.device)

        self._setup_scheduler(cfg)

    def _setup_scheduler(self, cfg: Dict[str, Any]) -> None:
        """Setup learning rate scheduler."""
        epochs = int(cfg.get('epochs', 20))
        scheduler_name = cfg.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            self.scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs,
                    eta_min=1e-6
                )
            )
        elif scheduler_name == 'plateau':
            self.scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=False
                )
            )
        else:
            self.scheduler = None

    def setup_privacy_engine(self, train_loader: DataLoader) -> DataLoader:
        """Setup Opacus PrivacyEngine for DP training."""
        if not self.dp_config.enabled:
            logger.info("DP disabled - using standard training")
            return train_loader

        logger.info(f"Setting up PrivacyEngine: {self.dp_config}")

        is_compatible, _ = check_dp_compatibility(self.model)
        if not is_compatible:
            logger.warning("Model may not be fully DP-compatible")

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

            logger.info(
                f"PrivacyEngine configured:\n"
                f"  noise_multiplier: {self.dp_config.noise_multiplier}\n"
                f"  max_grad_norm: {self.dp_config.max_grad_norm}\n"
                f"  delta: {self.dp_config.delta}"
            )

            return self.dp_train_loader

        except Exception as e:
            logger.error(f"PrivacyEngine setup failed: {e}")
            raise

    def _get_epsilon(self) -> float:
        """Get current privacy budget (epsilon)."""
        if self.privacy_engine is None:
            return float('inf')

        try:
            epsilon = self.privacy_engine.get_epsilon(
                self.dp_config.delta
            )
            return float(epsilon)
        except Exception:
            return float('inf')

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loader = (self.dp_train_loader if self.dp_train_loader
                 else train_loader)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item() * batch_x.size(0)

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        self.cleanup_memory()

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
        """
        Main training loop with DP.

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
        self.setup_privacy_engine(train_loader)

        validation_frequency = max(1, epochs // 5)
        start_time = time.time()

        print(f"\nDP Training")
        print(f"  Noise multiplier: {self.dp_config.noise_multiplier}")
        print(f"  Max grad norm: {self.dp_config.max_grad_norm}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Validation frequency: every {validation_frequency} epochs")
        print(f"  Early stopping patience: {patience}\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)

            should_validate = (
                (epoch % validation_frequency == 0) or (epoch == epochs)
            )

            if should_validate:
                val_loss, val_acc = self.validate(val_loader)
                epsilon = self._get_epsilon()

                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                eps_str = (f"{epsilon:.4f}"
                          if epsilon != float('inf') else "N/A")

                print(
                    f"Epoch {epoch:2d}: "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"eps={eps_str}",
                    flush=True
                )

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    self.best_model_state = deepcopy(
                        self.model.state_dict()
                    )

                    if output_path is not None:
                        self.save_checkpoint(
                            output_path / 'best_model.pth'
                        )
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= patience:
                        print(
                            f"\nEarly stopping at epoch {epoch} "
                            f"(no improvement for {patience} checks)",
                            flush=True
                        )
                        break

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_acc)

        elapsed = time.time() - start_time
        final_epsilon = self._get_epsilon()

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        elif (output_path is not None and
              (output_path / 'best_model.pth').exists()):
            self.load_checkpoint(output_path / 'best_model.pth')

        print(f"\n{'='*60}")
        print(f"Training completed: {epoch} epochs in {elapsed:.1f}s")
        print(f"  Best val acc: {self.best_val_acc:.4f}")
        if final_epsilon != float('inf'):
            print(f"  Final epsilon: {final_epsilon:.4f}")
        print(f"{'='*60}\n")

        return {
            'total_epochs': epoch,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history,
            'final_epsilon': final_epsilon if final_epsilon != float('inf') 
                            else None
        }

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation with all metrics."""
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
        final_epsilon = self._get_epsilon()

        metrics = {
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
            'class_names': (
                self.config['dataset'].get('class_names', [])
            ),
            'final_epsilon': final_epsilon if final_epsilon != float('inf') 
                            else None
        }

        return metrics


if __name__ == "__main__":
    print("DP Trainer utility module")
    print("Install Opacus with: pip install opacus")