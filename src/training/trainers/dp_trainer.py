#!/usr/bin/env python3
"""Differential Privacy training with Opacus."""

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
    """Differential Privacy configuration."""

    def __init__(self, config: Dict[str, Any]):
        dp_cfg = config.get('differential_privacy', {})
        self.enabled = bool(dp_cfg.get('enabled', False))
        self.noise_multiplier = float(dp_cfg.get('noise_multiplier', 0.9))
        self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        self.delta = float(dp_cfg.get('delta', 1e-5))
        self.poisson_sampling = bool(dp_cfg.get('poisson_sampling', True))
        self.grad_sample_mode = str(dp_cfg.get('grad_sample_mode', 'hooks'))


class DPTrainer(BaseTrainer):
    """Differential Privacy trainer with Opacus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = DPConfig(self.config)
        self.privacy_engine = None
        self.dp_train_loader = None
        self.best_model_state = None

    def setup_optimizer_and_loss(self) -> None:
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
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            momentum = float(cfg.get('momentum', 0.7))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )

        dataset_cfg = self.config.get('dataset', {})
        use_class_weights = dataset_cfg.get('use_class_weights', False)
        label_smoothing = float(cfg.get('label_smoothing', 0.0))

        class_weights = None
        if use_class_weights and 'class_weights' in dataset_cfg:
            weights_dict = dataset_cfg['class_weights']
            n_classes = dataset_cfg.get('n_classes', 2)
            class_weights = torch.zeros(n_classes, dtype=torch.float32)
            for class_idx, weight in weights_dict.items():
                class_weights[int(class_idx)] = float(weight)
            class_weights = class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction='mean'
        ).to(self.device)

        self._setup_scheduler()

    def _setup_scheduler(self) -> None:
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
            return train_loader

        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.dp_train_loader = (
            privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=self.dp_config.noise_multiplier,
                max_grad_norm=self.dp_config.max_grad_norm,
                poisson_sampling=self.dp_config.poisson_sampling,
                grad_sample_mode=self.dp_config.grad_sample_mode,
            )
        )
        self.privacy_engine = privacy_engine
        return self.dp_train_loader

    def _get_epsilon(self) -> Optional[float]:
        """Get current epsilon."""
        if self.privacy_engine is None:
            return None

        try:
            epsilon = float(
                self.privacy_engine.get_epsilon(self.dp_config.delta)
            )
            if isinstance(epsilon, (int, float)) and 0 <= epsilon < float('inf'):
                return epsilon
        except Exception:
            pass

        return None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loader = self.dp_train_loader if self.dp_train_loader else train_loader

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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 8,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train with DP."""
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        self._reset_training_state()
        self.setup_optimizer_and_loss()
        self.setup_privacy_engine(train_loader)

        self._print_header()

        start_time = time.time()
        best_epoch = 0
        epoch_num = 0
        epsilon_history = []

        for epoch_num in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            epsilon = self._get_epsilon()

            if epsilon is not None:
                epsilon_history.append(epsilon)

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

                self.best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                if output_path:
                    self.save_checkpoint(output_path / 'best_model.pth')
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

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        elif output_path and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')

        self._print_summary(epoch_num, best_epoch, elapsed, epsilon_history)

        return {
            'total_epochs': epoch_num,
            'best_epoch': best_epoch,
            'epochs_no_improve': self.epochs_no_improve,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'final_epsilon': epsilon_history[-1] if epsilon_history else None,
            'epsilon_history': epsilon_history,
        }

    def _print_header(self) -> None:
        print(f"\nDP Training")
        print(f"  Noise multiplier: {self.dp_config.noise_multiplier}")
        print(f"  Max grad norm: {self.dp_config.max_grad_norm}")
        print(f"  Delta: {self.dp_config.delta}\n")

    def _print_summary(
        self,
        total_epochs: int,
        best_epoch: int,
        elapsed: float,
        epsilon_history: list,
    ) -> None:
        print(f"\nCompleted: {total_epochs} epochs in {elapsed:.1f}s")
        print(f"  Best: epoch {best_epoch} (acc={self.best_val_acc:.4f})")
        if epsilon_history:
            print(f"  Final epsilon: {epsilon_history[-1]:.4f}\n")

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation."""
        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)

                y_true_list.append(batch_y.cpu())
                y_pred_list.append(predicted.cpu())

        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
        unique_labels = np.unique(y_true)

        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

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
        }