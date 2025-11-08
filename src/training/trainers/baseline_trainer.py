import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_fscore_support
)

from src.training.base_trainer import BaseTrainer

try:
    from src.losses.focal_loss import FocalLoss
except ImportError:
    FocalLoss = None


class BaselineTrainer(BaseTrainer):
    """Standard training with Focal Loss (optimized)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = None
        self.use_amp = False

    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer, loss, and AMP."""
        cfg = self.config['training']

        # Optimizer
        lr = float(cfg['learning_rate'])
        weight_decay = float(cfg.get('weight_decay', 1e-4))
        opt_name = cfg.get('optimizer', 'adamw').lower()

        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
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

        # Loss function (with optional class weights)
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

        # LR Scheduler
        self._setup_scheduler()

        # Mixed precision (AMP)
        self.use_amp = cfg.get('use_amp', True) and \
                       self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        cfg = self.config['training']
        total_epochs = int(cfg.get('epochs', 100))
        scheduler_name = cfg.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=False
            )
        else:
            self.scheduler = None

    def train_epoch(
        self, train_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Train one epoch with mixed precision.

        Args:
            train_loader: Training data loader

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                # Scaled backward
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clipping', True):
                    clip_norm = float(
                        self.config['training'].get(
                            'gradient_clip_norm', 1.0
                        )
                    )
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                if self.config['training'].get('gradient_clipping', True):
                    clip_norm = float(
                        self.config['training'].get(
                            'gradient_clip_norm', 1.0
                        )
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_norm
                    )

                self.optimizer.step()

            # Accumulate metrics
            batch_loss = loss.detach().item()
            total_loss += batch_loss * batch_x.size(0)

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        # Step scheduler (cosine doesn't need val metrics)
        if self.scheduler is not None and \
           not isinstance(self.scheduler,
                         torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return total_loss / total, correct / total

    def evaluate_full(
        self, test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with detailed metrics
        """
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                _, predicted = torch.max(outputs, 1)

                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)

        # Per-class metrics
        (
            precision_per_class, recall_per_class, f1_per_class, _
        ) = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

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
            'class_names': self.config['dataset'].get('class_names', [])
        }

        return metrics