#!/usr/bin/env python3
"""
Baseline Trainer - Standard training with SGD/Adam
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

from src.training.base_trainer import BaseTrainer


class BaselineTrainer(BaseTrainer):
    """Standard training."""

    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss."""
        cfg = self.config['training']

        lr = float(cfg.get('learning_rate', 1e-4))
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
        else:  # SGD (default, best for features)
            momentum = float(cfg.get('momentum', 0.9))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
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
                )
            )
        else:
            self.scheduler = None

    def train_epoch(
        self, train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            loss.backward()

            # Gradient clipping (optional)
            if self.config['training'].get('gradient_clipping', False):
                clip_norm = float(
                    self.config['training'].get(
                        'gradient_clip_norm', 0.5
                    )
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_norm
                )

            self.optimizer.step()

            total_loss += loss.detach().item() * batch_x.size(0)

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        if (self.scheduler is not None and
                not isinstance(self.scheduler,
                         torch.optim.lr_scheduler.ReduceLROnPlateau)):
            self.scheduler.step()

        return total_loss / total, correct / total

    def evaluate_full(
        self, test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Full evaluation with all metrics."""
        # Diagnostics: test set size (if available)
        try:
            total_expected = len(getattr(test_loader, 'dataset', []))
            print(f"[EVAL] Test set size (expected): {total_expected}")
        except Exception:
            pass
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(batch_y.detach().cpu().numpy())
                y_pred.extend(predicted.detach().cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)

        # Diagnostics: distributions
        try:
            true_counts = np.bincount(y_true.astype(int))
            pred_counts = np.bincount(y_pred.astype(int))
            print(f"[EVAL] y_true dist: {true_counts.tolist()}")
            print(f"[EVAL] y_pred dist: {pred_counts.tolist()}")
        except Exception:
            pass

        (precision_per_class, recall_per_class, f1_per_class, _) = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        # Diagnostics: confusion matrix
        try:
            print(f"[EVAL] Confusion matrix:\n{cm}")
        except Exception:
            pass

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
            'class_names': (
                self.config['dataset'].get('class_names', [])
            )
        }
        