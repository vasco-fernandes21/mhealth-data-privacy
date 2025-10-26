#!/usr/bin/env python3
"""
Baseline trainer with FocalLoss, AdamW, and CosineAnnealing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

from src.training.base_trainer import BaseTrainer
from src.training.utils import GradientMonitor
from src.training.schedulers import (
    CosineAnnealingWarmRestarts, WarmupCosineAnnealingScheduler
)
try:
    from src.losses.focal_loss import FocalLoss
except ImportError:
    FocalLoss = None
from src.utils.logging_utils import get_logger


class BaselineTrainer(BaseTrainer):
    """Baseline trainer with improved optimization."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize baseline trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
        """
        super().__init__(model, config, device)
        self.logger = get_logger(__name__)
        self.gradient_monitor = None
        self.lr_scheduler = None
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        training_cfg = self.config['training']
        
        # Optimizer
        lr = float(training_cfg['learning_rate'])
        weight_decay = float(training_cfg.get('weight_decay', 1e-4))
        optimizer_name = training_cfg.get('optimizer', 'adamw').lower()
        
        if optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            self.logger.info(
                f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})"
            )
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
            self.logger.info(
                f"Optimizer: SGD (lr={lr}, weight_decay={weight_decay})"
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        loss_name = training_cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = float(
            training_cfg.get('label_smoothing', 0.0)
        )
        
        if loss_name == 'focal_loss' and FocalLoss is not None:
            focal_alpha = float(
                training_cfg.get('focal_alpha', 0.25)
            )
            focal_gamma = float(
                training_cfg.get('focal_gamma', 2.0)
            )
            
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean'
            )
            self.logger.info(
                f"Loss: FocalLoss (alpha={focal_alpha}, "
                f"gamma={focal_gamma})"
            )
        elif loss_name == 'focal_loss' and FocalLoss is None:
            self.logger.warning(
                "FocalLoss not available, using CrossEntropyLoss"
            )
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
            self.logger.info(
                f"Loss: CrossEntropyLoss "
                f"(label_smoothing={label_smoothing})"
            )
        elif loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
            self.logger.info(
                f"Loss: CrossEntropyLoss "
                f"(label_smoothing={label_smoothing})"
            )
        elif loss_name == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss()
            self.logger.info("Loss: BCEWithLogitsLoss")
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        self.criterion = self.criterion.to(self.device)
        
        # Learning rate scheduler
        scheduler_type = training_cfg.get(
            'lr_scheduler', 'none'
        ).lower()
        total_epochs = int(training_cfg.get('epochs', 100))
        
        if scheduler_type == 'cosine_annealing_warm_restarts':
            T_0 = int(training_cfg.get('scheduler_T0', 10))
            T_mult = int(training_cfg.get('scheduler_Tmult', 2))
            
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=1e-6
            )
            self.logger.info(
                f"LR Scheduler: CosineAnnealingWarmRestarts "
                f"(T0={T_0}, Tmult={T_mult})"
            )
        elif scheduler_type == 'warmup_cosine':
            warmup_epochs = int(
                training_cfg.get('warmup_epochs', 5)
            )
            
            self.lr_scheduler = WarmupCosineAnnealingScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                eta_min=1e-6
            )
            self.logger.info(
                f"LR Scheduler: WarmupCosineAnnealing "
                f"(warmup={warmup_epochs})"
            )
        elif scheduler_type == 'none':
            self.lr_scheduler = None
            self.logger.info("LR Scheduler: None (fixed LR)")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        # Gradient monitor
        self.gradient_monitor = GradientMonitor(self.model)
        self.logger.info("Gradient monitor initialized")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clipping', True):
                clip_norm = float(
                    self.config['training'].get(
                        'gradient_clip_norm', 1.0
                    )
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_norm
                )
            
            # Log gradients
            if self.gradient_monitor:
                self.gradient_monitor.log_gradients()
            
            # Optimizer step
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation with all metrics."""
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get unique labels
        unique_labels = np.unique(y_true)
        
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
            'confusion_matrix': confusion_matrix(
                y_true, y_pred, labels=unique_labels
            ).tolist(),
            'class_names': self.config['dataset'].get(
                'class_names', []
            )
        }
        
        return metrics
    
    def _log_epoch(self, epoch: int, train_loss: float,
                   train_acc: float, val_loss: float,
                   val_acc: float) -> None:
        """Log epoch results."""
        grad_summary = (
            self.gradient_monitor.get_summary()
            if self.gradient_monitor
            else None
        )
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        msg = (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        
        if grad_summary:
            msg += (
                f" | grad_norm={grad_summary['norm_mean']:.6f}"
            )
        
        print(f"\n {msg}")
        self.logger.info(msg)