#!/usr/bin/env python3
"""
Baseline trainer without privacy.

Trains models without any privacy protection.
Provides upper bound on accuracy for privacy-utility tradeoff analysis.

Design principles for paper:
- Simple and reproducible
- No class weights (hurt WESAD performance)
- No LR scheduler (fixed LR for fair DP comparison)
- Gradient clipping for DP compatibility
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
from src.training.utils import ProgressBar, GradientMonitor
from src.utils.logging_utils import get_logger


class BaselineTrainer(BaseTrainer):
    """Trainer without privacy (baseline).
    
    Optimized for scientific paper:
    - Standard hyperparameters from literature
    - Fixed learning rate (no scheduling)
    - No class weighting (simple setup)
    - Gradient clipping for DP compatibility
    """
    
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
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        training_cfg = self.config['training']
        
        # ============================================================
        # 1. OPTIMIZER
        # ============================================================
        lr = training_cfg['learning_rate']
        weight_decay = training_cfg.get('weight_decay', 1e-4)
        
        # Fix type conversion issues for numeric values
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        if isinstance(lr, str):
            lr = float(lr)
            
        optimizer_name = training_cfg.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            self.logger.info(
                f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})"
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
        
        # ============================================================
        # 2. LOSS FUNCTION
        # ============================================================
        loss_name = training_cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = training_cfg.get('label_smoothing', 0.0)
        
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
            self.logger.info(
                f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})"
            )
        elif loss_name == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss()
            self.logger.info("Loss: BCEWithLogitsLoss")
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Move criterion to device
        self.criterion = self.criterion.to(self.device)
        
        # ============================================================
        # 3. GRADIENT MONITOR
        # ============================================================
        self.gradient_monitor = GradientMonitor(self.model)
        self.logger.info("Gradient monitor initialized")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            (loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = ProgressBar(len(train_loader), "Training")
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for DP compatibility)
            if self.config['training'].get('gradient_clipping', True):
                clip_norm = self.config['training'].get(
                    'gradient_clip_norm', 1.0
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
            
            pbar.update(1)
        
        pbar.finish()
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with all metrics
        """
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
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(
                precision_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            ),
            'recall': float(
                recall_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            ),
            'f1_score': float(
                f1_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'class_names': self.config['dataset'].get('class_names', [])
        }
        
        return metrics
    
    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch results."""
        grad_summary = (
            self.gradient_monitor.get_summary()
            if self.gradient_monitor
            else None
        )
        
        msg = (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        
        if grad_summary:
            msg += f" | grad_norm={grad_summary['norm_mean']:.6f}"
        
        self.logger.info(msg)