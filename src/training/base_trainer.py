#!/usr/bin/env python3
"""
Baseline trainer without privacy.

Trains models without any privacy protection.
Provides upper bound on accuracy for privacy-utility tradeoff analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from src.training.base_trainer import BaseTrainer
from src.training.utils import ProgressBar, GradientMonitor
from src.utils.logging_utils import get_logger


class BaselineTrainer(BaseTrainer):
    """Trainer without privacy (baseline)."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 y_train: Optional[np.ndarray] = None):
        """
        Initialize baseline trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
            y_train: Training labels (for computing class weights)
        """
        super().__init__(model, config, device)
        self.logger = get_logger(__name__)
        self.gradient_monitor = None
        self.lr_scheduler = None
        self.y_train = y_train
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        training_cfg = self.config['training']
        
        # ============================================================
        # 1. OPTIMIZER
        # ============================================================
        lr = training_cfg['learning_rate']
        weight_decay = training_cfg.get('weight_decay', 1e-4)
        optimizer_name = training_cfg.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            self.logger.info(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
        
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
            self.logger.info(f"Optimizer: SGD (lr={lr}, weight_decay={weight_decay})")
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # ============================================================
        # 2. LOSS FUNCTION WITH CLASS WEIGHTS
        # ============================================================
        loss_name = training_cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = training_cfg.get('label_smoothing', 0.0)
        
        if loss_name == 'cross_entropy':
            # Compute class weights if needed
            weights = None
            if training_cfg.get('use_class_weights', False):
                if self.y_train is not None:
                    class_weights = compute_class_weight(
                        'balanced',
                        classes=np.unique(self.y_train),
                        y=self.y_train
                    )
                    weights = torch.tensor(class_weights, dtype=torch.float32)
                    weights = weights.to(self.device)
                    self.logger.info(f"Class weights: {weights.cpu().numpy()}")
                else:
                    self.logger.warning("use_class_weights=True but y_train not provided")
            
            self.criterion = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing
            )
            self.logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")
        
        elif loss_name == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss()
            self.logger.info("Loss: BCEWithLogitsLoss")
        
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Move criterion to device
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(self.device)
        
        # ============================================================
        # 3. LEARNING RATE SCHEDULER
        # ============================================================
        scheduler_type = training_cfg.get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=6,
                min_lr=1e-6,
                verbose=False
            )
            self.logger.info("LR Scheduler: ReduceLROnPlateau")
        
        elif scheduler_type == 'exponential':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
            self.logger.info("LR Scheduler: ExponentialLR")
        
        elif scheduler_type == 'none':
            self.lr_scheduler = None
            self.logger.info("LR Scheduler: None")
        
        # ============================================================
        # 4. GRADIENT MONITOR
        # ============================================================
        self.gradient_monitor = GradientMonitor(self.model)
    
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
            
            # ✅ Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # ✅ Backward pass
            loss.backward()
            
            # ✅ Gradient clipping
            if self.config['training'].get('gradient_clipping', True):
                clip_norm = self.config['training'].get('gradient_clip_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            # ✅ Log gradients
            if self.gradient_monitor:
                self.gradient_monitor.log_gradients()
            
            # ✅ Optimizer step
            self.optimizer.step()
            
            # ✅ Metrics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.update(1)
        
        pbar.finish()
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with LR scheduler.
        
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
            
            # ✅ UPDATE LEARNING RATE (before early stopping check)
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_acc)
                else:
                    self.lr_scheduler.step()
            
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
                self.logger.info(f"Early stopping triggered after {epoch} epochs (patience={patience})")
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
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'class_names': self.config['dataset'].get('class_names', [])
        }
        
        return metrics
    
    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch results with gradient info."""
        grad_summary = self.gradient_monitor.get_summary() if self.gradient_monitor else None
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        msg = (f"Epoch {epoch:03d}: "
               f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
               f"lr={current_lr:.2e}")
        
        if grad_summary:
            msg += f" | grad_norm={grad_summary['norm_mean']:.6f}"
        
        self.logger.info(msg)