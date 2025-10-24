#!/usr/bin/env python3
"""
Baseline trainer with FocalLoss, AdamW, and CosineAnnealing.

Minimal changes for maximum compatibility with DP/FL.
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
from src.training.schedulers import CosineAnnealingWarmRestarts, WarmupCosineAnnealingScheduler
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
        
        # ============================================================
        # 1. OPTIMIZER (AdamW instead of Adam)
        # ============================================================
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
        
        # ============================================================
        # 2. LOSS FUNCTION (FocalLoss for class imbalance)
        # ============================================================
        loss_name = training_cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = float(training_cfg.get('label_smoothing', 0.0))
        
        if loss_name == 'focal_loss' and FocalLoss is not None:
            focal_alpha = float(training_cfg.get('focal_alpha', 0.25))
            focal_gamma = float(training_cfg.get('focal_gamma', 2.0))
            
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean'
            )
            self.logger.info(
                f"Loss: FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})"
            )
        elif loss_name == 'focal_loss' and FocalLoss is None:
            self.logger.warning("FocalLoss not available, falling back to CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
            self.logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")
        elif loss_name == 'cross_entropy':
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
        
        self.criterion = self.criterion.to(self.device)
        
        # ============================================================
        # 3. LEARNING RATE SCHEDULER
        # ============================================================
        scheduler_type = training_cfg.get('lr_scheduler', 'none').lower()
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
                f"LR Scheduler: CosineAnnealingWarmRestarts (T0={T_0}, Tmult={T_mult})"
            )
        elif scheduler_type == 'warmup_cosine':
            warmup_epochs = int(training_cfg.get('warmup_epochs', 5))
            
            self.lr_scheduler = WarmupCosineAnnealingScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                eta_min=1e-6
            )
            self.logger.info(
                f"LR Scheduler: WarmupCosineAnnealing (warmup={warmup_epochs})"
            )
        elif scheduler_type == 'none':
            self.lr_scheduler = None
            self.logger.info("LR Scheduler: None (fixed LR)")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        # ============================================================
        # 4. GRADIENT MONITOR
        # ============================================================
        self.gradient_monitor = GradientMonitor(self.model)
        self.logger.info("Gradient monitor initialized")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
                clip_norm = float(self.config['training'].get('gradient_clip_norm', 1.0))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
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
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 10,
            output_dir: str = None) -> Dict[str, Any]:
        """Main training loop."""
        from pathlib import Path
        import time
        
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss
        self.setup_optimizer_and_loss()
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n Starting Epoch {epoch}/{epochs}")
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
            
            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch - 1)
            
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
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})"
                )
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
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        msg = (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        
        if grad_summary:
            msg += f" | grad_norm={grad_summary['norm_mean']:.6f}"
        
        # Print to console for real-time visibility
        print(f"\n {msg}")
        
        # Also log to file
        self.logger.info(msg)