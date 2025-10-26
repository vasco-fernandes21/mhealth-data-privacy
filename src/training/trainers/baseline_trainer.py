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
from src.utils.logging_utils import get_logger

try:
    from src.losses.focal_loss import FocalLoss
except ImportError:
    FocalLoss = None


class BaselineTrainer(BaseTrainer):
    """Standard training without privacy."""
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss."""
        cfg = self.config['training']
        
        # Optimizer
        lr = float(cfg['learning_rate'])
        weight_decay = float(cfg.get('weight_decay', 1e-4))
        opt_name = cfg.get('optimizer', 'adamw').lower()
        
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr,
                weight_decay=weight_decay, momentum=0.9
            )
        
        # Loss
        loss_name = cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        
        if loss_name == 'focal_loss' and FocalLoss:
            self.criterion = FocalLoss(
                alpha=float(cfg.get('focal_alpha', 0.25)),
                gamma=float(cfg.get('focal_gamma', 2.0))
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
        
        self.criterion = self.criterion.to(self.device)
        
        # LR Scheduler
        self._setup_scheduler()
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        cfg = self.config['training']
        scheduler_type = cfg.get('lr_scheduler', 'none').lower()
        
        if scheduler_type == 'warmup_cosine':
            from src.training.schedulers import WarmupCosineAnnealingScheduler
            self.scheduler = WarmupCosineAnnealingScheduler(
                self.optimizer,
                warmup_epochs=int(cfg.get('warmup_epochs', 5)),
                total_epochs=int(cfg.get('epochs', 100)),
                eta_min=1e-6
            )
        elif scheduler_type == 'cosine_annealing_warm_restarts':
            from src.training.schedulers import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(cfg.get('scheduler_T0', 10)),
                T_mult=int(cfg.get('scheduler_Tmult', 2)),
                eta_min=1e-6
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Forward
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clipping', True):
                clip_norm = float(
                    self.config['training'].get('gradient_clip_norm', 1.0)
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_norm
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / total, correct / total
    
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, confusion matrix
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
            'class_names': self.config['dataset'].get('class_names', [])
        }
        
        return metrics