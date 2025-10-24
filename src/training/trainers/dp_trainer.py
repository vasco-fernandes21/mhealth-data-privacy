#!/usr/bin/env python3
"""
Differential Privacy trainer using Opacus.

Trains models with DP guarantees.
Provides privacy-utility tradeoff analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import time

from src.training.base_trainer import BaseTrainer
from src.training.utils import GradientMonitor
from src.privacy.dp_utils import DPConfig, setup_privacy_engine, get_epsilon, check_dp_compatibility
from src.utils.logging_utils import get_logger


class DPTrainer(BaseTrainer):
    """Trainer with Differential Privacy (Opacus)."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize DP trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
        """
        super().__init__(model, config, device)
        self.logger = get_logger(__name__)
        self.gradient_monitor = None
        
        # DP-specific attributes
        self.privacy_engine = None
        self.dp_config = DPConfig(config)
        self.privacy_budget_history = []
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer and loss function."""
        # Check DP compatibility
        is_compatible, incompatible = check_dp_compatibility(self.model)
        if not is_compatible:
            self.logger.warning(f"Model contains DP-incompatible layers: {incompatible}")
        
        training_cfg = self.config['training']
        
        # Learning rate
        lr = float(training_cfg['learning_rate'])
        weight_decay = float(training_cfg.get('weight_decay', 1e-4))
        
        # ✅ Optimizer (Adam for DP - works better with Opacus)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.logger.info(
            f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})"
        )
        
        # Loss function
        label_smoothing = float(training_cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.criterion = self.criterion.to(self.device)
        
        self.logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")
        
        # Gradient monitor
        self.gradient_monitor = GradientMonitor(self.model)
        self.logger.info("Gradient monitor initialized")
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with DP.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Max epochs
            patience: Early stopping patience
            output_dir: Directory to save checkpoints
        
        Returns:
            Training results dictionary
        """
        from pathlib import Path
        
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss first
        self.setup_optimizer_and_loss()
        
        # Setup Privacy Engine
        self.logger.info("Setting up Differential Privacy Engine...")
        self.model, self.optimizer, self.privacy_engine = setup_privacy_engine(
            self.model,
            self.optimizer,
            train_loader,
            self.dp_config,
            device=str(self.device)
        )
        self.logger.info("✅ Privacy Engine initialized")
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch, train_loader)
            
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
                    f"Early stopping triggered after {epoch} epochs (patience={patience})"
                )
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        if output_path is not None and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
        
        # Get final epsilon
        final_epsilon = get_epsilon(
            self.privacy_engine,
            float(self.dp_config.target_delta)
        )
        
        return {
            'total_epochs': epoch,
            'training_time_seconds': training_time,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history,
            'privacy_budget_history': self.privacy_budget_history,
            'final_epsilon': float(final_epsilon),
            'target_epsilon': self.dp_config.target_epsilon,
            'target_delta': self.dp_config.target_delta
        }
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch with DP.
        
        ✅ IMPORTANT: 
        - No gradient accumulation
        - optimizer.step() called immediately after loss.backward()
        - PrivacyEngine handles clipping and noise injection
        
        Args:
            epoch: Current epoch number
            train_loader: Training data loader
        
        Returns:
            (loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n Starting Epoch {epoch}/{self.config['training'].get('epochs', 100)}")
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # ✅ Step 1: Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # ✅ Step 2: Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # ✅ Step 3: Backward pass (PrivacyEngine hooks compute per-sample gradients)
            loss.backward()
            
            # ✅ Step 4: Log gradients
            if self.gradient_monitor:
                self.gradient_monitor.log_gradients()
            
            # ✅ Step 5: Optimizer step (PrivacyEngine clips and adds noise)
            # IMPORTANT: Must be called immediately after backward()
            # (No gradient accumulation with DP + Poisson sampling)
            self.optimizer.step()
            
            # ✅ Step 6: Compute metrics
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
        }
        
        return metrics
    
    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch with DP metrics."""
        # Get current epsilon
        epsilon = get_epsilon(
            self.privacy_engine,
            float(self.dp_config.target_delta)
        )
        self.privacy_budget_history.append(epsilon)
        
        grad_summary = (
            self.gradient_monitor.get_summary()
            if self.gradient_monitor
            else None
        )
        
        msg = (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"ε={epsilon:.4f}"
        )
        
        if grad_summary:
            msg += f" | grad_norm={grad_summary['norm_mean']:.6f}"
        
        print(f"\n {msg}")
        self.logger.info(msg)
        
        # Warn if budget exceeded
        if epsilon > self.dp_config.target_epsilon:
            self.logger.warning(
                f"⚠️ Privacy budget exceeded! ε={epsilon:.4f} > "
                f"{self.dp_config.target_epsilon:.4f}"
            )