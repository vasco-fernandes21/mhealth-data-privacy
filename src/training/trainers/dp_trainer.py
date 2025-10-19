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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.training.base_trainer import BaseTrainer
from src.training.utils import ProgressBar, GradientMonitor
from src.privacy.dp_utils import DPConfig, setup_privacy_engine, get_epsilon, check_dp_compatibility
from src.utils.logging_utils import get_logger


class DPTrainer(BaseTrainer):
    """Trainer with Differential Privacy."""
    
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
            self.logger.warning("Model contains DP-incompatible layers!")
        
        training_cfg = self.config['training']
        
        # Learning rate
        lr = training_cfg['learning_rate']
        
        # Optimizer (Adam for DP)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=training_cfg.get('weight_decay', 1e-4)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=training_cfg.get('label_smoothing', 0.0)
        )
        self.criterion = self.criterion.to(self.device)
        
        # Gradient monitor
        self.gradient_monitor = GradientMonitor(self.model)
    
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
        # Setup optimizer and loss first
        self.setup_optimizer_and_loss()
        
        # Setup Privacy Engine
        self.logger.info("Setting up Differential Privacy...")
        self.model, self.optimizer, self.privacy_engine = setup_privacy_engine(
            self.model,
            self.optimizer,
            train_loader,
            self.dp_config,
            device=str(self.device)
        )
        
        # Call parent fit method
        results = super().fit(train_loader, val_loader, epochs, patience, output_dir)
        
        # Add privacy metrics
        results['privacy_budget'] = self.privacy_budget_history
        results['final_epsilon'] = get_epsilon(
            self.privacy_engine,
            self.dp_config.target_delta
        )
        results['target_epsilon'] = self.dp_config.target_epsilon
        results['target_delta'] = self.dp_config.target_delta
        
        return results
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch with DP.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            (loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = ProgressBar(len(train_loader), "Training (DP)")
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass (PrivacyEngine handles gradient sampling)
            loss.backward()
            
            # Log gradients
            if self.gradient_monitor:
                self.gradient_monitor.log_gradients()
            
            # Optimizer step (PrivacyEngine handles clipping and noise)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.update(1)
        
        pbar.finish()
        
        # Get current epsilon
        if self.privacy_engine:
            epsilon = get_epsilon(
                self.privacy_engine,
                self.dp_config.target_delta
            )
            self.privacy_budget_history.append(epsilon)
            self.logger.info(f"  Current privacy budget: ε={epsilon:.4f}")
            
            # Check if budget exceeded
            if epsilon > self.dp_config.target_epsilon * 1.1:
                self.logger.warning(
                    f"Privacy budget exceeded! ε={epsilon:.4f} > "
                    f"{self.dp_config.target_epsilon * 1.1:.4f}"
                )
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float) -> None:
        """Log epoch with DP metrics."""
        msg = (f"Epoch {epoch:03d}: "
               f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        
        if self.privacy_engine:
            epsilon = get_epsilon(
                self.privacy_engine,
                self.dp_config.target_delta
            )
            msg += f" | ε={epsilon:.4f}"
        
        self.logger.info(msg)