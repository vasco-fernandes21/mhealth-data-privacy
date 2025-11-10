#!/usr/bin/env python3
"""
Federated Learning Client.

Represents a single client in federated learning.
Trains locally and sends updates to server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import copy


class FLClient:
    """Federated Learning Client."""
    
    def __init__(self,
                 client_id: str,
                 model: nn.Module,
                 train_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        """
        Initialize FL client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Local training data loader
            config: Configuration dictionary
            device: Device to use (cpu/cuda)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)
        
        # FL config
        fl_cfg = config.get('federated_learning', {})
        self.local_epochs = fl_cfg.get('local_epochs', 1)
        
        # Setup optimizer, scheduler, criterion
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
    
    def setup_optimizer(self) -> None:
        """Setup optimizer based on config."""
        optimizer_name = (
            self.config['training'].get('optimizer', 'adamw').lower()
        )
        lr = self.config['training'].get('learning_rate', 0.001)
        wd = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_name = (
            self.config['training'].get('scheduler', 'none').lower()
        )
        
        if scheduler_name == 'none':
            self.scheduler = None
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        elif scheduler_name == 'cosine':
            epochs = self.config['training'].get('epochs', 40)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            self.scheduler = None
    
    def setup_criterion(self) -> None:
        """Setup loss criterion."""
        label_smoothing = float(
            self.config['training'].get('label_smoothing', 0.0)
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def train_local(self) -> Dict[str, float]:
        """
        Train model locally for local_epochs.
        
        Returns:
            Dictionary with local training metrics
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for _ in range(self.local_epochs):
            self.model.train()
            
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clipping', True):
                    clip_norm = self.config['training'].get(
                        'gradient_clip_norm', 1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_norm
                    )
                
                # Update
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)
        
        # Scheduler step (once per round, not per batch)
        if self.scheduler is not None and isinstance(
            self.scheduler, 
            torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            self.scheduler.step()
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        }
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights from dictionary."""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone().to(self.device)
    
    def get_weight_updates(
        self, initial_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get weight updates (delta_w = w_current - w_initial)."""
        current = self.get_weights()
        return {
            name: current[name] - initial_weights[name]
            for name in current.keys()
        }