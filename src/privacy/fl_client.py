#!/usr/bin/env python3
"""
Federated Learning Client.

Represents a single client (subject) in federated learning.
Trains locally and sends updates to server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Optional
import copy

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FLClient:
    """Federated Learning Client."""
    
    def __init__(self,
                 client_id: str,
                 model: nn.Module,
                 train_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize FL client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Local training data loader
            config: Configuration dictionary
            device: Device to use
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)
        
        # Training config
        fl_cfg = config.get('federated_learning', {})
        self.local_epochs = fl_cfg.get('local_epochs', 5)
        self.learning_rate = config['training'].get('learning_rate', 0.001)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Stats
        self.local_steps = 0
        self.update_count = 0
    
    def train_local(self) -> Dict[str, float]:
        """
        Train model locally for local_epochs.
        
        Returns:
            Dictionary with training metrics
        """
        metrics = {
            'local_loss': 0.0,
            'local_accuracy': 0.0,
            'local_samples': 0,
            'local_epochs': self.local_epochs
        }
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
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
                    clip_norm = self.config['training'].get('gradient_clip_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                
                # Update
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == batch_y).sum().item()
                epoch_total += batch_y.size(0)
                
                self.local_steps += 1
            
            epoch_loss /= epoch_total
            epoch_acc = epoch_correct / epoch_total
            
            metrics['local_loss'] = epoch_loss
            metrics['local_accuracy'] = epoch_acc
            metrics['local_samples'] = epoch_total
        
        self.update_count += 1
        return metrics
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get model weights.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Set model weights.
        
        Args:
            weights: Dictionary mapping parameter names to tensors
        """
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone().to(self.device)
    
    def get_weight_updates(self, initial_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get weight updates (current - initial).
        
        Args:
            initial_weights: Initial weights before local training
        
        Returns:
            Dictionary of weight updates
        """
        current_weights = self.get_weights()
        updates = {}
        
        for name, current in current_weights.items():
            initial = initial_weights[name]
            updates[name] = current - initial
        
        return updates
    
    def __repr__(self) -> str:
        return f"FLClient(id={self.client_id}, local_epochs={self.local_epochs})"