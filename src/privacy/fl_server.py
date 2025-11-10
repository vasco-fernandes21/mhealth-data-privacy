#!/usr/bin/env python3
"""
Federated Learning Server.

Manages global model and coordinates client training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import copy

from src.privacy.fl_client import FLClient
from src.privacy.fl_aggregators import create_aggregator


class FLServer: 
    """Federated Learning Server - coordinates FL training."""
    
    def __init__(self,
                 model: nn.Module,
                 clients: List[FLClient],
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        """
        Initialize FL server.
        
        Args:
            model: Global model
            clients: List of FL clients
            config: Configuration dictionary
            device: Device to use (cpu/cuda)
        """
        self.model = model.to(device)
        self.clients = clients
        self.config = config
        self.device = torch.device(device)
        
        # FL config
        fl_cfg = config.get('federated_learning', {})
        self.aggregation_method = fl_cfg.get('aggregation_method', 'fedavg')
        
        # Aggregator
        self.aggregator = create_aggregator(self.aggregation_method)
        
        # Loss function
        label_smoothing = float(
            config['training'].get('label_smoothing', 0.0)
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
    
    def broadcast_model(self) -> None:
        """Send global model weights to all clients."""
        global_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        for client in self.clients:
            client.set_weights(global_weights)
    
    def aggregate_updates(
        self, initial_weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        Aggregate client updates and update global model.
        
        Args:
            initial_weights: Global weights before local training
        """
        # Collect updates from all clients
        client_updates = []
        client_weights = []
        
        for client in self.clients:
            updates = client.get_weight_updates(initial_weights)
            client_updates.append(updates)
            client_weights.append(1.0)  # Equal weight for all clients
        
        # Aggregate
        aggregated_updates = self.aggregator.aggregate(
            client_updates, client_weights
        )
        
        # Apply to global model: w_new = w_old + delta_w
        for name, param in self.model.named_parameters():
            if name in aggregated_updates:
                param.data = (
                    initial_weights[name] + aggregated_updates[name]
                )
    
    def evaluate_on_clients(
        self, val_loaders: List
    ) -> Dict[str, float]:
        """
        Evaluate global model on all clients' validation data.
        
        Args:
            val_loaders: List of validation loaders (one per client)
        
        Returns:
            Average metrics across clients
        """
        accuracies = []
        losses = []
        
        self.model.eval()
        with torch.no_grad():
            for val_loader in val_loaders:
                correct = 0
                total = 0
                batch_loss = 0.0
                
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    
                    batch_loss += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_x.size(0)
                
                acc = correct / total if total > 0 else 0.0
                loss = batch_loss / total if total > 0 else 0.0
                
                accuracies.append(acc)
                losses.append(loss)
        
        return {
            'accuracy': (
                sum(accuracies) / len(accuracies) if accuracies else 0.0
            ),
            'loss': sum(losses) / len(losses) if losses else 0.0,
        }
    
    def train_round(self) -> Dict[str, float]:
        """
        Execute one federated learning round.
        
        Returns:
            Metrics for this round
        """
        # Save initial weights
        initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Broadcast and train locally
        self.broadcast_model()
        
        client_metrics = []
        for client in self.clients:
            metrics = client.train_local()
            client_metrics.append(metrics)
        
        # Aggregate updates
        self.aggregate_updates(initial_weights)
        
        # Average metrics
        avg_loss = (
            sum(m['loss'] for m in client_metrics) / len(client_metrics)
        )
        avg_acc = (
            sum(m['accuracy'] for m in client_metrics) / len(client_metrics)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
        }