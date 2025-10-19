#!/usr/bin/env python3
"""
Federated Learning Server.

Manages global model and coordinates client training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional
import copy

from src.privacy.fl_client import FLClient
from src.privacy.fl_aggregators import create_aggregator
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FLServer:
    """Federated Learning Server."""
    
    def __init__(self,
                 model: nn.Module,
                 clients: List[FLClient],
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize FL server.
        
        Args:
            model: Global model
            clients: List of FL clients
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.clients = clients
        self.config = config
        self.device = torch.device(device)
        
        # FL config
        fl_cfg = config.get('federated_learning', {})
        self.global_rounds = fl_cfg.get('global_rounds', 100)
        self.aggregation_method = fl_cfg.get('aggregation_method', 'fedavg')
        
        # Aggregator
        self.aggregator = create_aggregator(self.aggregation_method)
        
        # History
        self.round_history = {
            'global_round': [],
            'avg_client_accuracy': [],
            'avg_client_loss': [],
            'aggregation_time': []
        }
    
    def broadcast_model(self) -> None:
        """Send global model to all clients."""
        global_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        for client in self.clients:
            client.set_weights(global_weights)
    
    def aggregate_updates(self) -> None:
        """Aggregate client updates and update global model."""
        # Get initial weights before aggregation
        initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Collect updates from all clients
        client_updates = []
        client_sizes = []
        
        for client in self.clients:
            updates = client.get_weight_updates(initial_weights)
            client_updates.append(updates)
            
            # Use 1.0 as weight for now (equal weighting)
            # Could use dataset size for weighted averaging
            client_sizes.append(1.0)
        
        # Aggregate updates
        aggregated_updates = self.aggregator.aggregate(client_updates, client_sizes)
        
        # Apply aggregated updates to global model
        for name, param in self.model.named_parameters():
            if name in aggregated_updates:
                param.data = param.data + aggregated_updates[name]
    
    def evaluate_on_clients(self, val_loaders: List) -> Dict[str, float]:
        """
        Evaluate global model on all clients' validation data.
        
        Args:
            val_loaders: List of validation loaders for each client
        
        Returns:
            Average metrics
        """
        accuracies = []
        losses = []
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for client_id, val_loader in enumerate(val_loaders):
                correct = 0
                total = 0
                batch_loss = 0
                
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    batch_loss += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                
                accuracy = correct / total if total > 0 else 0
                loss_val = batch_loss / total if total > 0 else 0
                
                accuracies.append(accuracy)
                losses.append(loss_val)
        
        return {
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'min_accuracy': min(accuracies) if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0
        }
    
    def train_round(self) -> Dict[str, Any]:
        """
        Execute one federated learning round.
        
        Returns:
            Metrics for this round
        """
        # Broadcast model
        self.broadcast_model()
        
        # Train on clients
        client_metrics = []
        for client in self.clients:
            metrics = client.train_local()
            client_metrics.append(metrics)
        
        # Aggregate updates
        self.aggregate_updates()
        
        # Compute averages
        avg_loss = sum(m['local_loss'] for m in client_metrics) / len(client_metrics)
        avg_acc = sum(m['local_accuracy'] for m in client_metrics) / len(client_metrics)
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'n_clients': len(self.clients)
        }
    
    def __repr__(self) -> str:
        return (f"FLServer(n_clients={len(self.clients)}, "
                f"aggregation={self.aggregation_method})")