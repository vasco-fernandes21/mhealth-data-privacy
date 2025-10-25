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
    
    def aggregate_updates(self,
                         initial_weights: Dict[str, torch.Tensor]) -> None:
        """
        Aggregate client updates using initial weights.
        
        Args:
            initial_weights: Global weights BEFORE local training
        """
        # Collect updates from all clients
        client_updates = []
        client_sizes = []
        
        for client in self.clients:
            # Calculate updates: trained_weights - initial_weights
            updates = client.get_weight_updates(initial_weights)
            client_updates.append(updates)
            client_sizes.append(1.0)
        
        # Aggregate updates
        aggregated_updates = self.aggregator.aggregate(
            client_updates,
            client_sizes
        )
        
        # Apply aggregated updates to global model
        # w_global = w_initial + aggregated_updates
        for name, param in self.model.named_parameters():
            if name in aggregated_updates:
                param.data = (
                    initial_weights[name] + aggregated_updates[name]
                )
    
    def evaluate_on_clients(self,
                            val_loaders: List) -> Dict[str, float]:
        """
        Evaluate global model on all clients.
        
        Args:
            val_loaders: List of validation loaders (one per client)
        
        Returns:
            Average metrics across clients
        """
        accuracies = []
        losses = []
        
        # Setup loss function - use same as training!
        training_cfg = self.config['training']
        loss_name = training_cfg.get('loss', 'cross_entropy').lower()
        label_smoothing = float(training_cfg.get('label_smoothing', 0.0))
        
        if loss_name == 'focal_loss':
            try:
                from src.losses.focal_loss import FocalLoss
                focal_alpha = float(training_cfg.get('focal_alpha', 0.25))
                focal_gamma = float(training_cfg.get('focal_gamma', 2.0))
                
                criterion = FocalLoss(
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    reduction='mean'
                )
            except ImportError:
                logger.warning("FocalLoss not available, using CrossEntropyLoss")
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
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
            'avg_accuracy': (
                sum(accuracies) / len(accuracies) if accuracies else 0
            ),
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
        # STEP 1: Save global weights BEFORE training
        initial_global_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        # STEP 2: Broadcast model
        self.broadcast_model()
        
        # STEP 3: Train on clients
        client_metrics = []
        for client in self.clients:
            metrics = client.train_local()
            client_metrics.append(metrics)
        
        # STEP 4: Aggregate updates using initial weights
        self.aggregate_updates(initial_global_weights)
        
        # Compute averages
        avg_loss = (
            sum(m['local_loss'] for m in client_metrics) /
            len(client_metrics)
        )
        avg_acc = (
            sum(m['local_accuracy'] for m in client_metrics) /
            len(client_metrics)
        )
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'n_clients': len(self.clients)
        }
    
    def __repr__(self) -> str:
        return (f"FLServer(n_clients={len(self.clients)}, "
                f"aggregation={self.aggregation_method})")