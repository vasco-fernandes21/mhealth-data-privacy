#!/usr/bin/env python3
"""Federated Learning server implementation."""

import torch
import torch.nn as nn
from typing import List, Dict, Any

from src.privacy.fl_client import FLClient
from src.privacy.fl_aggregators import create_aggregator


class FLServer:
    
    def __init__(self,
                 model: nn.Module,
                 clients: List[FLClient],
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.clients = clients
        self.config = config
        self.device = torch.device(device)
        
        fl_cfg = config.get('federated_learning', {})
        self.aggregation_method = fl_cfg.get('aggregation_method', 'fedavg')
        self.aggregator = create_aggregator(self.aggregation_method)
        
        label_smoothing = float(config['training'].get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def broadcast_model(self) -> None:
        global_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        for client in self.clients:
            client.set_weights(global_weights)
    
    def aggregate_updates(self, initial_weights: Dict[str, torch.Tensor]) -> None:
        client_updates = []
        client_weights = []
        
        for client in self.clients:
            updates = client.get_weight_updates(initial_weights)
            client_updates.append(updates)
            client_weights.append(1.0)
        
        aggregated_updates = self.aggregator.aggregate(client_updates, client_weights)
        
        for name, param in self.model.named_parameters():
            if name in aggregated_updates:
                param.data = initial_weights[name] + aggregated_updates[name]
    
    def evaluate_on_clients(self, val_loaders: List) -> Dict[str, float]:
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
            'accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
            'loss': sum(losses) / len(losses) if losses else 0.0,
        }
    
    def train_round(self) -> Dict[str, float]:
        initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        self.broadcast_model()
        
        client_metrics = []
        for client in self.clients:
            metrics = client.train_local()
            client_metrics.append(metrics)
        
        self.aggregate_updates(initial_weights)
        
        avg_loss = sum(m['loss'] for m in client_metrics) / len(client_metrics)
        avg_acc = sum(m['accuracy'] for m in client_metrics) / len(client_metrics)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
        }