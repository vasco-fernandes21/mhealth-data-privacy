#!/usr/bin/env python3
"""Federated Learning aggregation methods."""

import torch
import numpy as np
from typing import List, Dict


class FedAvgAggregator:
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        aggregated = {}
        for param_name in client_updates[0].keys():
            aggregated[param_name] = torch.zeros_like(client_updates[0][param_name])
            
            for client_idx, update in enumerate(client_updates):
                weight = normalized_weights[client_idx]
                aggregated[param_name] += weight * update[param_name]
        
        return aggregated


class MedianAggregator:
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            updates_stacked = torch.stack([
                update[param_name] for update in client_updates
            ])
            aggregated[param_name] = torch.median(updates_stacked, dim=0)[0]
        
        return aggregated


class TrimmedMeanAggregator:
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None,
        trim_fraction: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        n_clients = len(client_updates)
        n_trim = int(np.ceil(n_clients * trim_fraction))
        
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            updates = torch.stack([update[param_name] for update in client_updates])
            sorted_updates, _ = torch.sort(updates, dim=0)
            trimmed = sorted_updates[n_trim:n_clients-n_trim]
            aggregated[param_name] = torch.mean(trimmed, dim=0)
        
        return aggregated


def create_aggregator(method: str = 'fedavg'):
    method = method.lower()
    
    if method == 'fedavg':
        return FedAvgAggregator()
    elif method == 'median':
        return MedianAggregator()
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator()
    else:
        raise ValueError(f"Unknown aggregation: {method}")