#!/usr/bin/env python3
"""
Federated Learning Aggregation Methods.

Implements FedAvg (default), Median (robust), Trimmed Mean.
"""

import torch
import numpy as np
from typing import List, Dict


class FedAvgAggregator:
    """Federated Averaging (FedAvg) - Standard aggregation."""
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation: weighted average of client updates.
        
        Args:
            client_updates: List of weight updates from each client
            client_weights: Weights for each client (e.g., sample counts)
        
        Returns:
            Aggregated updates dictionary
        """
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Aggregate
        aggregated = {}
        for param_name in client_updates[0].keys():
            aggregated[param_name] = torch.zeros_like(
                client_updates[0][param_name]
            )
            
            for client_idx, update in enumerate(client_updates):
                weight = normalized_weights[client_idx]
                aggregated[param_name] += weight * update[param_name]
        
        return aggregated


class MedianAggregator:
    """Robust aggregation using median (Byzantine-resistant)."""
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Median aggregation: element-wise median of updates.
        
        Args:
            client_updates: List of weight updates
            client_weights: Ignored (not used for median)
        
        Returns:
            Aggregated updates dictionary
        """
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            # Stack all updates for this parameter
            updates_stacked = torch.stack([
                update[param_name] for update in client_updates
            ])
            
            # Compute element-wise median
            aggregated[param_name] = torch.median(updates_stacked, dim=0)[0]
        
        return aggregated


class TrimmedMeanAggregator:
    """Robust aggregation using trimmed mean (removes outliers)."""
    
    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None,
        trim_fraction: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean aggregation: average after removing outliers.
        
        Args:
            client_updates: List of weight updates
            client_weights: Ignored
            trim_fraction: Fraction to trim from each side (default 0.1)
        
        Returns:
            Aggregated updates dictionary
        """
        n_clients = len(client_updates)
        n_trim = int(np.ceil(n_clients * trim_fraction))
        
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            # Stack all updates
            updates = torch.stack([
                update[param_name] for update in client_updates
            ])  # (n_clients, ...)
            
            # Sort and trim along client dimension
            sorted_updates, _ = torch.sort(updates, dim=0)
            trimmed = sorted_updates[n_trim:n_clients-n_trim]
            
            # Average
            aggregated[param_name] = torch.mean(trimmed, dim=0)
        
        return aggregated


def create_aggregator(method: str = 'fedavg'):
    """
    Factory to create aggregator.
    
    Args:
        method: 'fedavg', 'median', or 'trimmed_mean'
    
    Returns:
        Aggregator instance
    """
    method = method.lower()
    
    if method == 'fedavg':
        return FedAvgAggregator()
    elif method == 'median':
        return MedianAggregator()
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator()
    else:
        raise ValueError(f"Unknown aggregation: {method}")