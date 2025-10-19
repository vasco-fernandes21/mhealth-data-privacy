#!/usr/bin/env python3
"""
Federated Learning Aggregation Methods.

Implements different aggregation strategies:
- FedAvg (default)
- Median (robust)
- Trimmed Mean
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FLAggregator:
    """Base class for FL aggregation."""
    
    @staticmethod
    def aggregate(client_updates: List[Dict[str, torch.Tensor]],
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: List of weight updates from each client
            client_weights: Weights for each client (e.g., sample counts)
        
        Returns:
            Aggregated updates
        """
        raise NotImplementedError


class FedAvgAggregator(FLAggregator):
    """Federated Averaging (FedAvg)."""
    
    @staticmethod
    def aggregate(client_updates: List[Dict[str, torch.Tensor]],
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation: weighted average of client updates.
        
        Args:
            client_updates: List of weight updates
            client_weights: Weights for averaging
        
        Returns:
            Aggregated updates
        """
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Aggregate
        aggregated = {}
        for param_name in client_updates[0].keys():
            aggregated[param_name] = torch.zeros_like(client_updates[0][param_name])
            
            for client_idx, update in enumerate(client_updates):
                weight = normalized_weights[client_idx]
                aggregated[param_name] += weight * update[param_name]
        
        return aggregated


class MedianAggregator(FLAggregator):
    """Robust aggregation using median (Byzantine-resistant)."""
    
    @staticmethod
    def aggregate(client_updates: List[Dict[str, torch.Tensor]],
                 client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Median aggregation: element-wise median of updates.
        
        Args:
            client_updates: List of weight updates
            client_weights: Ignored (not used for median)
        
        Returns:
            Aggregated updates
        """
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            # Stack all updates for this parameter
            updates_stacked = torch.stack([
                update[param_name] for update in client_updates
            ])
            
            # Compute median along client dimension
            aggregated[param_name] = torch.median(updates_stacked, dim=0)[0]
        
        return aggregated


class TrimmedMeanAggregator(FLAggregator):
    """Robust aggregation using trimmed mean (removes outliers)."""
    
    @staticmethod
    def aggregate(client_updates: List[Dict[str, torch.Tensor]],
                 client_weights: List[float] = None,
                 trim_fraction: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean aggregation: average after removing outliers.
        
        Args:
            client_updates: List of weight updates
            client_weights: Ignored
            trim_fraction: Fraction to trim from each side
        
        Returns:
            Aggregated updates
        """
        n_clients = len(client_updates)
        n_trim = int(np.ceil(n_clients * trim_fraction))
        
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            # Flatten all updates for this parameter
            updates_list = [
                update[param_name].flatten() for update in client_updates
            ]
            updates_stacked = torch.stack(updates_list)  # (n_clients, n_params)
            
            # Sort along client dimension
            sorted_updates, _ = torch.sort(updates_stacked, dim=0)
            
            # Trim and average
            trimmed = sorted_updates[n_trim:n_clients-n_trim, :]
            mean = torch.mean(trimmed, dim=0)
            
            # Reshape back
            aggregated[param_name] = mean.reshape(client_updates[0][param_name].shape)
        
        return aggregated


def create_aggregator(aggregation_method: str = 'fedavg', **kwargs) -> FLAggregator:
    """
    Factory function to create aggregator.
    
    Args:
        aggregation_method: 'fedavg', 'median', or 'trimmed_mean'
        **kwargs: Additional arguments for specific aggregators
    
    Returns:
        Aggregator instance
    """
    method = aggregation_method.lower()
    
    if method == 'fedavg':
        return FedAvgAggregator()
    elif method == 'median':
        return MedianAggregator()
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


if __name__ == "__main__":
    # Test aggregation
    print("Testing FL Aggregators...\n")
    
    # Create dummy updates
    client_updates = [
        {
            'weight': torch.randn(10, 5) + 0.1,
            'bias': torch.randn(5) + 0.05
        },
        {
            'weight': torch.randn(10, 5) + 0.15,
            'bias': torch.randn(5) + 0.03
        },
        {
            'weight': torch.randn(10, 5) + 0.12,
            'bias': torch.randn(5) + 0.04
        }
    ]
    
    client_weights = [100, 90, 110]
    
    # Test FedAvg
    print("Testing FedAvg...")
    aggregator = FedAvgAggregator()
    agg_fedavg = aggregator.aggregate(client_updates, client_weights)
    print(f"  Aggregated weight shape: {agg_fedavg['weight'].shape}")
    print(f"  ✅ FedAvg works\n")
    
    # Test Median
    print("Testing Median...")
    aggregator = MedianAggregator()
    agg_median = aggregator.aggregate(client_updates)
    print(f"  Aggregated weight shape: {agg_median['weight'].shape}")
    print(f"  ✅ Median works\n")
    
    # Test Trimmed Mean
    print("Testing Trimmed Mean...")
    aggregator = TrimmedMeanAggregator()
    agg_trimmed = aggregator.aggregate(client_updates, trim_fraction=0.33)
    print(f"  Aggregated weight shape: {agg_trimmed['weight'].shape}")
    print(f"  ✅ Trimmed Mean works\n")
    
    print("✅ All aggregators tested successfully!")