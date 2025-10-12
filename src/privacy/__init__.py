"""
Privacy module for Differential Privacy and Federated Learning implementations.

This module provides:
- Differential Privacy training with TensorFlow Privacy
- Federated Learning simulation with Flower
- Privacy budget computation
"""

from .dp_training import train_with_dp, compute_epsilon, build_dp_model
from .fl_training import train_with_fl, HealthDataClient, split_data_for_clients

__all__ = [
    "train_with_dp",
    "compute_epsilon",
    "build_dp_model",
    "train_with_fl",
    "HealthDataClient",
    "split_data_for_clients",
]

