"""
Training implementations for different scenarios.

Exports:
- BaselineTrainer: Standard training
- DPTrainer: Differential Privacy training
- FLTrainer: Federated Learning training
- FLDPTrainer: Federated Learning + Differential Privacy
"""

from .baseline_trainer import BaselineTrainer
from .dp_trainer import DPTrainer
from .fl_trainer import FLTrainer

try:
    from .fl_dp_trainer import FLDPTrainer
except ImportError:
    FLDPTrainer = None

__all__ = [
    'BaselineTrainer',
    'DPTrainer',
    'FLTrainer',
    'FLDPTrainer',
]