# models/__init__.py
"""
Models package for privacy-accuracy tradeoff research.

Exports:
- BaseModel: Abstract base class for all models
- UnifiedLSTMModel: Concrete implementation for WESAD & Sleep-EDF
"""

from .base_model import BaseModel
from .unified_mlp_model import UnifiedMLPModel

__all__ = [
    'BaseModel',
    'UnifiedMLPModel',
]

__version__ = '1.0.0'