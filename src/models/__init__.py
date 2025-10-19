#!/usr/bin/env python3
"""
Model architectures for mHealth datasets.

Models:
- BaseModel: Abstract base class for all models
- SleepEDFModel: LSTM model for Sleep-EDF sleep stage classification
- WESADModel: LSTM model for WESAD stress detection

Each model:
- Inherits from BaseModel
- Implements device management
- Supports save/load
- Includes get_model_info()

Usage:
    from src.models import SleepEDFModel, WESADModel
    
    config = {'dataset': {...}, 'model': {...}}
    model = SleepEDFModel(config, device='cuda')
    
    # Forward pass
    output = model(x)  # x shape: (batch, seq, features)
    
    # Save/Load
    model.save('model.pth')
    loaded = SleepEDFModel.load('model.pth', device='cuda')
    
    # Info
    info = model.get_model_info()
    model.print_model_summary()
"""

from .base_model import BaseModel

from .sleep_edf_model import (
    SleepEDFModel,
    create_sleep_edf_model,
)

from .wesad_model import (
    WESADModel,
    create_wesad_model,
)

__all__ = [
    # Base
    'BaseModel',
    # Sleep-EDF
    'SleepEDFModel',
    'create_sleep_edf_model',
    # WESAD
    'WESADModel',
    'create_wesad_model',
]