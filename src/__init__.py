#!/usr/bin/env python3
"""
mHealth Privacy - Main Package

Privacy-Preserving Techniques Evaluation for Mobile Health Applications

This package provides:
- Data preprocessing (Sleep-EDF, WESAD)
- Model architectures (LSTM-based)
- Training loops (Baseline, DP, FL, FL+DP)
- Privacy utilities (Differential Privacy, Federated Learning)
- Evaluation metrics and analysis tools

Usage:
    from src.utils import set_all_seeds, setup_logging
    from src.models import SleepEDFModel, WESADModel
    from src.training import BaseTrainer, ProgressBar
    
    # Setup
    set_all_seeds(42)
    logger = setup_logging('./logs', level='INFO')
    
    # Create model
    config = {...}
    model = SleepEDFModel(config, device='cuda')
    
    # Train (see src.training for trainer implementations)
    trainer = BaselineTrainer(model, config, device='cuda')
    results = trainer.fit(train_loader, val_loader)
"""

__version__ = "0.1.0"
__author__ = "Eduardo Barbosa, Filipe Correia, Vasco Fernandes"
__description__ = "Privacy-Preserving Techniques in Mobile Health Applications"

# Lazy imports - avoid circular dependencies
# Users should import directly from submodules:
#   from src.utils.seed_utils import set_all_seeds
#   from src.models.sleep_edf_model import SleepEDFModel

__all__ = [
    'preprocessing',
    'models',
    'training',
    'privacy',
    'evaluation',
    'utils',
    'configs',
]