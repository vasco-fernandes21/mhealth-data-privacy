#!/usr/bin/env python3
"""
Configuration management.

Provides:
- YAML-based configurations for datasets
- Training hyperparameters
- Privacy parameters
- Experiment settings

Configuration files:
- sleep_edf.yaml: Sleep-EDF dataset config
- wesad.yaml: WESAD dataset config
- training_defaults.yaml: Default training parameters
- privacy_defaults.yaml: Default DP/FL parameters

Usage:
    import yaml
    
    with open('src/configs/sleep_edf.yaml') as f:
        config = yaml.safe_load(f)
    
    model = SleepEDFModel(config, device='cuda')
"""

import os
from pathlib import Path

# Config directory
CONFIG_DIR = Path(__file__).parent

__all__ = [
    'CONFIG_DIR',
]