#!/usr/bin/env python3
"""
MHealth Privacy: Privacy-Preserving Analysis for Mobile Health Data

A comprehensive framework for evaluating privacy-preserving techniques
(Federated Learning and Differential Privacy) in mobile health applications.

Main modules:
    - preprocessing: Data loading and preparation
    - models: Neural network models
    - training: Training loops for different privacy scenarios
    - privacy: Privacy-related utilities (DP, FL)
    - evaluation: Metrics and analysis
    - utils: Device, logging, seeding utilities
    - configs: Configuration management

Usage:
    from src.preprocessing.sleep_edf import load_windowed_sleep_edf
    from src.models.sleep_edf_model import SleepEDFModel
    from src.training.trainers.baseline_trainer import BaselineTrainer
    from src.utils.seed_utils import set_reproducible

Example:
    >>> from src.utils.seed_utils import set_reproducible
    >>> set_reproducible(seed=42, device='cuda')
    >>> from src.models.sleep_edf_model import SleepEDFModel
    >>> config = {'dataset': {...}, 'model': {...}}
    >>> model = SleepEDFModel(config, device='cuda')
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Import main modules for convenient access
from . import utils
from . import models
from . import training
from . import privacy
from . import preprocessing
from . import evaluation
from . import configs

# Import commonly used classes and functions
from .utils.seed_utils import set_all_seeds, set_reproducible, set_deterministic
from .utils.logging_utils import setup_logging, get_logger, ExperimentLogger
from .models.base_model import BaseModel
from .training.base_trainer import BaseTrainer

__all__ = [
    # Modules
    'utils',
    'models',
    'training',
    'privacy',
    'preprocessing',
    'evaluation',
    'configs',
    # Commonly used classes
    'BaseModel',
    'BaseTrainer',
    # Commonly used functions
    'set_all_seeds',
    'set_reproducible',
    'set_deterministic',
    'setup_logging',
    'get_logger',
    'ExperimentLogger',
]


def __repr__():
    """Package representation."""
    return (
        f"mhealth-privacy v{__version__}\n"
        f"Privacy-Preserving Analysis for Mobile Health Data"
    )


print(f"MHealth Privacy v{__version__} loaded")