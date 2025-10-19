#!/usr/bin/env python3
"""
Model factory for creating models based on dataset.

Provides unified interface for model creation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def create_model(dataset: str, config: Dict[str, Any], device: str = 'cuda') -> nn.Module:
    """
    Create model for specified dataset.
    
    Args:
        dataset: Dataset name ('sleep-edf', 'wesad')
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Initialized model
    """
    dataset = dataset.lower()
    
    if dataset == 'sleep-edf':
        logger.info("Creating Sleep-EDF model (LSTM)")
        model = SleepEDFModel(config, device=device)
    
    elif dataset == 'wesad':
        logger.info("Creating WESAD model (LSTM)")
        model = WESADModel(config, device=device)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    logger.info(f"Model created: {model.__class__.__name__}")
    return model


def get_model_info(dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model information without creating it.
    
    Args:
        dataset: Dataset name
        config: Configuration dictionary
    
    Returns:
        Model information
    """
    dataset = dataset.lower()
    
    if dataset == 'sleep-edf':
        return {
            'name': 'SleepEDFModel',
            'architecture': 'LSTM (Bidirectional)',
            'input_shape': (None, config['dataset']['sequence_length'], config['dataset']['n_features']),
            'output_shape': (None, config['dataset']['n_classes']),
            'parameters': {
                'lstm_units': config['model']['lstm_units'],
                'lstm_layers': config['model']['lstm_layers'],
                'dropout': config['model']['dropout'],
                'dense_layers': config['model']['dense_layers']
            }
        }
    
    elif dataset == 'wesad':
        return {
            'name': 'WESADModel',
            'architecture': 'LSTM (Bidirectional) with Channel Projection',
            'input_shape': (None, config['dataset']['n_channels'], config['dataset']['sequence_length']),
            'output_shape': (None, config['dataset']['n_classes']),
            'parameters': {
                'input_projection': 128,
                'lstm_units': config['model']['lstm_units'],
                'lstm_layers': config['model']['lstm_layers'],
                'dropout': config['model']['dropout'],
                'dense_layers': config['model']['dense_layers']
            }
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    print("Model factory loaded successfully")