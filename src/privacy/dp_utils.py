#!/usr/bin/env python3
"""Differential Privacy utilities using Opacus."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from opacus import PrivacyEngine
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DPConfig:
    
    def __init__(self, config: Dict[str, Any]):
        dp_cfg = config.get('differential_privacy', {})
        
        self.enabled = dp_cfg.get('enabled', False)
        self.target_epsilon = dp_cfg.get('target_epsilon', 8.0)
        self.target_delta = dp_cfg.get('target_delta', 1e-5)
        self.max_grad_norm = dp_cfg.get('max_grad_norm', 1.0)
        self.noise_multiplier = dp_cfg.get('noise_multiplier', 0.9)
        self.poisson_sampling = dp_cfg.get('poisson_sampling', True)
        self.grad_sample_mode = dp_cfg.get('grad_sample_mode', 'hooks')
    
    def __repr__(self) -> str:
        return (
            f"DPConfig(eps={self.target_epsilon}, delta={self.target_delta}, "
            f"noise={self.noise_multiplier}, clip={self.max_grad_norm})"
        )


def setup_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    dp_config: DPConfig,
    device: str = 'cuda'
) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
    
    if not dp_config.enabled:
        logger.warning("DP not enabled")
        return model, optimizer, None
    
    logger.info(f"Setting up PrivacyEngine: {dp_config}")
    
    privacy_engine = PrivacyEngine()
    
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
        poisson_sampling=dp_config.poisson_sampling,
        clipping_mode='flat',
        grad_sample_mode=dp_config.grad_sample_mode
    )
    
    return dp_model, dp_optimizer, privacy_engine


def get_epsilon(privacy_engine: Optional[PrivacyEngine], 
               delta: float) -> float:
    if privacy_engine is None:
        return float('inf')
    
    try:
        return privacy_engine.get_epsilon(delta)
    except Exception as e:
        logger.error(f"Error computing epsilon: {e}")
        return float('inf')


def check_dp_compatibility(model: nn.Module) -> Tuple[bool, list]:
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            incompatible_layers.append((name, type(module).__name__))
    
    is_compatible = len(incompatible_layers) == 0
    
    if not is_compatible:
        logger.warning(f"Model has {len(incompatible_layers)} DP-incompatible layers")
        for name, layer_type in incompatible_layers:
            logger.warning(f"  {name}: {layer_type}")
    
    return is_compatible, incompatible_layers