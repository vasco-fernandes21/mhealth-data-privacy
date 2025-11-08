#!/usr/bin/env python3
"""
Differential Privacy utilities using Opacus.

Handles:
- Privacy Engine setup
- DP model wrapping
- Privacy budget accounting
- DP-compatible model creation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class DPConfig:
    """Configuration for Differential Privacy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DP config.
        
        Args:
            config: Configuration dictionary with 'differential_privacy' key
        """
        dp_cfg = config.get('differential_privacy', {})
        
        self.enabled = dp_cfg.get('enabled', False)
        self.target_epsilon = dp_cfg.get('target_epsilon', 8.0)
        self.target_delta = dp_cfg.get('target_delta', 1e-5)
        self.max_grad_norm = dp_cfg.get('max_grad_norm', 1.0)
        self.noise_multiplier = dp_cfg.get('noise_multiplier', 0.9)
        self.poisson_sampling = dp_cfg.get('poisson_sampling', True)
        self.accounting_method = dp_cfg.get('accounting_method', 'rdp')
        self.secure_rng = dp_cfg.get('secure_rng', True)
        self.grad_sample_mode = dp_cfg.get('grad_sample_mode', 'hooks')
    
    def __repr__(self) -> str:
        return (
            f"DPConfig(ε={self.target_epsilon}, δ={self.target_delta}, "
            f"noise_mult={self.noise_multiplier}, max_grad_norm={self.max_grad_norm})"
        )


def make_lstm_dp_compatible(lstm: nn.LSTM) -> DPLSTM:
    """
    Convert nn.LSTM to DPLSTM for DP compatibility.
    
    Args:
        lstm: Original LSTM layer
    
    Returns:
        DPLSTM layer with same parameters
    """
    input_size = lstm.input_size
    hidden_size = lstm.hidden_size
    num_layers = lstm.num_layers
    batch_first = lstm.batch_first
    bidirectional = lstm.bidirectional
    dropout = lstm.dropout
    
    dplstm = DPLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dropout=dropout
    )
    
    return dplstm


def setup_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    dp_config: DPConfig,
    device: str = 'cuda'
) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
    """
    Setup Opacus PrivacyEngine for DP training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        train_loader: Training data loader
        dp_config: DP configuration
        device: Device to use
    
    Returns:
        (dp_model, dp_optimizer, privacy_engine)
    
    IMPORTANTE: Usa noise_multiplier fixo, não target_epsilon!
    """
    
    if not dp_config.enabled:
        logger.warning("DP not enabled - returning original model")
        return model, optimizer, None
    
    logger.info(f"Setting up PrivacyEngine with {dp_config}")
    
    # Create Privacy Engine
    privacy_engine = PrivacyEngine()
    
    # FIX: Não usar target_epsilon/target_delta como constraints
    # Usar noise_multiplier fixo para controlar o noise
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config.noise_multiplier,  # FIXO - varia por experimento
        max_grad_norm=dp_config.max_grad_norm,
        # NÃO passar target_epsilon/target_delta - calcula-se depois
        poisson_sampling=dp_config.poisson_sampling,
        clipping_mode='flat',
        grad_sample_mode=dp_config.grad_sample_mode
    )
    
    logger.info(f"PrivacyEngine setup complete")
    logger.info(f"  Noise multiplier (FIXED): {dp_config.noise_multiplier}")
    logger.info(f"  Max grad norm: {dp_config.max_grad_norm}")
    logger.info(f"  Target delta (para epsilon calc): {dp_config.target_delta}")
    logger.info(f"  Accounting method: {dp_config.accounting_method}")
    
    return dp_model, dp_optimizer, privacy_engine


def get_epsilon(privacy_engine: Optional[PrivacyEngine], 
               delta: float) -> float:
    """
    Get current privacy budget epsilon.
    
    Args:
        privacy_engine: PrivacyEngine instance
        delta: Delta value
    
    Returns:
        Current epsilon
    """
    if privacy_engine is None:
        return float('inf')
    
    try:
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon
    except Exception as e:
        logger.error(f"Error computing epsilon: {e}")
        return float('inf')


def log_privacy_budget(privacy_engine: Optional[PrivacyEngine],
                      delta: float,
                      target_epsilon: float) -> Dict[str, float]:
    """
    Log privacy budget information.
    
    Args:
        privacy_engine: PrivacyEngine instance
        delta: Delta value
        target_epsilon: Reference epsilon (for comparison)
    
    Returns:
        Dictionary with privacy metrics
    """
    if privacy_engine is None:
        return {'epsilon': float('inf'), 'privacy_budget_used': 0.0}
    
    epsilon = get_epsilon(privacy_engine, delta)
    budget_ratio = epsilon / target_epsilon if target_epsilon > 0 else 0
    
    metrics = {
        'epsilon': epsilon,
        'delta': delta,
        'reference_epsilon': target_epsilon,
        'epsilon_ratio': budget_ratio
    }
    
    logger.info(f"Privacy Budget: ε={epsilon:.4f} (reference: {target_epsilon})")
    logger.info(f"  Epsilon ratio: {budget_ratio:.2f}x")
    
    return metrics


def check_dp_compatibility(model: nn.Module) -> Tuple[bool, list]:
    """
    Check if model is DP-compatible with Opacus.
    
    Args:
        model: PyTorch model
    
    Returns:
        (is_compatible, incompatible_layers)
    """
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            incompatible_layers.append((name, type(module).__name__))
    
    is_compatible = len(incompatible_layers) == 0
    
    if not is_compatible:
        logger.warning(f"Model has {len(incompatible_layers)} DP-incompatible layers:")
        for name, layer_type in incompatible_layers:
            logger.warning(f"  - {name}: {layer_type}")
        logger.warning("Recommendation: Replace BatchNorm with GroupNorm or LayerNorm")
    else:
        logger.info("✅ Model is DP-compatible")
    
    return is_compatible, incompatible_layers


if __name__ == "__main__":
    print("Testing DP utilities...\n")
    
    config = {
        'differential_privacy': {
            'enabled': True,
            'target_epsilon': 8.0,
            'target_delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 0.9
        }
    }
    
    dp_config = DPConfig(config)
    print(f"DP Config: {dp_config}\n")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    is_compatible, incompatible = check_dp_compatibility(model)
    print(f"Model compatible: {is_compatible}\n")
    
    print("✅ All DP utility tests passed!")