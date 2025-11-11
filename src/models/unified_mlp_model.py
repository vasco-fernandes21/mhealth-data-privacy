#!/usr/bin/env python3
"""
Unified MLP Model for Privacy Research

Features:
- Works with any input dimension (features-only)
- Fast training (no temporal complexity)
- Fair comparison across datasets
- DP-compatible (LayerNorm, no BatchNorm)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class UnifiedMLPModel(BaseModel):
    """
    Unified MLP architecture for privacy-preserving research.
    
    Design principles:
    - Features-only input (no temporal)
    - LayerNorm for DP compatibility
    - Flexible hidden dimensions
    - Fast training for extensive experiments
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(config, device)

        dataset_cfg = config.get('dataset', {})
        model_cfg = config.get('model', {})

        # Input: always features (1D)
        input_dim = dataset_cfg.get('input_dim')
        self.n_classes = dataset_cfg.get('n_classes', 2)

        hidden_dims = model_cfg.get('hidden_dims', [128, 64])
        dropout = model_cfg.get('dropout', 0.3)

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # DP-compatible
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.n_classes))

        self.mlp = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features tensor
               - 2D (batch, features): normal case
               - 3D (batch, seq_len, features): temporal → average pool
        
        Returns:
            (batch, n_classes)
        """
        # Handle temporal input (convert to features)
        if x.dim() == 3:
            # Average pool over temporal dimension
            x = x.mean(dim=1)  # (batch, seq_len, features) → (batch, features)
        
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D or 3D input, got {x.dim()}D: {x.shape}"
            )

        return self.mlp(x)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'model_name': 'UnifiedMLPModel',
            'architecture': 'MLP (features-only)',
            'type': 'privacy_research_baseline',
            'dp_compatible': True,
            'temporal_support': 'average_pooling'
        })
        return info


if __name__ == "__main__":
    print("="*70)
    print("UNIFIED MLP MODEL - PRIVACY RESEARCH")
    print("="*70)

    # Sleep-EDF config (features)
    sleep_config = {
        'dataset': {
            'name': 'sleep-edf',
            'input_dim': 24,  # 8 features * 3 channels
            'n_classes': 5,
        },
        'model': {
            'hidden_dims': [128, 64],
            'dropout': 0.3
        }
    }

    # WESAD config (features)
    wesad_config = {
        'dataset': {
            'name': 'wesad',
            'input_dim': 140,  
            'n_classes': 2,
        },
        'model': {
            'hidden_dims': [128, 64],
            'dropout': 0.3
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    