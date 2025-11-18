#!/usr/bin/env python3
"""
Unified MLP Model for Privacy Research

Features:
- CONSISTENT architecture across datasets (scientific rigor)
- DP-compatible (LayerNorm, no BatchNorm)
- Fast training (features-only, no temporal complexity)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class UnifiedMLPModel(BaseModel):
    """
    Unified MLP architecture for privacy-preserving research.
    
    Design principles:
    - FIXED architecture [256, 128] for all datasets (scientific consistency)
    - Features-only input (no temporal)
    - LayerNorm for DP compatibility
    - Dropout for regularization
    
    Architecture:
        Input (D) → Linear(D, 256) → LayerNorm → ReLU → Dropout(0.3)
                 → Linear(256, 128) → LayerNorm → ReLU → Dropout(0.3)
                 → Linear(128, n_classes)
    
    where D = input_dim (140 for WESAD, 24 for Sleep-EDF)
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(config, device)

        dataset_cfg = config.get('dataset', {})
        model_cfg = config.get('model', {})

        input_dim = dataset_cfg.get('input_dim')
        self.n_classes = dataset_cfg.get('n_classes', 2)

        # FIXED hidden dimensions for scientific consistency
        # Override config if present to enforce uniformity
        hidden_dims = [256, 128]  
        dropout = model_cfg.get('dropout', 0.3)
        
        print(f"Unified MLP: {input_dim}D → {hidden_dims} → {self.n_classes} classes")

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
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,} (trainable: {trainable_params:,})")

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
        if x.dim() == 3:
            x = x.mean(dim=1)  # Temporal → features
        
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
            'hidden_dims': [256, 128],
            'type': 'privacy_research_baseline',
            'dp_compatible': True,
            'normalization': 'LayerNorm',
            'temporal_support': 'average_pooling'
        })
        return info


if __name__ == "__main__":
    print("="*70)
    print("UNIFIED MLP MODEL - PRIVACY RESEARCH")
    print("="*70)
    print("Fixed architecture [256, 128] for scientific consistency\n")

    # Sleep-EDF config
    sleep_config = {
        'dataset': {
            'name': 'sleep-edf',
            'input_dim': 24,
            'n_classes': 5,
        },
        'model': {
            'dropout': 0.3
        }
    }

    # WESAD config
    wesad_config = {
        'dataset': {
            'name': 'wesad',
            'input_dim': 140,  
            'n_classes': 2,
        },
        'model': {
            'dropout': 0.3
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Test both configurations
    print("Creating models:")
    sleep_model = UnifiedMLPModel(sleep_config, device)
    print()
    wesad_model = UnifiedMLPModel(wesad_config, device)
    
    print("\n" + "="*70)