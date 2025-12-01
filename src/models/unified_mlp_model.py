#!/usr/bin/env python3
"""Unified MLP model with fixed architecture for consistent comparisons."""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class UnifiedMLPModel(BaseModel):
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(config, device)

        dataset_cfg = config.get('dataset', {})
        model_cfg = config.get('model', {})

        input_dim = dataset_cfg.get('input_dim')
        self.n_classes = dataset_cfg.get('n_classes', 2)

        hidden_dims = [256, 128]
        dropout = model_cfg.get('dropout', 0.3)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.n_classes))

        self.mlp = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D: {x.shape}")
        return self.mlp(x)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'model_name': 'UnifiedMLPModel',
            'architecture': 'MLP',
            'hidden_dims': [256, 128],
            'dp_compatible': True
        })
        return info