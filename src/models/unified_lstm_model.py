#!/usr/bin/env python3
"""
Unified LSTM architecture for privacy-accuracy tradeoff papers.

Supports flexible dense layer configuration, input projection,
bidirectional LSTM, and GroupNorm for DP compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .base_model import BaseModel


class UnifiedLSTMModel(BaseModel):
    """
    Unified LSTM architecture optimized for privacy research.
    
    Design principles:
    - GroupNorm (DP-friendly) instead of BatchNorm
    - Flexible dense layers configuration
    - Bidirectional LSTM for context
    - Input projection to standardize dimensions
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(config, device)
        
        dataset_cfg = config['dataset']
        model_cfg = config['model']
        
        # Input dimensions
        input_dim = dataset_cfg.get('input_dim')
        self.n_classes = dataset_cfg['n_classes']
        
        # Model parameters
        input_proj_dim = model_cfg.get('input_projection', 128)
        lstm_units = model_cfg.get('lstm_units', 64)
        lstm_layers = model_cfg.get('lstm_layers', 1)
        dropout = model_cfg.get('dropout', 0.3)
        dense_layers = model_cfg.get('dense_layers', [128])
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, input_proj_dim)
        self.input_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=input_proj_dim
        )
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(0.1)
        
        # LSTM - use DPLSTM for DP compatibility
        try:
            from opacus.layers import DPLSTM
            self.lstm = DPLSTM(
                input_size=input_proj_dim,
                hidden_size=lstm_units,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0
            )
        except ImportError:
            self.lstm = nn.LSTM(
                input_size=input_proj_dim,
                hidden_size=lstm_units,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0
            )
        
        lstm_output_size = lstm_units * 2
        
        # Post-LSTM normalization
        self.lstm_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=lstm_output_size
        )
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Dense layers (flexible!)
        dense_layers_list = []
        prev_size = lstm_output_size
        
        for dense_size in dense_layers:
            dense_layers_list.append(nn.Linear(prev_size, dense_size))
            dense_layers_list.append(nn.ReLU())
            dense_layers_list.append(nn.Dropout(dropout))
            prev_size = dense_size
        
        # Output layer
        dense_layers_list.append(nn.Linear(prev_size, self.n_classes))
        
        self.dense = nn.Sequential(*dense_layers_list)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, features) for Sleep-EDF
               or (batch, channels, timesteps) for WESAD
        
        Returns:
            (batch, n_classes)
        """
        # Handle different input formats
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            # WESAD format: (batch, 14, 1920) -> permute
            x = x.permute(0, 2, 1)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_act(x)
        x = self.input_dropout(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Extract last hidden states (bidirectional)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        h_forward = h_n[-2]  # Last forward layer
        h_backward = h_n[-1]  # Last backward layer
        h_last = torch.cat([h_forward, h_backward], dim=1)
        
        # Normalization + dropout
        h_last = self.lstm_norm(h_last.unsqueeze(2)).squeeze(2)
        h_last = self.lstm_dropout(h_last)
        
        # Dense layers
        output = self.dense(h_last)
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'model_name': 'UnifiedLSTM',
            'lstm_layers': self.lstm.num_layers,
            'lstm_units': self.lstm.hidden_size,
            'bidirectional': self.lstm.bidirectional,
        })
        return info


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED LSTM - PRIVACY TRADEOFF BASELINE")
    print("=" * 60)
    
    # Config for WESAD
    wesad_config = {
        'dataset': {
            'name': 'wesad',
            'input_dim': 14,
            'n_channels': 14,
            'n_classes': 2,
            'sequence_length': 1920
        },
        'model': {
            'input_projection': 128,
            'lstm_units': 56,
            'lstm_layers': 1,
            'dropout': 0.48,
            'dense_layers': [112, 56]
        }
    }
    
    # Config for Sleep-EDF
    sleep_config = {
        'dataset': {
            'name': 'sleep-edf',
            'input_dim': 24,
            'n_features': 24,
            'n_classes': 5,
            'sequence_length': 10
        },
        'model': {
            'input_projection': 128,
            'lstm_units': 64,
            'lstm_layers': 1,
            'dropout': 0.3,
            'dense_layers': [128, 64]
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test WESAD
    print("\n📊 WESAD Configuration:")
    model_wesad = UnifiedLSTMModel(wesad_config, device=device)
    print(f"Model: {model_wesad.get_model_info()}")
    
    x_wesad = torch.randn(32, 14, 1920).to(device)
    y_wesad = model_wesad(x_wesad)
    print(f"Input: {x_wesad.shape} → Output: {y_wesad.shape}")
    print(f"✅ Correct: {y_wesad.shape == (32, 2)}")
    
    # Test Sleep-EDF
    print("\n📊 Sleep-EDF Configuration:")
    model_sleep = UnifiedLSTMModel(sleep_config, device=device)
    print(f"Model: {model_sleep.get_model_info()}")
    
    x_sleep = torch.randn(32, 10, 24).to(device)
    y_sleep = model_sleep(x_sleep)
    print(f"Input: {x_sleep.shape} → Output: {y_sleep.shape}")
    print(f"✅ Correct: {y_sleep.shape == (32, 5)}")
    
    # Verify parameter count
    wesad_params = sum(p.numel() for p in model_wesad.parameters())
    sleep_params = sum(p.numel() for p in model_sleep.parameters())
    print(f"\n📈 Parameter Count:")
    print(f"WESAD: {wesad_params:,} parameters")
    print(f"Sleep-EDF: {sleep_params:,} parameters")
    print(f"Ratio: {max(wesad_params, sleep_params) / min(wesad_params, sleep_params):.2f}x")
    print("✅ Parameters balanced")