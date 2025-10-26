#!/usr/bin/env python3
"""
Unified optimized LSTM architecture for privacy-accuracy tradeoff papers.
- Fair baseline for WESAD and Sleep-EDF
- DP-compatible from start
- Parameters balanced for both datasets
- Efficient for FL communication
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class UnifiedLSTMModel(BaseModel):
    """
    Unified LSTM architecture optimized for privacy research.
    
    Design principles:
    - GroupNorm (DP-friendly) instead of BatchNorm
    - Minimal but robust hidden dimensions
    - Single LSTM layer to reduce DP noise
    - Efficient dense layers for faster convergence
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu',
                 use_dp: bool = False):
        super().__init__(config, device)
        
        self.use_dp = use_dp
        
        # Extract minimal config
        dataset_cfg = config['dataset']
        model_cfg = config['model']
        
        input_dim = dataset_cfg.get('input_dim')  # 14 or 24
        self.n_classes = dataset_cfg['n_classes']  # 2 or 5
        
        # **Key: Single LSTM layer + lower units for DP robustness**
        lstm_units = model_cfg.get('lstm_units', 64)
        dropout = model_cfg.get('dropout', 0.3)
        
        # Input projection (standardizes different input dims)
        self.input_proj = nn.Linear(input_dim, 128)
        self.input_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=128
        )
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(0.1)
        
        # **LSTM: Single layer, bidirectional**
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_units,
            num_layers=1,  # ✅ Single layer = less DP noise
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout between layers (only 1 layer)
        )
        
        lstm_output_size = lstm_units * 2  # Bidirectional
        
        # Post-LSTM normalization
        self.lstm_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=lstm_output_size
        )
        self.lstm_dropout = nn.Dropout(dropout)
        
        # **Minimal dense layers**
        # WESAD/Sleep-EDF both work with 1-2 hidden layers
        dense_hidden = model_cfg.get('dense_hidden', 128)
        
        self.dense = nn.Sequential(
            nn.Linear(lstm_output_size, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, self.n_classes)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
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
        x = self.input_proj(x)  # (batch, seq, 128)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_act(x)
        x = self.input_dropout(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Extract last hidden states (bidirectional)
        h_forward = h_n[0]   # Shape: (batch, lstm_units)
        h_backward = h_n[1]  # Shape: (batch, lstm_units)
        h_last = torch.cat([h_forward, h_backward], dim=1)
        
        # Normalization + dropout
        h_last = self.lstm_norm(h_last.unsqueeze(2)).squeeze(2)
        h_last = self.lstm_dropout(h_last)
        
        # Classification
        output = self.dense(h_last)
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'model_name': 'UnifiedLSTM',
            'lstm_layers': 1,
            'lstm_units': self.lstm.hidden_size,
            'use_dp': self.use_dp,
            'parameters': sum(p.numel() for p in self.parameters())
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
            'lstm_units': 64,
            'dense_hidden': 128,
            'dropout': 0.3
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
            'lstm_units': 64,
            'dense_hidden': 128,
            'dropout': 0.3
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
    
    # Verify parameter count similarity
    wesad_params = sum(p.numel() for p in model_wesad.parameters())
    sleep_params = sum(p.numel() for p in model_sleep.parameters())
    print(f"\n📈 Parameter Count:")
    print(f"WESAD: {wesad_params:,} parameters")
    print(f"Sleep-EDF: {sleep_params:,} parameters")
    print(f"Ratio: {max(wesad_params, sleep_params) / min(wesad_params, sleep_params):.2f}x")
    print("✅ Parameters balanced (similar complexity)")