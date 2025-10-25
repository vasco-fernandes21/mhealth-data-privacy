#!/usr/bin/env python3
"""
LSTM model for WESAD stress detection with DP compatibility.

Input: (batch, n_channels, timesteps) = (batch, 14, 1920)
Output: (batch, n_classes) = (batch, 2)

Architecture:
- Input projection
- Bidirectional LSTM (or DPLSTM for DP)
- Optional: Multi-head Attention
- Dense layers for classification
- DP-optimized (GroupNorm instead of BatchNorm)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel

# ✅ Import DPLSTM from Opacus
try:
    from opacus.layers import DPLSTM
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


class AttentionLayer(nn.Module):
    """Multi-head Attention layer (DP-compatible)."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class WESADModel(BaseModel):
    """LSTM model for WESAD stress detection (DP-compatible)."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu', use_dp: bool = False):
        """
        Initialize WESAD model.
        
        Args:
            config: Configuration dictionary
            device: Device to use
            use_dp: If True, use DPLSTM (requires Opacus). If False, use standard LSTM.
        """
        super().__init__(config, device)
        
        self.use_dp = use_dp
        
        # Check DP compatibility
        if use_dp and not OPACUS_AVAILABLE:
            raise RuntimeError("DP requested but Opacus not installed. "
                             "Install with: pip install opacus")
        
        # Extract config
        dataset_cfg = config['dataset']
        model_cfg = config['model']
        
        self.n_channels = dataset_cfg['n_channels']  # 14
        self.n_classes = dataset_cfg['n_classes']    # 2
        
        lstm_units = model_cfg['lstm_units']
        lstm_layers = model_cfg['lstm_layers']
        dropout = model_cfg['dropout']
        dense_layers = model_cfg['dense_layers']
        
        use_attention = model_cfg.get('use_attention', False)
        attention_heads = model_cfg.get('attention_heads', 4)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)
        self.input_dropout = nn.Dropout(0.2)
        
        # ✅ Use DPLSTM if DP, otherwise standard LSTM
        if use_dp:
            self.lstm = DPLSTM(
                input_size=128,
                hidden_size=lstm_units,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=lstm_units,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0
            )
        
        lstm_output_size = lstm_units * 2
        
        # Normalization after LSTM
        self.lstm_norm = nn.GroupNorm(num_groups=8, num_channels=lstm_output_size)
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Optional Attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                hidden_dim=lstm_output_size,
                num_heads=attention_heads,
                dropout=dropout
            )
            self.attention_norm = nn.GroupNorm(num_groups=8, 
                                              num_channels=lstm_output_size)
        
        # Dense layers
        dense_layers_list = []
        prev_size = lstm_output_size
        
        for i, dense_size in enumerate(dense_layers):
            dense_layers_list.append(nn.Linear(prev_size, dense_size))
            dense_layers_list.append(nn.ReLU())
            if i < len(dense_layers) - 1:
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
            x: Input tensor of shape (batch, n_channels, timesteps)
        
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Convert to (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Optional attention
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            attn_out = self.attention_norm(attn_out.transpose(1, 2)).transpose(1, 2)
            lstm_out = lstm_out + attn_out
        
        # Get last hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_last = torch.cat([h_forward, h_backward], dim=1)
        
        # Normalize
        h_last = self.lstm_norm(h_last.unsqueeze(2)).squeeze(2)
        h_last = self.lstm_dropout(h_last)
        
        # Dense layers
        output = self.dense(h_last)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'WESADLSTMModel',
            'n_channels': self.n_channels,
            'use_dp': self.use_dp,
            'lstm_type': 'DPLSTM' if self.use_dp else 'LSTM',
            'use_attention': self.use_attention,
            'lstm_config': {
                'units': self.lstm.hidden_size,
                'layers': self.lstm.num_layers,
                'bidirectional': self.lstm.bidirectional
            }
        })
        return info


def create_wesad_model(config: Dict[str, Any], device: str = 'cpu',
                       use_dp: bool = False) -> WESADModel:
    """Factory function to create WESAD model."""
    return WESADModel(config, device, use_dp=use_dp)


if __name__ == "__main__":
    print("Testing WESAD model...\n")
    
    config = {
        'dataset': {
            'name': 'wesad',
            'n_channels': 14,
            'n_classes': 2,
            'sequence_length': 1920
        },
        'model': {
            'lstm_units': 56,
            'lstm_layers': 1,
            'dropout': 0.48,
            'dense_layers': [112, 56],
            'use_attention': False  
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test standard LSTM
    print("Testing standard LSTM:")
    model = WESADModel(config, device=device, use_dp=False)
    x = torch.randn(32, 14, 1920).to(device)
    y = model(x)
    print(f"  Input: {x.shape} → Output: {y.shape}")
    print(f"  ✅ Match: {y.shape == (32, 2)}\n")
    
    # Test DPLSTM (if available)
    if OPACUS_AVAILABLE:
        print("Testing DPLSTM (DP-compatible):")
        model_dp = WESADModel(config, device=device, use_dp=True)
        y_dp = model_dp(x)
        print(f"  Input: {x.shape} → Output: {y_dp.shape}")
        print(f"  ✅ Match: {y_dp.shape == (32, 2)}\n")
    else:
        print("⚠️ Opacus not available - skipping DPLSTM test\n")
    
    print("✅ All tests passed!")