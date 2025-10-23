#!/usr/bin/env python3
"""
LSTM model for WESAD stress detection with optional Attention.

Input: (batch, n_channels, timesteps) = (batch, 16, 1920)
Output: (batch, n_classes) = (batch, 2)

Architecture:
- Input projection
- Bidirectional LSTM (core)
- Optional: Multi-head Attention (can be disabled)
- Dense layers for classification
- DP-optimized (GroupNorm instead of BatchNorm)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class AttentionLayer(nn.Module):
    """
    Multi-head Attention layer (DP-compatible).
    
    Can be completely removed for baseline consistency
    if needed. Optional component.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Attention dropout
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention.
        
        Args:
            x: (batch, seq_len, hidden_dim)
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class WESADModel(BaseModel):
    """LSTM model for WESAD stress detection."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize WESAD model.
        
        Args:
            config: Configuration dictionary with:
                - dataset.n_channels
                - dataset.n_classes
                - model.lstm_units
                - model.lstm_layers
                - model.dropout
                - model.dense_layers
                - model.use_attention (optional, default=False)
                - model.attention_heads (optional, default=4)
            device: Device to use
        """
        super().__init__(config, device)
        
        # Extract config
        dataset_cfg = config['dataset']
        model_cfg = config['model']
        
        self.n_channels = dataset_cfg['n_channels']  # 16
        self.n_classes = dataset_cfg['n_classes']    # 2 (binary)
        
        lstm_units = model_cfg['lstm_units']         # 64
        lstm_layers = model_cfg['lstm_layers']       # 2
        dropout = model_cfg['dropout']               # 0.3
        dense_layers = model_cfg['dense_layers']     # [64, 32]
        
        # Optional attention
        use_attention = model_cfg.get('use_attention', False)
        attention_heads = model_cfg.get('attention_heads', 4)
        
        # Initial projection layer to reduce from 16 channels to 128
        self.input_proj = nn.Linear(self.n_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)
        self.input_dropout = nn.Dropout(0.2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Dimension after bidirectional LSTM
        lstm_output_size = lstm_units * 2  # 128
        
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
        
        for dense_size in dense_layers:  # [64, 32]
            dense_layers_list.append(nn.Linear(prev_size, dense_size))
            dense_layers_list.append(nn.ReLU())
            if dense_size != dense_layers[-1]:
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
               e.g., (64, 16, 1920)
        
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Convert to (batch, timesteps, channels)
        x = x.permute(0, 2, 1)  # (batch, 1920, 16)
        
        # Initial projection
        x = self.input_proj(x)  # (batch, 1920, 128)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, 1920, 128)
        
        # Optional: Apply attention
        if self.use_attention:
            attn_out = self.attention(lstm_out)  # (batch, 1920, 128)
            attn_out = self.attention_norm(attn_out.transpose(1, 2)).transpose(1, 2)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        # Get last hidden states (concat forward + backward)
        h_forward = h_n[-2]  # Last forward layer
        h_backward = h_n[-1]  # Last backward layer
        h_last = torch.cat([h_forward, h_backward], dim=1)  # (batch, 128)
        
        # Apply normalization
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
            'use_attention': self.use_attention,
            'lstm_config': {
                'units': self.lstm.hidden_size,
                'layers': self.lstm.num_layers,
                'bidirectional': self.lstm.bidirectional
            }
        })
        return info


def create_wesad_model(config: Dict[str, Any], 
                       device: str = 'cpu') -> WESADModel:
    """Factory function to create WESAD model."""
    model = WESADModel(config, device)
    return model


if __name__ == "__main__":
    print("Testing WESAD model...\n")
    
    config = {
        'dataset': {
            'name': 'wesad',
            'n_channels': 16,
            'n_classes': 2,
            'sequence_length': 1920
        },
        'model': {
            'lstm_units': 64,
            'lstm_layers': 2,
            'dropout': 0.3,
            'dense_layers': [64, 32],
            'use_attention': False  
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WESADModel(config, device=device)
    
    model.print_model_summary()
    
    print("Test forward pass:")
    x = torch.randn(64, 16, 1920).to(device)
    y = model(x)
    print(f"  Input: {x.shape} → Output: {y.shape}")
    print(f"  ✅ Match: {y.shape == (64, 2)}\n")
    
    print("✅ All tests passed!")