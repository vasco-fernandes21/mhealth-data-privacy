#!/usr/bin/env python3
"""
LSTM model for WESAD stress detection.

Input: (batch, n_channels, timesteps) = (batch, 16, 1920)
Output: (batch, n_classes) = (batch, 2)

Architecture:
- Initial projection to reduce dimensionality
- Bidirectional LSTM for temporal context
- Dense layers for classification
- DP-optimized (uses GroupNorm instead of BatchNorm)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


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
        
        # Initial projection layer to reduce from 16 channels to 128
        self.input_proj = nn.Linear(self.n_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # DP-safe
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
        
        # Normalization after LSTM (DP-safe)
        self.lstm_norm = nn.GroupNorm(num_groups=8, num_channels=lstm_output_size)
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Dense layers
        dense_layers_list = []
        prev_size = lstm_output_size
        
        for dense_size in dense_layers:  # [64, 32]
            dense_layers_list.append(nn.Linear(prev_size, dense_size))
            dense_layers_list.append(nn.ReLU())
            if dense_size != dense_layers[-1]:  # Don't apply dropout after last
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
            e.g., (64, 2)
        """
        # Input is (batch, channels, timesteps)
        # Need to convert to (batch, timesteps, channels) for LSTM
        x = x.permute(0, 2, 1)  # (batch, 1920, 16)
        
        # Initial projection
        x = self.input_proj(x)  # (batch, 1920, 128)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)  # GroupNorm expects (N, C, L)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, 1920, 128)
        
        # Get last hidden states from both directions
        h_forward = h_n[-2]  # Last forward layer
        h_backward = h_n[-1]  # Last backward layer
        
        # Concatenate: (batch, 128)
        h_last = torch.cat([h_forward, h_backward], dim=1)
        
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
            'lstm_config': {
                'units': self.lstm.hidden_size,
                'layers': self.lstm.num_layers,
                'bidirectional': self.lstm.bidirectional
            }
        })
        return info


def create_wesad_model(config: Dict[str, Any], 
                       device: str = 'cpu') -> WESADModel:
    """
    Factory function to create WESAD model.
    
    Args:
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Initialized model
    """
    model = WESADModel(config, device)
    return model


if __name__ == "__main__":
    # Test script
    print("Testing WESAD model...\n")
    
    # Create config
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
            'dense_layers': [64, 32]
        }
    }
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WESADModel(config, device=device)
    
    # Print info
    model.print_model_summary()
    
    # Test forward pass
    print("Test forward pass:")
    x = torch.randn(64, 16, 1920).to(device)  # (batch, channels, timesteps)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected output shape: (64, 2)")
    print(f"  ✅ Match: {y.shape == (64, 2)}\n")
    
    # Test model save/load
    print("Test save/load:")
    model.save("/tmp/wesad_model.pth")
    loaded_model = WESADModel.load("/tmp/wesad_model.pth", device=device)
    print(f"✅ Model saved and loaded successfully\n")
    
    print("✅ All tests passed!")