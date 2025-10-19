#!/usr/bin/env python3
"""
LSTM model for Sleep-EDF sleep stage classification.

Input: (batch, sequence_length, n_features) = (batch, 10, 24)
Output: (batch, n_classes) = (batch, 5)

Architecture:
- LSTM-based for temporal modeling
- Bidirectional LSTM for context
- Simple and efficient for DP compatibility
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_model import BaseModel


class SleepEDFModel(BaseModel):
    """LSTM model for Sleep-EDF dataset."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize Sleep-EDF model.
        
        Args:
            config: Configuration dictionary with:
                - dataset.n_features
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
        
        self.n_features = dataset_cfg['n_features']  # 24
        self.n_classes = dataset_cfg['n_classes']    # 5
        
        lstm_units = model_cfg['lstm_units']         # 128
        lstm_layers = model_cfg['lstm_layers']       # 2
        dropout = model_cfg['dropout']               # 0.3
        dense_layers = model_cfg['dense_layers']     # [64, 32]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Dimension after bidirectional LSTM
        lstm_output_size = lstm_units * 2  # 256
        
        # Dense layers
        dense_layers_list = []
        prev_size = lstm_output_size
        
        for dense_size in dense_layers:  # [64, 32]
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
            x: Input tensor of shape (batch, sequence_length, n_features)
               e.g., (32, 10, 24)
        
        Returns:
            Output tensor of shape (batch, n_classes)
            e.g., (32, 5)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use concatenation of last hidden states from both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # For bidirectional: (lstm_layers*2, batch, lstm_units)
        
        # Get last layer hidden states from both directions
        h_forward = h_n[-2]  # Last forward layer
        h_backward = h_n[-1]  # Last backward layer
        
        # Concatenate: (batch, lstm_units * 2)
        h_last = torch.cat([h_forward, h_backward], dim=1)
        
        # Dense layers
        output = self.dense(h_last)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'SleepEDFLSTM',
            'n_features': self.n_features,
            'lstm_config': {
                'units': self.lstm.hidden_size,
                'layers': self.lstm.num_layers,
                'bidirectional': self.lstm.bidirectional
            }
        })
        return info


def create_sleep_edf_model(config: Dict[str, Any], 
                          device: str = 'cpu') -> SleepEDFModel:
    """
    Factory function to create Sleep-EDF model.
    
    Args:
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Initialized model
    """
    model = SleepEDFModel(config, device)
    return model


if __name__ == "__main__":
    # Test script
    print("Testing Sleep-EDF model...\n")
    
    # Create config
    config = {
        'dataset': {
            'name': 'sleep-edf',
            'n_features': 24,
            'n_classes': 5,
            'sequence_length': 10
        },
        'model': {
            'lstm_units': 128,
            'lstm_layers': 2,
            'dropout': 0.3,
            'dense_layers': [64, 32]
        }
    }
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SleepEDFModel(config, device=device)
    
    # Print info
    model.print_model_summary()
    
    # Test forward pass
    print("Test forward pass:")
    x = torch.randn(32, 10, 24).to(device)  # (batch, sequence, features)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected output shape: (32, 5)")
    print(f"  ✅ Match: {y.shape == (32, 5)}\n")
    
    # Test model save/load
    print("Test save/load:")
    model.save("/tmp/sleep_edf_model.pth")
    loaded_model = SleepEDFModel.load("/tmp/sleep_edf_model.pth", device=device)
    print(f"✅ Model saved and loaded successfully\n")
    
    print("✅ All tests passed!")