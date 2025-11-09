#!/usr/bin/env python
"""
Abstract base class for all models.

Defines:
- Common model interface
- Device management
- Serialization methods
- Parameter counting

All dataset-specific models inherit from BaseModel.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import json


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
            device: Device to use ('cuda', 'cpu', 'mps')
        """
        super().__init__()
        self.config = config
        self.device_name = device
        self._device = torch.device(device)
        
        # Extract common config
        self.dataset_name = config.get('dataset', {}).get('name', 'unknown')
        self.n_classes = config.get('dataset', {}).get('n_classes', )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        pass
    
    def to_device(self, device: Optional[str] = None) -> 'BaseModel':
        """
        Move model to device.
        
        Args:
            device: Target device (if None, uses self.device_name)
        
        Returns:
            Self (for chaining)
        """
        if device is not None:
            self.device_name = device
            self._device = torch.device(device)
        
        return self.to(self._device)
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model stats
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(total_params * 4 / (1024 ** 2)),  # Assume float32 (4 bytes per param)
            'dataset': self.dataset_name,
            'n_classes': self.n_classes,
            'device': self.device_name
        }
    
    def print_model_summary(self) -> None:
        """Print model summary to console."""
        info = self.get_model_info()
        print(f"\n{'='*0}")
        print(f"Model Summary: {self.__class__.__name__}")
        print(f"{'='*0}")
        print(f"Dataset: {info['dataset']}")
        print(f"Classes: {info['n_classes']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.f} MB")
        print(f"Device: {info['device']}")
        print(f"{'='*0}\n")
    
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, path)
        
        print(f" Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load on
        
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(config, device=device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state'])
        
        print(f" Model loaded from {path}")
        return model
    
    def freeze(self) -> None:
        """Freeze all parameters (no gradient updates)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters (allow gradient updates)."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze/unfreeze backbone (for transfer learning).
        
        Args:
            freeze: Whether to freeze
        
        Note:
            Subclasses should override if backbone structure differs
        """
        for param in self.parameters():
            param.requires_grad = not freeze
    
    def get_parameters_by_layer(self) -> Dict[str, int]:
        """
        Get parameter count by layer.
        
        Returns:
            Dictionary mapping layer names to parameter counts
        """
        layer_params = {}
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_params[name] = params
        
        return layer_params
    
    def count_flops(self, input_shape: tuple) -> int:
        """
        Estimate FLOPs (requires additional utilities).
        
        Args:
            input_shape: Expected input shape (without batch)
        
        Returns:
            Estimated FLOPs (rough estimate)
        
        Note:
            This is a placeholder - use fvcore or ptflops for accurate counting
        """
        # Placeholder - implement with fvcore if needed
        return 0
    
    def __repr__(self) -> str:
        """String representation of model."""
        info = self.get_model_info()
        return (
            f"{self.__class__.__name__}(\n"
            f"  dataset={info['dataset']},\n"
            f"  classes={info['n_classes']},\n"
            f"  params={info['total_parameters']:,},\n"
            f"  device={info['device']}\n"
            f")"
        )