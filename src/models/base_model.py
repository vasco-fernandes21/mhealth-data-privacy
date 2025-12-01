#!/usr/bin/env python
"""Base class for all models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional


class BaseModel(ABC, nn.Module):
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.device_name = device
        self._device = torch.device(device)
        self.dataset_name = config.get('dataset', {}).get('name', 'unknown')
        self.n_classes = config.get('dataset', {}).get('n_classes', 2)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def to_device(self, device: Optional[str] = None) -> 'BaseModel':
        if device is not None:
            self.device_name = device
            self._device = torch.device(device)
        return self.to(self._device)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(total_params * 4 / (1024 ** 2)),
            'dataset': self.dataset_name,
            'n_classes': self.n_classes,
            'device': self.device_name
        }
    
    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'BaseModel':
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model = cls(config, device=device)
        model.load_state_dict(checkpoint['model_state'])
        return model
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"{self.__class__.__name__}("
            f"dataset={info['dataset']}, "
            f"classes={info['n_classes']}, "
            f"params={info['total_parameters']:,}, "
            f"device={info['device']})"
        )