"""
Base trainer interface with common functionality.
All trainers must implement fit() and evaluate_full().
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base class for all trainers with common functionality."""
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Dict[str, Any], 
        device: str = "cpu",
        callback: Optional[Callable] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.callback = callback or (lambda *args, **kwargs: None)
        
        self.history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        # In-memory best model snapshot for early stopping
        self.best_model_state = None
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        """Core training loop. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate_full(self, test_loader) -> Dict[str, Any]:
        """Full metrics evaluation including fairness metrics."""
        pass
    
    def _reset_training_state(self):
        """Reset training history."""
        self.history = {k: [] for k in self.history}
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
    
    def validate(self, val_loader):
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = getattr(self, 'criterion', nn.CrossEntropyLoss())
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0
    
    def log(self, message: str):
        """Helper to send logs to UI via callback."""
        self.callback(progress=None, log=message)
