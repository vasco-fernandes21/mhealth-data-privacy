#!/usr/bin/env python3
"""
Federated Learning Trainer.

Coordinates FL training across clients.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
from pathlib import Path

from src.training.base_trainer import BaseTrainer
from src.privacy.fl_client import FLClient
from src.privacy.fl_server import FLServer
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FLTrainer(BaseTrainer):
    """Federated Learning Trainer."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize FL trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
        """
        super().__init__(model, config, device)
        self.logger = get_logger(__name__)
        self.server = None
        self.clients = []
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup loss function (clients have their own optimizers)."""
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
    
    def setup_federated_learning(self,
                                train_loaders: List[DataLoader],
                                client_ids: List[str]) -> None:
        """
        Setup FL clients and server.
        
        Args:
            train_loaders: List of training loaders (one per client)
            client_ids: List of client IDs
        """
        # Create clients
        self.clients = []
        for client_id, train_loader in zip(client_ids, train_loaders):
            # Create fresh model for each client
            client_model = type(self.model)(self.config, device=str(self.device))
            
            client = FLClient(
                client_id=client_id,
                model=client_model,
                train_loader=train_loader,
                config=self.config,
                device=str(self.device)
            )
            self.clients.append(client)
        
        # Create server
        self.server = FLServer(
            model=self.model,
            clients=self.clients,
            config=self.config,
            device=str(self.device)
        )
        
        self.logger.info(f"FL setup complete: {len(self.clients)} clients")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Not used in FL - use train_federated_round instead.
        """
        raise NotImplementedError("Use train_federated_round for FL training")
    
    def train_federated_round(self) -> Tuple[float, float]:
        """
        Execute one federated learning round.
        
        Returns:
            (loss, accuracy) for the round
        """
        metrics = self.server.train_round()
        return metrics['avg_loss'], metrics['avg_accuracy']
    
    def fit(self,
            train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with federated learning.
        
        Args:
            train_loaders: List of training loaders (one per client)
            val_loaders: List of validation loaders (one per client)
            client_ids: List of client IDs
            epochs: Maximum global rounds
            patience: Early stopping patience
            output_dir: Directory to save checkpoints
        
        Returns:
            Training results
        """
        # Setup
        self.setup_optimizer_and_loss()
        self.setup_federated_learning(train_loaders, client_ids)
        
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = None
        import time
        start_time = time.time()
        
        self.best_val_acc = -1.0
        self.epochs_no_improve = 0
        
        for round_num in range(1, epochs + 1):
            # FL round
            train_loss, train_acc = self.train_federated_round()
            
            # Validation
            val_metrics = self.server.evaluate_on_clients(val_loaders)
            val_loss = val_metrics['avg_loss']
            val_acc = val_metrics['avg_accuracy']
            
            # Store history
            self.history['epoch'].append(round_num)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log
            self.logger.info(
                f"Round {round_num:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                
                if output_path is not None:
                    self.save_checkpoint(output_path / 'best_model.pth')
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= patience:
                self.logger.info(f"Early stopping after {round_num} rounds")
                break
        
        elapsed_time = time.time() - start_time
        
        if output_path is not None and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
        
        return {
            'total_rounds': round_num,
            'training_time_seconds': elapsed_time,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history,
            'n_clients': len(self.clients)
        }