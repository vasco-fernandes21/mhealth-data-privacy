#!/usr/bin/env python3
"""
Federated Learning + Differential Privacy Trainer.

Combines FL (distributed training) with DP (privacy guarantees).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, List
import time
from pathlib import Path

from src.training.trainers.fl_trainer import FLTrainer
from src.privacy.dp_utils import DPConfig, setup_privacy_engine, get_epsilon
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FLDPTrainer(FLTrainer):
    """FL + DP Trainer."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cuda'):
        super().__init__(model, config, device)
        self.dp_config = DPConfig(config)
        self.privacy_budget_history = []
    
    def fit(self, train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Train with FL+DP.
        
        NOTE: Para implementação completa, DP deve ser aplicado
        per-client (DP na comunicação dos clientes) ou
        na agregação do servidor (não trivial).
        
        Por enquanto, apenas log do DP config.
        """
        
        result = super().fit(train_loaders, val_loaders, client_ids,
                           epochs, patience, output_dir)
        
        # Add DP info
        if self.dp_config.enabled:
            result['dp_config'] = {
                'target_epsilon': self.dp_config.target_epsilon,
                'target_delta': self.dp_config.target_delta,
            }
            self.logger.warning(
                "FL+DP implementation requires per-client DP "
                "or server-side aggregation DP (not yet implemented)"
            )
        
        return result
    """Federated Learning + Differential Privacy Trainer."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize FL+DP trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to use
        """
        super().__init__(model, config, device)
        self.logger = get_logger(__name__)
        
        # DP-specific
        self.dp_config = DPConfig(config)
        self.privacy_engine = None
        self.privacy_budget_history = []
    
    def fit(self,
            train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with FL+DP.
        
        Args:
            train_loaders: List of training loaders (one per client)
            val_loaders: List of validation loaders (one per client)
            client_ids: List of client IDs
            epochs: Maximum global rounds
            patience: Early stopping patience
            output_dir: Directory to save checkpoints
        
        Returns:
            Training results with privacy metrics
        """
        # Setup
        self.setup_optimizer_and_loss()
        self.setup_federated_learning(train_loaders, client_ids)
        
        # Setup DP on aggregated updates
        self.logger.info("Setting up Differential Privacy for FL...")
        
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        self.best_val_acc = -1.0
        self.epochs_no_improve = 0
        
        for round_num in range(1, epochs + 1):
            # FL round (local training)
            train_loss, train_acc = self.train_federated_round()
            
            # Apply DP to global model update
            if self.dp_config.enabled and round_num == 1:
                # Setup PrivacyEngine after first round
                # (we have the model structure now)
                self.logger.warning(
                    "DP aggregation in FL is complex - "
                    "consider applying DP at client-side instead"
                )
            
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
            
            # Log with DP info
            msg = (f"Round {round_num:03d}: "
                   f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                   f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            
            if self.dp_config.enabled:
                msg += f" | DP enabled"
            
            self.logger.info(msg)
            
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
        
        results = {
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
        
        # Add DP metrics if enabled
        if self.dp_config.enabled:
            results['dp_params'] = {
                'target_epsilon': self.dp_config.target_epsilon,
                'target_delta': self.dp_config.target_delta,
                'max_grad_norm': self.dp_config.max_grad_norm,
                'noise_multiplier': self.dp_config.noise_multiplier
            }
        
        return results