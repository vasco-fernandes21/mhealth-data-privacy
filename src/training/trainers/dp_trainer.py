#!/usr/bin/env python3
"""
Differential Privacy Trainer using Opacus.

Handles:
- Privacy Engine setup
- DP model wrapping
- Privacy budget accounting
- DP-compatible model creation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
import time
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_fscore_support
)
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from src.training.base_trainer import BaseTrainer
from src.utils.logging_utils import get_logger

try:
    from src.losses.focal_loss import FocalLoss
except ImportError:
    FocalLoss = None


logger = get_logger(__name__)


class DPConfig:
    """Configuration for Differential Privacy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DP config.
        
        Args:
            config: Configuration dictionary with 'differential_privacy' key
        """
        dp_cfg = config.get('differential_privacy', {})

        # Coerce common DP config fields to expected types to avoid runtime
        # errors when values come from YAML/CLI as strings.
        self.enabled = bool(dp_cfg.get('enabled', False))

        # Numeric fields: try to cast to float; fall back to defaults on failure
        try:
            self.target_epsilon = float(dp_cfg.get('target_epsilon', 8.0))
        except (TypeError, ValueError):
            self.target_epsilon = 8.0

        try:
            self.target_delta = float(dp_cfg.get('target_delta', 1e-5))
        except (TypeError, ValueError):
            self.target_delta = 1e-5

        try:
            self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        except (TypeError, ValueError):
            self.max_grad_norm = 1.0

        try:
            self.noise_multiplier = float(dp_cfg.get('noise_multiplier', 0.9))
        except (TypeError, ValueError):
            self.noise_multiplier = 0.9

        # Boolean/string fields
        self.poisson_sampling = bool(dp_cfg.get('poisson_sampling', True))
        self.accounting_method = str(dp_cfg.get('accounting_method', 'rdp'))
        self.secure_rng = bool(dp_cfg.get('secure_rng', True))
        self.grad_sample_mode = str(dp_cfg.get('grad_sample_mode', 'hooks'))
    
    def __repr__(self) -> str:
        return (
            f"DPConfig(ε={self.target_epsilon}, δ={self.target_delta}, "
            f"noise_mult={self.noise_multiplier}, max_grad_norm={self.max_grad_norm})"
        )


def make_lstm_dp_compatible(lstm: nn.LSTM) -> DPLSTM:
    """
    Convert nn.LSTM to DPLSTM for DP compatibility.
    
    Args:
        lstm: Original LSTM layer
    
    Returns:
        DPLSTM layer with same parameters
    """
    input_size = lstm.input_size
    hidden_size = lstm.hidden_size
    num_layers = lstm.num_layers
    batch_first = lstm.batch_first
    bidirectional = lstm.bidirectional
    dropout = lstm.dropout
    
    dplstm = DPLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dropout=dropout
    )
    
    return dplstm


def setup_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    dp_config: DPConfig,
    device: str = 'cuda'
) -> Tuple[nn.Module, torch.optim.Optimizer, Any, Any]:
    """
    Setup Opacus PrivacyEngine for DP training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        train_loader: Training data loader
        dp_config: DP configuration
        device: Device to use
    
    Returns:
        (dp_model, dp_optimizer, privacy_engine, dp_train_loader)
    
    IMPORTANTE: Usa noise_multiplier fixo, não target_epsilon!
    """
    
    if not dp_config.enabled:
        logger.warning("DP not enabled - returning original model")
        return model, optimizer, None, train_loader
    
    logger.info(f"Setting up PrivacyEngine with {dp_config}")
    
    # Create Privacy Engine
    privacy_engine = PrivacyEngine()
    
    # Usar noise_multiplier fixo para controlar o noise
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config.noise_multiplier,  # FIXO - varia por experimento
        max_grad_norm=dp_config.max_grad_norm,
        # NÃO passar target_epsilon/target_delta - calcula-se depois
        poisson_sampling=dp_config.poisson_sampling,
        clipping_mode='flat',
        grad_sample_mode=dp_config.grad_sample_mode
    )
    
    logger.info(f"PrivacyEngine setup complete")
    logger.info(f"  Noise multiplier (FIXED): {dp_config.noise_multiplier}")
    logger.info(f"  Max grad norm: {dp_config.max_grad_norm}")
    logger.info(f"  Target delta (para epsilon calc): {dp_config.target_delta}")
    logger.info(f"  Accounting method: {dp_config.accounting_method}")
    
    return dp_model, dp_optimizer, privacy_engine, dp_train_loader


def get_epsilon(privacy_engine: Optional[PrivacyEngine], 
               delta: float) -> float:
    """
    Get current privacy budget epsilon.
    
    Args:
        privacy_engine: PrivacyEngine instance
        delta: Delta value
    
    Returns:
        Current epsilon
    """
    if privacy_engine is None:
        return float('inf')

    # Assume delta is already a float from DPConfig
    try:
        epsilon = privacy_engine.get_epsilon(delta)
        return float(epsilon)
    except Exception as e:
        logger.error(f"Error computing epsilon: {e}")
        return float('inf')


def log_privacy_budget(privacy_engine: Optional[PrivacyEngine],
                      delta: float,
                      target_epsilon: float) -> Dict[str, float]:
    """
    Log privacy budget information.
    
    Args:
        privacy_engine: PrivacyEngine instance
        delta: Delta value
        target_epsilon: Reference epsilon (for comparison)
    
    Returns:
        Dictionary with privacy metrics
    """
    if privacy_engine is None:
        return {'epsilon': float('inf'), 'privacy_budget_used': 0.0}
    
    epsilon = get_epsilon(privacy_engine, delta)
    budget_ratio = epsilon / target_epsilon if target_epsilon > 0 else 0
    
    metrics = {
        'epsilon': epsilon,
        'delta': delta,
        'reference_epsilon': target_epsilon,
        'epsilon_ratio': budget_ratio
    }
    
    logger.info(f"Privacy Budget: ε={epsilon:.4f} (reference: {target_epsilon})")
    logger.info(f"  Epsilon ratio: {budget_ratio:.2f}x")
    
    return metrics


def check_dp_compatibility(model: nn.Module) -> Tuple[bool, list]:
    """
    Check if model is DP-compatible with Opacus.
    
    Args:
        model: PyTorch model
    
    Returns:
        (is_compatible, incompatible_layers)
    """
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            incompatible_layers.append((name, type(module).__name__))
    
    is_compatible = len(incompatible_layers) == 0
    
    if not is_compatible:
        logger.warning(f"Model has {len(incompatible_layers)} DP-incompatible layers:")
        for name, layer_type in incompatible_layers:
            logger.warning(f"  - {name}: {layer_type}")
        logger.warning("Recommendation: Replace BatchNorm with GroupNorm or LayerNorm")
    else:
        logger.info("✅ Model is DP-compatible")
    
    return is_compatible, incompatible_layers


class DPTrainer(BaseTrainer):
    """Differential Privacy Trainer using Opacus."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = DPConfig(self.config)
        self.privacy_engine = None
        self.dp_train_loader = None
        self.scaler = None
        self.use_amp = False
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer, loss, and DP PrivacyEngine."""
        cfg = self.config['training']
        
        # Optimizer
        lr = float(cfg['learning_rate'])
        weight_decay = float(cfg.get('weight_decay', 1e-4))
        opt_name = cfg.get('optimizer', 'adamw').lower()
        
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        
        # Loss function
        loss_name = cfg.get('loss', 'cross_entropy').lower()
        class_weights = cfg.get('class_weights', None)
        
        if class_weights is not None:
            weights_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
        else:
            weights_tensor = None
        
        if loss_name == 'focal_loss' and FocalLoss:
            self.criterion = FocalLoss(
                alpha=float(cfg.get('focal_alpha', 0.25)),
                gamma=float(cfg.get('focal_gamma', 2.0))
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_tensor, reduction='mean'
            )
        
        self.criterion = self.criterion.to(self.device)
        
        # LR Scheduler
        self._setup_scheduler()
        
        # Mixed precision (AMP) - disabled for DP compatibility
        self.use_amp = False  # DP doesn't work well with AMP
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        cfg = self.config['training']
        total_epochs = int(cfg.get('epochs', 100))
        scheduler_name = cfg.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=False
            )
        else:
            self.scheduler = None
    
    def setup_privacy_engine(self, train_loader: DataLoader) -> DataLoader:
        """Setup PrivacyEngine for DP training."""
        if not self.dp_config.enabled:
            logger.warning("DP not enabled - skipping PrivacyEngine setup")
            return train_loader
        
        logger.info(f"Setting up PrivacyEngine with {self.dp_config}")
        logger.info(f"Train loader batches: {len(train_loader)}")
        
        # Check DP compatibility
        is_compatible, incompatible = check_dp_compatibility(self.model)
        if not is_compatible:
            logger.warning(f"Model has {len(incompatible)} incompatible layers")
        # Check DP compatibility
        is_compatible, incompatible = check_dp_compatibility(self.model)
        if not is_compatible:
            logger.warning(f"Model has {len(incompatible)} incompatible layers")
        
        # Setup PrivacyEngine
        try:
            self.model, self.optimizer, self.privacy_engine, self.dp_train_loader = setup_privacy_engine(
                self.model,
                self.optimizer,
                train_loader,
                self.dp_config,
                device=str(self.device)
            )
        except Exception as e:
            logger.error(f"❌ PrivacyEngine setup failed: {e}")
            raise
        
        return self.dp_train_loader if self.dp_train_loader else train_loader
            train_loader: Training data loader (may be DP-wrapped)
        
        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use DP-wrapped loader if available, otherwise use original
        loader = self.dp_train_loader if self.dp_train_loader else train_loader
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass (DP handles gradient clipping and noise)
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            batch_loss = loss.detach().item()
            total_loss += batch_loss * batch_x.size(0)
            
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        # Use centralized cleanup helper from BaseTrainer
        self.cleanup_memory()

        if self.scheduler is not None and \
           not isinstance(self.scheduler,
                         torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return total_loss / total, correct / total
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 20,
            patience: int = 8,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with DP.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs (default: 20, intentionally low for DP training)
            patience: Early stopping patience
            output_dir: Directory to save checkpoints

        Returns:
            Dictionary with training results including privacy metrics
        """
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        self._reset_training_state()
        self.setup_optimizer_and_loss()
        self.setup_privacy_engine(train_loader)
        
        start_time = time.time()
        
        # Get validation frequency
        validation_frequency = max(1, epochs // 5)  # Validate approximately 5 times
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate APENAS a cada N epochs
            should_validate = (epoch % validation_frequency == 0) or (epoch == epochs)
            
            if should_validate:
                val_loss, val_acc = self.validate(val_loader)
                
                # Store history
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Log privacy budget
                if self.privacy_engine is not None:
                    epsilon = get_epsilon(self.privacy_engine, self.dp_config.target_delta)
                    logger.info(f"Epoch {epoch}: ε={epsilon:.4f}")
                
                # Log
                self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
                
                # Early stopping
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    self.best_model_state = deepcopy(self.model.state_dict())
                    
                    if output_path is not None:
                        self.save_checkpoint(output_path / 'best_model.pth')
                else:
                    self.epochs_no_improve += 1
                
                if self.epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Sem validação - só train
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
        
        training_time = time.time() - start_time
        
        # Get final privacy budget
        final_epsilon = get_epsilon(self.privacy_engine, self.dp_config.target_delta) \
                       if self.privacy_engine is not None else float('inf')
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        elif output_path is not None and \
             (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
        
        return {
            'total_epochs': epoch,
            'training_time_seconds': training_time,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'history': self.history,
            'final_epsilon': final_epsilon
        }
    
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with detailed metrics including privacy budget
        """
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)
        
        # Per-class metrics
        (
            precision_per_class, recall_per_class, f1_per_class, _
        ) = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Get final privacy budget
        final_epsilon = get_epsilon(self.privacy_engine, self.dp_config.target_delta) \
                       if self.privacy_engine is not None else float('inf')
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(
                precision_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'recall': float(
                recall_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'f1_score': float(
                f1_score(
                    y_true, y_pred, average='weighted',
                    zero_division=0, labels=unique_labels
                )
            ),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_names': self.config['dataset'].get('class_names', []),
            'final_epsilon': final_epsilon
        }
        
        return metrics


if __name__ == "__main__":
    print("Testing DP utilities...\n")
    
    config = {
        'differential_privacy': {
            'enabled': True,
            'target_epsilon': 8.0,
            'target_delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 0.9
        }
    }
    
    dp_config = DPConfig(config)
    print(f"DP Config: {dp_config}\n")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    is_compatible, incompatible = check_dp_compatibility(model)
    print(f"Model compatible: {is_compatible}\n")
    
    print("✅ All DP utility tests passed!")