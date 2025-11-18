#!/usr/bin/env python3
"""
Federated Learning + Differential Privacy Trainer.

Combines FL (distributed training) with DP (privacy at client-level).
Each client trains locally with differential privacy, then aggregates
via standard FedAvg at the server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

from opacus import PrivacyEngine
from src.training.base_trainer import BaseTrainer
from src.privacy.fl_client import FLClient
from src.privacy.fl_server import FLServer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DPConfig:
    """Configuration for Differential Privacy in FL."""

    def __init__(self, config: Dict[str, Any]):
        dp_cfg = config.get('differential_privacy', {})
        self.enabled = bool(dp_cfg.get('enabled', False))
        self.noise_multiplier = float(dp_cfg.get('noise_multiplier', 1.0))
        self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        self.delta = float(dp_cfg.get('delta', 1e-5))
        logger.info(
            "DPConfig loaded: enabled=%s, noise_mult=%.4f",
            self.enabled,
            self.noise_multiplier,
        )

    def __repr__(self) -> str:
        return (
            f"DPConfig(enabled={self.enabled}, "
            f"noise_mult={self.noise_multiplier}, "
            f"max_grad={self.max_grad_norm}, delta={self.delta})"
        )


class FLDPClient(FLClient):
    """FL Client with Differential Privacy support."""

    def __init__(self, client_id: str, model: nn.Module,
                 train_loader: DataLoader, config: Dict[str, Any],
                 device: str = 'cuda',
                 dp_config: Optional[DPConfig] = None):
        """
        Initialize FL DP Client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Local training data loader
            config: Configuration dictionary
            device: Device to use (cpu/cuda)
            dp_config: Differential privacy config
        """
        self.dp_config = dp_config or DPConfig(config)
        self.privacy_engine = None
        self.dp_train_loader = None

        # Manual init to avoid double optimizer creation
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)

        fl_cfg = config.get('federated_learning', {})
        self.local_epochs = fl_cfg.get('local_epochs', 1)

        # Initialize as None
        self.optimizer = None
        self.scheduler = None

        # Setup criterion with class weights if available
        self._setup_criterion()

        # Setup DP (creates/wraps optimizer)
        self._setup_dp()

        # NOW setup scheduler (optimizer exists now)
        self.setup_scheduler()

        logger.info(
            "Client %s initialized: DP=%s, privacy_engine=%s",
            self.client_id,
            self.dp_config.enabled,
            self.privacy_engine is not None,
        )

    def _setup_criterion(self) -> None:
        """Setup loss criterion with class weights if available."""
        cfg = self.config['training']
        dataset_cfg = self.config.get('dataset', {})

        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        use_class_weights = dataset_cfg.get('use_class_weights', False)

        class_weights = None
        if use_class_weights:
            if 'class_weights' in dataset_cfg:
                weights_dict = dataset_cfg['class_weights']
                n_classes = dataset_cfg.get('n_classes', 2)
                class_weights = torch.zeros(n_classes, dtype=torch.float32)
                for class_idx, weight in weights_dict.items():
                    class_weights[int(class_idx)] = float(weight)
                class_weights = class_weights.to(self.device)
                logger.warning(
                    "[CLASS WEIGHTS] Client %s: USING class weights: %s",
                    self.client_id,
                    class_weights.tolist()
                )
            else:
                logger.error(
                    "[CLASS WEIGHTS] Client %s: use_class_weights=True but 'class_weights' NOT in dataset_cfg!",
                    self.client_id
                )
        else:
            logger.warning(
                "[CLASS WEIGHTS] Client %s: use_class_weights=False (no class weights)",
                self.client_id
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction='mean'
        ).to(self.device)

    def _setup_dp(self) -> None:
        """Setup Opacus PrivacyEngine if DP enabled."""
        logger.info(
            "Client %s: _setup_dp() called, DP enabled=%s",
            self.client_id,
            self.dp_config.enabled,
        )

        if not self.dp_config.enabled:
            # DP disabled - create normal optimizer
            self.setup_optimizer()
            logger.debug(
                "Client %s: DP disabled, using normal training",
                self.client_id
            )
            return

        logger.info(
            "Client %s: Setting up DP (noise_mult=%.4f, max_grad=%.2f)",
            self.client_id,
            self.dp_config.noise_multiplier,
            self.dp_config.max_grad_norm,
        )

        try:
            cfg = self.config['training']
            lr = float(cfg.get('learning_rate', 1e-3))
            weight_decay = float(cfg.get('weight_decay', 0.0))
            opt_name = cfg.get('optimizer', 'sgd').lower()

            # Create optimizer based on config (like DPTrainer)
            if opt_name == 'adamw':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
                logger.info(
                    "Client %s: Using AdamW (lr=%.4f, wd=%.4f)",
                    self.client_id, lr, weight_decay
                )
            elif opt_name == 'adam':
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
                logger.info(
                    "Client %s: Using Adam (lr=%.4f, wd=%.4f)",
                    self.client_id, lr, weight_decay
                )
            else:
                momentum = float(cfg.get('momentum', 0.7))
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
                logger.info(
                    "Client %s: Using SGD (lr=%.4f, momentum=%.2f, wd=%.4f)",
                    self.client_id, lr, momentum, weight_decay
                )

            # Attach PrivacyEngine
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dp_train_loader = (
                privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=self.dp_config.noise_multiplier,
                    max_grad_norm=self.dp_config.max_grad_norm,
                    poisson_sampling=True,
                    clipping_mode='flat',
                    grad_sample_mode='hooks'
                )
            )
            self.privacy_engine = privacy_engine
            logger.info(
                "Client %s: PrivacyEngine ready",
                self.client_id
            )

        except Exception as e:
            logger.error(
                "Client %s: DP setup failed: %s",
                self.client_id,
                str(e)
            )
            raise
            
    def train_local(self) -> Dict[str, float]:
        """
        Train model locally for local_epochs with optional DP.

        Overrides parent to use DP-wrapped loader if available.

        Returns:
            Dictionary with local training metrics
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(self.local_epochs):
            self.model.train()

            # Use DP-wrapped loader if available, else regular loader
            loader = (
                self.dp_train_loader
                if self.dp_train_loader is not None
                else self.train_loader
            )

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach().item() * batch_x.size(0)
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)

        # Scheduler step (once per round, not per batch)
        if self.scheduler is not None and isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            self.scheduler.step()

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': (
                total_correct / total_samples if total_samples > 0 else 0.0
            ),
        }

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current model weights.

        Handles _module. prefix added by Opacus wrapping.

        Returns:
            Dict of weight tensors with normalized names
        """
        weights = {}
        for name, param in self.model.named_parameters():
            # Remove _module. prefix if present (from Opacus)
            clean_name = name[8:] if name.startswith('_module.') else name
            weights[clean_name] = param.data.clone()
        return weights

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Set model weights from dictionary.

        Handles _module. prefix mismatch between server and DP client.

        Args:
            weights: Dictionary of weight tensors
        """
        for name, param in self.model.named_parameters():
            # Try exact match first
            if name in weights:
                param.data = weights[name].clone().to(self.device)
            else:
                # Try matching with normalized name (without _module.)
                clean_name = (
                    name[8:] if name.startswith('_module.') else name
                )
                if clean_name in weights:
                    param.data = (
                        weights[clean_name].clone().to(self.device)
                    )

    def get_epsilon(self) -> Optional[float]:
        """
        Get current epsilon from PrivacyEngine.

        Returns:
            Epsilon value or None if DP disabled
        """
        if self.privacy_engine is None:
            return None
        try:
            epsilon = float(
                self.privacy_engine.get_epsilon(self.dp_config.delta)
            )
            return epsilon
        except Exception:
            return None


class FLDPTrainer(BaseTrainer):
    """Federated Learning + Differential Privacy Trainer."""

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cuda'):
        super().__init__(model, config, device)
        self.dp_config = DPConfig(config)
        self.clients = []
        self.server = None
        self.best_model_state = None
        logger.info(f"FLDPTrainer initialized: {self.dp_config}")

    def setup_optimizer_and_loss(self) -> None:
        """Setup loss function."""
        cfg = self.config['training']
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
        logger.info("Loss function initialized")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Not used in FL context."""
        raise NotImplementedError(
            "FLDPTrainer uses federated rounds, not epochs. "
            "Use fit() instead."
        )

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation with metrics."""
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

        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        return {
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
            'class_names': self.config['dataset'].get('class_names', [])
        }

    def fit(self, train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 40,
            patience: int = 15,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with FL + DP.

        Each client trains locally with DP, then aggregates at server
        via standard FedAvg.

        Args:
            train_loaders: List of training loaders (one per client)
            val_loaders: List of validation loaders (one per client)
            client_ids: List of client IDs
            epochs: Number of global rounds
            patience: Early stopping patience (in validation rounds)
            output_dir: Directory to save checkpoints

        Returns:
            Dictionary with training results and privacy metrics
        """
        self.setup_optimizer_and_loss()
        self._reset_training_state()

        fl_cfg = self.config.get('federated_learning', {})
        validation_frequency = fl_cfg.get('validation_frequency', 5)
        local_epochs = fl_cfg.get('local_epochs', 1)

        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Create DP-enabled clients
        logger.info(f"Creating {len(client_ids)} FL-DP clients...")
        # Debug: Check if class weights are in config
        dataset_cfg = self.config.get('dataset', {})
        logger.warning(
            "[DEBUG] Config check - use_class_weights: %s, class_weights in config: %s",
            dataset_cfg.get('use_class_weights', False),
            'class_weights' in dataset_cfg
        )
        if 'class_weights' in dataset_cfg:
            logger.warning(
                "[DEBUG] Config class_weights: %s",
                dataset_cfg['class_weights']
            )
        self.clients = [
            FLDPClient(
                client_id=cid,
                model=type(self.model)(
                    self.config, device=str(self.device)
                ),
                train_loader=tl,
                config=self.config,
                device=str(self.device),
                dp_config=self.dp_config
            )
            for cid, tl in zip(client_ids, train_loaders)
        ]

        self.server = FLServer(
            model=self.model,
            clients=self.clients,
            config=self.config,
            device=str(self.device)
        )

        start_time = time.time()

        # Print header
        print(f"\n{'='*70}")
        print("FEDERATED LEARNING + DIFFERENTIAL PRIVACY TRAINING")
        print(f"{'='*70}")
        print(f"  Clients: {len(self.clients)}")
        print(f"  Global rounds: {epochs}")
        print(f"  Local epochs per round: {local_epochs}")
        print(f"  Validation frequency: every {validation_frequency} rounds")
        print(f"  Early stopping patience: {patience} validation rounds")
        print(f"\n  Differential Privacy:")
        print(f"    Enabled: {self.dp_config.enabled}")
        if self.dp_config.enabled:
            print(f"    Noise multiplier: {self.dp_config.noise_multiplier}")
            print(f"    Max grad norm: {self.dp_config.max_grad_norm}")
            print(f"    Delta: {self.dp_config.delta}")
        print(f"{'='*70}\n")

        best_epoch = 0
        round_num = 0
        epsilon_history = []

        for round_num in range(1, epochs + 1):
            # FL round
            train_metrics = self.server.train_round()
            train_loss = train_metrics['loss']
            train_acc = train_metrics['accuracy']

            should_validate = (
                (round_num % validation_frequency == 0) or
                (round_num == epochs)
            )

            # Get epsilon only during validation rounds (to save computation time)
            epsilon = None
            if should_validate and self.dp_config.enabled and len(self.clients) > 0:
                try:
                    epsilon = self.clients[0].get_epsilon()
                    if epsilon is not None:
                        epsilon_history.append(epsilon)
                except Exception as e:
                    logger.warning(f"Failed to compute epsilon: {e}")
                    epsilon = None

            if should_validate:
                val_metrics = self.server.evaluate_on_clients(val_loaders)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']

                self.history['epoch'].append(round_num)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Print with epsilon
                eps_str = f"ε={epsilon:.4f}" if epsilon else ""
                print(
                    f"Round {round_num:3d}: "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"{eps_str}",
                    flush=True
                )

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    best_epoch = round_num

                    # Store in memory ALWAYS
                    self.best_model_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }

                    if output_path:
                        self.save_checkpoint(output_path / 'best_model.pth')
                        logger.info(
                            f"Saved best model at round {round_num} "
                            f"(val_acc={val_acc:.4f})"
                        )
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= patience:
                        print(
                            f"\nEarly stopping at round {round_num} "
                            f"(patience={patience})",
                            flush=True
                        )
                        break

        elapsed = time.time() - start_time

        # Restore best model from memory or disk
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model from memory")
        elif output_path and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')
            logger.info("Loaded best model from disk")

        self.cleanup_memory()

        print(f"\n{'='*70}")
        print(f"Training completed: {round_num} rounds in {elapsed:.1f}s")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f} "
              f"(round {best_epoch})")
        if self.dp_config.enabled and epsilon_history:
            print(f"  Final epsilon: {epsilon_history[-1]:.4f}")
        print(f"{'='*70}\n", flush=True)

        results = {
            'total_rounds': round_num,
            'best_round': best_epoch,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': (
                self.history['val_loss'][-1]
                if self.history['val_loss'] else 0.0
            ),
            'final_val_acc': (
                self.history['val_acc'][-1]
                if self.history['val_acc'] else 0.0
            ),
            'history': self.history,
            'n_clients': len(self.clients),
            'local_epochs': local_epochs,
            'validation_frequency': validation_frequency
        }

        # Add DP metrics
        if self.dp_config.enabled:
            results['differential_privacy'] = {
                'enabled': True,
                'noise_multiplier': self.dp_config.noise_multiplier,
                'max_grad_norm': self.dp_config.max_grad_norm,
                'delta': self.dp_config.delta,
                'final_epsilon': (
                    epsilon_history[-1] if epsilon_history else None
                ),
                'epsilon_history': epsilon_history
            }

        return results