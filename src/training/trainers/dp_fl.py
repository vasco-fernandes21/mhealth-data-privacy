#!/usr/bin/env python3
"""Federated Learning with Differential Privacy."""

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
    """Differential Privacy configuration."""

    def __init__(self, config: Dict[str, Any]):
        dp_cfg = config.get('differential_privacy', {})
        self.enabled = bool(dp_cfg.get('enabled', False))
        self.noise_multiplier = float(dp_cfg.get('noise_multiplier', 1.0))
        self.max_grad_norm = float(dp_cfg.get('max_grad_norm', 5.0))
        self.delta = float(dp_cfg.get('delta', 1e-5))


class FLDPClient(FLClient):
    """FL Client with optional Differential Privacy."""

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        dp_config: Optional[DPConfig] = None,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)
        self.dp_config = dp_config or DPConfig(config)

        fl_cfg = config.get('federated_learning', {})
        self.local_epochs = fl_cfg.get('local_epochs', 1)

        self.optimizer = None
        self.privacy_engine = None
        self.dp_train_loader = None

        self._setup_criterion()
        self._setup_dp() if self.dp_config.enabled else self.setup_optimizer()
        self.setup_scheduler()

    def _setup_criterion(self) -> None:
        """Setup loss with optional class weights."""
        cfg = self.config['training']
        dataset_cfg = self.config.get('dataset', {})

        class_weights = None
        if dataset_cfg.get('use_class_weights', False):
            if 'class_weights' in dataset_cfg:
                weights_dict = dataset_cfg['class_weights']
                n_classes = dataset_cfg.get('n_classes', 2)
                class_weights = torch.zeros(n_classes, dtype=torch.float32)
                for class_idx, weight in weights_dict.items():
                    class_weights[int(class_idx)] = float(weight)
                class_weights = class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(cfg.get('label_smoothing', 0.0)),
            reduction='mean',
        ).to(self.device)

    def _setup_dp(self) -> None:
        """Setup Opacus PrivacyEngine."""
        cfg = self.config['training']
        lr = float(cfg.get('learning_rate', 1e-3))
        weight_decay = float(cfg.get('weight_decay', 0.0))
        opt_name = cfg.get('optimizer', 'sgd').lower()

        optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}

        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(), **optimizer_kwargs
            )
        elif opt_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), **optimizer_kwargs
            )
        else:
            optimizer_kwargs['momentum'] = float(
                cfg.get('momentum', 0.7)
            )
            optimizer = torch.optim.SGD(
                self.model.parameters(), **optimizer_kwargs
            )

        privacy_engine = PrivacyEngine()
        try:
            self.model, self.optimizer, self.dp_train_loader = (
                privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=self.dp_config.noise_multiplier,
                    max_grad_norm=self.dp_config.max_grad_norm,
                    poisson_sampling=False,
                    grad_sample_mode='hooks',
                )
            )
            self.privacy_engine = privacy_engine
        except Exception as e:
            logger.error(
                f"Client {self.client_id}: PrivacyEngine setup failed: {e}"
            )
            raise

    def train_local(self) -> Dict[str, float]:
        """Train locally for local_epochs."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loader = (
            self.dp_train_loader
            if self.dp_train_loader is not None
            else self.train_loader
        )

        for _ in range(self.local_epochs):
            self.model.train()

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

        if self.scheduler is not None and isinstance(
            self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            self.scheduler.step()

        return {
            'loss': (
                total_loss / total_samples if total_samples > 0 else 0.0
            ),
            'accuracy': (
                total_correct / total_samples if total_samples > 0 else 0.0
            ),
        }

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights, handling Opacus _module prefix."""
        weights = {}
        for name, param in self.model.named_parameters():
            clean_name = name[8:] if name.startswith('_module.') else name
            weights[clean_name] = param.data.clone()
        return weights

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights with flexible name matching."""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone().to(self.device)
            else:
                clean_name = (
                    name[8:] if name.startswith('_module.') else name
                )
                if clean_name in weights:
                    param.data = (
                        weights[clean_name].clone().to(self.device)
                    )

    def get_epsilon(self) -> Optional[float]:
        """Get current epsilon from PrivacyEngine."""
        if self.privacy_engine is None:
            return None

        try:
            epsilon = float(
                self.privacy_engine.get_epsilon(self.dp_config.delta)
            )
            if isinstance(epsilon, (int, float)):
                if epsilon != epsilon or epsilon < 0 or epsilon == float('inf'):
                    return None
                return epsilon
            return None
        except Exception:
            return None


class FLDPTrainer(BaseTrainer):
    """Federated Learning + Differential Privacy trainer."""

    def __init__(
        self, model: nn.Module, config: Dict[str, Any],
        device: str = 'cuda'
    ):
        super().__init__(model, config, device)
        self.dp_config = DPConfig(config)
        self.clients = []
        self.server = None
        self.best_model_state = None

    def setup_optimizer_and_loss(self) -> None:
        cfg = self.config['training']
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.get('label_smoothing', 0.0))
        ).to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        raise NotImplementedError("Use fit() for federated training.")

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation with all metrics."""
        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)

                y_true_list.append(batch_y.cpu())
                y_pred_list.append(predicted.cpu())

        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
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
                    y_true,
                    y_pred,
                    average='weighted',
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            'recall': float(
                recall_score(
                    y_true,
                    y_pred,
                    average='weighted',
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            'f1_score': float(
                f1_score(
                    y_true,
                    y_pred,
                    average='weighted',
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_names': self.config['dataset'].get('class_names', []),
        }

    def fit(
        self,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        client_ids: List[str],
        epochs: int = 40,
        patience: int = 15,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Federated training with DP.

        Args:
            train_loaders: Training loaders (one per client)
            val_loaders: Validation loaders (one per client)
            client_ids: Client identifiers
            epochs: Global rounds
            patience: Early stopping patience
            output_dir: Checkpoint directory

        Returns:
            Training results with privacy metrics
        """
        self.setup_optimizer_and_loss()
        self._reset_training_state()

        fl_cfg = self.config.get('federated_learning', {})
        validation_frequency = fl_cfg.get('validation_frequency', 5)
        local_epochs = fl_cfg.get('local_epochs', 1)

        output_path = (
            Path(output_dir) if output_dir else None
        )
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        self.clients = [
            FLDPClient(
                client_id=cid,
                model=type(self.model)(
                    self.config, device=str(self.device)
                ),
                train_loader=tl,
                config=self.config,
                device=str(self.device),
                dp_config=self.dp_config,
            )
            for cid, tl in zip(client_ids, train_loaders)
        ]

        self.server = FLServer(
            model=self.model,
            clients=self.clients,
            config=self.config,
            device=str(self.device),
        )

        self._print_header(
            len(self.clients), epochs, local_epochs,
            validation_frequency, patience
        )

        start_time = time.time()
        best_epoch = 0
        epsilon_history = []

        for round_num in range(1, epochs + 1):
            train_metrics = self.server.train_round()

            epsilon = self._get_epsilon()
            if epsilon is not None:
                epsilon_history.append(epsilon)
            elif epsilon_history:
                epsilon = epsilon_history[-1]

            should_validate = (
                (round_num % validation_frequency == 0)
                or (round_num == epochs)
            )

            if should_validate:
                val_metrics = self.server.evaluate_on_clients(val_loaders)

                self.history['epoch'].append(round_num)
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])

                self._print_round(
                    round_num,
                    train_metrics['loss'],
                    train_metrics['accuracy'],
                    val_metrics['loss'],
                    val_metrics['accuracy'],
                    epsilon,
                )

                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_no_improve = 0
                    best_epoch = round_num

                    self.best_model_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }

                    if output_path:
                        self.save_checkpoint(output_path / 'best_model.pth')
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= patience:
                        print(
                            f"\nEarly stopping at round {round_num} "
                            f"(patience={patience})"
                        )
                        break

        elapsed = time.time() - start_time

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        elif output_path and (output_path / 'best_model.pth').exists():
            self.load_checkpoint(output_path / 'best_model.pth')

        if self.dp_config.enabled and len(self.clients) > 0:
            try:
                final_epsilon = self.clients[0].get_epsilon()
                if final_epsilon is not None:
                    if not epsilon_history or epsilon_history[-1] != final_epsilon:
                        epsilon_history.append(final_epsilon)
                elif epsilon_history:
                    final_epsilon = epsilon_history[-1]
            except Exception:
                if epsilon_history:
                    final_epsilon = epsilon_history[-1]

        self.cleanup_memory()

        self._print_summary(round_num, best_epoch, elapsed, epsilon_history)

        return self._build_results(
            round_num,
            best_epoch,
            elapsed,
            epsilon_history,
            len(self.clients),
            local_epochs,
            validation_frequency,
        )

    def _get_epsilon(self) -> Optional[float]:
        """Get epsilon from first client (same as DPTrainer)."""
        if not self.dp_config.enabled:
            return None
        
        if len(self.clients) == 0:
            return None

        client = self.clients[0]
        if client.privacy_engine is None:
            return None

        try:
            epsilon = client.get_epsilon()
            if epsilon is not None and isinstance(epsilon, (int, float)) and 0 <= epsilon < float('inf'):
                return epsilon
        except Exception:
            pass

        return None

    def _print_header(
        self,
        n_clients: int,
        epochs: int,
        local_epochs: int,
        val_freq: int,
        patience: int,
    ) -> None:
        print(f"\n{'='*70}")
        print("FL + DP TRAINING")
        print(f"{'='*70}")
        print(f"  Clients: {n_clients}")
        print(f"  Rounds: {epochs}")
        print(f"  Local epochs/round: {local_epochs}")
        print(f"  Validation frequency: {val_freq}")
        print(f"  Early stopping patience: {patience}")
        if self.dp_config.enabled:
            print(f"\n  DP:")
            print(f"    Noise multiplier: {self.dp_config.noise_multiplier}")
            print(f"    Max grad norm: {self.dp_config.max_grad_norm}")
            print(f"    Delta: {self.dp_config.delta}")
        print(f"{'='*70}\n")

    def _print_round(
        self,
        round_num: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epsilon: Optional[float],
    ) -> None:
        eps_str = f"ε={epsilon:.4f}" if epsilon is not None else "ε=N/A"
        print(
            f"Round {round_num:3d}: "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{eps_str}",
            flush=True,
        )

    def _print_summary(
        self,
        total_rounds: int,
        best_round: int,
        elapsed: float,
        epsilon_history: List[float],
    ) -> None:
        print(f"\n{'='*70}")
        print(f"Completed: {total_rounds} rounds in {elapsed:.1f}s")
        print(f"  Best accuracy: {self.best_val_acc:.4f} (round {best_round})")
        if self.dp_config.enabled and epsilon_history:
            print(f"  Final epsilon: {epsilon_history[-1]:.4f}")
        print(f"{'='*70}\n", flush=True)

    def _build_results(
        self,
        total_rounds: int,
        best_round: int,
        elapsed: float,
        epsilon_history: List[float],
        n_clients: int,
        local_epochs: int,
        validation_frequency: int,
    ) -> Dict[str, Any]:
        model_size_bytes = sum(
            p.numel() * 4 for p in self.model.parameters()
        )

        results = {
            'total_rounds': total_rounds,
            'best_round': best_round,
            'epochs_no_improve': self.epochs_no_improve,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': n_clients,
            'local_epochs': local_epochs,
            'validation_frequency': validation_frequency,
            'communication': {
                'total_rounds': total_rounds,
                'model_size_bytes': model_size_bytes,
                'total_communication_bytes': (
                    model_size_bytes * total_rounds * n_clients
                ),
            },
        }

        if self.dp_config.enabled:
            results['differential_privacy'] = {
                'enabled': True,
                'noise_multiplier': self.dp_config.noise_multiplier,
                'max_grad_norm': self.dp_config.max_grad_norm,
                'delta': self.dp_config.delta,
                'final_epsilon': epsilon_history[-1] if epsilon_history else None,
                'epsilon_history': epsilon_history,
            }

        return results