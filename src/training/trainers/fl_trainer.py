#!/usr/bin/env python3
"""
Federated Learning Trainer.

Implements federated learning with multiple clients and a central server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Tuple
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

from src.training.base_trainer import BaseTrainer
from src.privacy.fl_client import FLClient
from src.privacy.fl_server import FLServer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FLTrainer(BaseTrainer):
    """Federated Learning trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_state = None

    def setup_optimizer_and_loss(self) -> None:
        """Setup loss (clients manage optimizers)."""
        cfg = self.config['training']
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
        logger.info("Loss function initialized")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch (not used in FL).

        FL uses federated rounds instead of epochs.
        """
        raise NotImplementedError(
            "FLTrainer uses federated rounds, not epochs. "
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
            epochs: int = 100,
            patience: int = 10,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with Federated Learning.

        Args:
            train_loaders: List of training loaders (one per client)
            val_loaders: List of validation loaders (one per client)
            client_ids: List of client identifiers
            epochs: Number of global rounds
            patience: Early stopping patience
            output_dir: Directory to save best model

        Returns:
            Dictionary with training results
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

        clients = [
            FLClient(
                client_id=cid,
                model=type(self.model)(
                    self.config, device=str(self.device)
                ),
                train_loader=tl,
                config=self.config,
                device=str(self.device)
            )
            for cid, tl in zip(client_ids, train_loaders)
        ]

        server = FLServer(
            model=self.model,
            clients=clients,
            config=self.config,
            device=str(self.device)
        )

        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING TRAINING")
        print(f"{'='*70}")
        print(f"  Clients: {len(clients)}")
        print(f"  Global rounds: {epochs}")
        print(f"  Local epochs per round: {local_epochs}")
        print(f"  Validation frequency: every {validation_frequency} rounds")
        print(f"  Early stopping patience: {patience} validation rounds")
        print(f"{'='*70}\n")

        best_epoch = 0
        round_num = 0

        for round_num in range(1, epochs + 1):
            train_metrics = server.train_round()
            train_loss = train_metrics['loss']
            train_acc = train_metrics['accuracy']

            should_validate = (
                (round_num % validation_frequency == 0) or
                (round_num == epochs)
            )

            if should_validate:
                val_metrics = server.evaluate_on_clients(val_loaders)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']

                self.history['epoch'].append(round_num)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                print(
                    f"Round {round_num:3d}: "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
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
        print(f"{'='*70}\n", flush=True)

        # Calculate model size for communication cost
        model_size_bytes = sum(
            p.numel() * 4  # float32 = 4 bytes
            for p in self.model.parameters()
        )
        
        return {
            'total_epochs': round_num,
            'best_epoch': best_epoch,
            'epochs_no_improve': self.epochs_no_improve,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': len(clients),
            'validation_frequency': validation_frequency,
            'local_epochs': local_epochs,
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
            'communication': {
                'total_rounds': round_num,
                'model_size_bytes': model_size_bytes,
                'total_communication_bytes': model_size_bytes * round_num * len(clients),
                'communication_per_round_bytes': model_size_bytes * len(clients)
            }
        }