#!/usr/bin/env python3
"""
Federated Learning Trainer.

Implements federated learning with multiple clients and a central server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple
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

    def setup_optimizer_and_loss(self) -> None:
        """Setup loss (clients manage optimizers)."""
        cfg = self.config['training']
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
        logger.info("Loss function initialized (CE with label smoothing)")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch (not used in FL).

        FL uses federated rounds instead of epochs. This method exists
        only to satisfy the abstract base class interface.
        """
        raise NotImplementedError(
            "FLTrainer uses federated rounds, not epochs. "
            "Use fit() with train_loaders and client_ids instead."
        )

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with accuracy, precision, recall, f1, per-class
            metrics, confusion matrix and class names.
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

        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

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
            'class_names': self.config['dataset'].get('class_names', [])
        }

        return metrics

    def _update_history(self, epoch_num: int, train_loss: float,
                       train_acc: float, val_loss: float = None,
                       val_acc: float = None) -> None:
        """Update training history."""
        self.history['epoch'].append(epoch_num)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)

    def fit(self, train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 10,
            output_dir: str = None) -> Dict[str, Any]:
        """
        Train with Federated Learning.

        Args:
            train_loaders: List of training loaders (one per client)
            val_loaders: List of validation loaders (one per client)
            client_ids: List of client identifiers
            epochs: Number of global rounds
            patience: Early stopping patience (validation rounds)
            output_dir: Directory to save best model

        Returns:
            Dictionary with training results
        """
        self.setup_optimizer_and_loss()
        self._reset_training_state()

        fl_cfg = self.config.get('federated_learning', {})
        validation_frequency = fl_cfg.get('validation_frequency', 5)
        local_epochs = fl_cfg.get('local_epochs', 1)

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
        validation_count = 0

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

                self._update_history(
                    round_num, train_loss, train_acc,
                    val_loss, val_acc
                )

                validation_count += 1

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

                    if output_dir:
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        self.save_checkpoint(
                            f"{output_dir}/best_model.pth"
                        )
                        logger.info(
                            f"Saved best model at round {round_num} "
                            f"(val_acc={val_acc:.4f})"
                        )
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= patience:
                        print(
                            f"\nEarly stopping triggered at round {round_num} "
                            f"(no improvement for {patience} validation rounds)",
                            flush=True
                        )
                        break
            else:
                self._update_history(round_num, train_loss, train_acc)

                if round_num % max(1, validation_frequency // 2) == 0:
                    print(
                        f"Round {round_num:3d}: "
                        f"loss={train_loss:.4f} acc={train_acc:.4f}",
                        flush=True
                    )

                if round_num % 10 == 0 and self.device.type == 'cuda':
                    self.cleanup_memory()

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        if round_num == epochs:
            print(f"Training completed: {round_num} rounds "
                  f"in {elapsed:.1f}s")
        else:
            print(f"Training stopped early: {round_num} rounds "
                  f"in {elapsed:.1f}s")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f} "
              f"(round {best_epoch})")
        print(f"  Validation checks: {validation_count}")
        print(f"{'='*70}\n", flush=True)

        if output_dir:
            best_path = Path(output_dir) / 'best_model.pth'
            if best_path.exists():
                self.load_checkpoint((best_path))
                logger.info(f"Loaded best model from {best_path}")

        self.cleanup_memory()

        return {
            'total_epochs': round_num,
            'best_epoch': best_epoch,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': len(clients),
            'validation_frequency': validation_frequency,
            'local_epochs': local_epochs,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': (self.history['val_loss'][-1]
                              if self.history['val_loss'] else 0.0),
            'final_val_acc': (self.history['val_acc'][-1]
                             if self.history['val_acc'] else 0.0),
            'validation_checks': validation_count
        }