#!/usr/bin/env python3
"""Federated Learning trainer."""

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
        cfg = self.config['training']
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.get('label_smoothing', 0.0))
        ).to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        raise NotImplementedError("Use fit() for federated training.")

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Full evaluation."""
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
        }

    def fit(
        self,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        client_ids: List[str],
        epochs: int = 100,
        patience: int = 10,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Federated training.

        Args:
            train_loaders: Training loaders (one per client)
            val_loaders: Validation loaders (one per client)
            client_ids: Client identifiers
            epochs: Global rounds
            patience: Early stopping patience
            output_dir: Checkpoint directory

        Returns:
            Training results
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

        self._print_header(
            len(clients), epochs, local_epochs,
            validation_frequency, patience
        )

        start_time = time.time()
        best_epoch = 0
        round_num = 0

        for round_num in range(1, epochs + 1):
            train_metrics = server.train_round()

            should_validate = (
                (round_num % validation_frequency == 0) or
                (round_num == epochs)
            )

            if should_validate:
                val_metrics = server.evaluate_on_clients(val_loaders)

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

        self.cleanup_memory()

        self._print_summary(round_num, best_epoch, elapsed)

        model_size_bytes = sum(
            p.numel() * 4 for p in self.model.parameters()
        )

        return {
            'total_epochs': round_num,
            'best_epoch': best_epoch,
            'epochs_no_improve': self.epochs_no_improve,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': len(clients),
            'local_epochs': local_epochs,
            'communication': {
                'total_rounds': round_num,
                'model_size_bytes': model_size_bytes,
                'total_communication_bytes': (
                    model_size_bytes * round_num * len(clients)
                ),
            },
        }

    def _print_header(
        self,
        n_clients: int,
        epochs: int,
        local_epochs: int,
        val_freq: int,
        patience: int,
    ) -> None:
        print(f"\n{'='*70}")
        print("FL TRAINING")
        print(f"{'='*70}")
        print(f"  Clients: {n_clients}")
        print(f"  Rounds: {epochs}")
        print(f"  Local epochs/round: {local_epochs}")
        print(f"  Validation frequency: {val_freq}")
        print(f"  Early stopping patience: {patience}")
        print(f"{'='*70}\n")

    def _print_round(
        self,
        round_num: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        print(
            f"Round {round_num:3d}: "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

    def _print_summary(
        self,
        total_rounds: int,
        best_round: int,
        elapsed: float,
    ) -> None:
        print(f"\n{'='*70}")
        print(f"Completed: {total_rounds} rounds in {elapsed:.1f}s")
        print(f"  Best: round {best_round} (acc={self.best_val_acc:.4f})")
        print(f"{'='*70}\n", flush=True)