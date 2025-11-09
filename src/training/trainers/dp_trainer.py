#!/usr/bin/env python3
"""
Differential Privacy Trainer using Opacus.

Implements privacy-preserving training with per-sample gradient clipping
and Gaussian noise addition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import json
import time
from pathlib import Path

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


class DPTrainer:
    """
    Trainer with Differential Privacy via Opacus.

    Features:
    - Per-sample gradient clipping
    - Gaussian noise addition
    - Privacy accounting (epsilon-delta)
    - Early stopping
    - Batch memory management
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cpu'):
        """
        Initialize DP trainer.

        Args:
            model: Neural network model
            config: Configuration dictionary
            device: Computing device (cuda/mps/cpu)
        """
        if not OPACUS_AVAILABLE:
            raise RuntimeError(
                "Opacus not installed. "
                "Install with: pip install opacus"
            )

        self.model = model
        self.config = config
        self.device = device
        self.logger = self._setup_logger()

        self.dp_config = config.get('differential_privacy', {})
        self.noise_multiplier = self.dp_config.get('noise_multiplier', 1.0)
        self.max_grad_norm = self.dp_config.get('max_grad_norm', 1.0)
        self.delta = self.dp_config.get('delta', 1e-5)
        self.target_epsilon = self.dp_config.get('target_epsilon', None)

        self.training_config = config.get('training', {})
        self.lr = self.training_config.get('learning_rate', 1e-3)
        self.weight_decay = self.training_config.get('weight_decay', 0.0)

        self.privacy_engine = None
        self.optimizer = None
        self.train_history = []
        self.val_history = []

    def _setup_logger(self):
        """Simple logger setup."""
        import logging
        logger = logging.getLogger('DPTrainer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_privacy_engine(self,
                             train_loader: DataLoader) -> PrivacyEngine:
        """
        Setup Opacus privacy engine.

        Args:
            train_loader: Training data loader

        Returns:
            Configured PrivacyEngine
        """
        privacy_engine = PrivacyEngine()

        self.model, self.optimizer, train_loader = (
            privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping_mode="per_sample"
            )
        )

        return privacy_engine, train_loader

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _train_epoch(self, train_loader: DataLoader,
                     criterion: nn.Module) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def _validate_epoch(self, val_loader: DataLoader,
                       criterion: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def _get_privacy_spent(self) -> Dict[str, float]:
        """
        Get privacy spent (epsilon).

        Returns:
            Dictionary with epsilon and delta
        """
        if (self.privacy_engine is None or
                not hasattr(self.privacy_engine, 'get_epsilon_spent')):
            return {'epsilon': float('inf'), 'delta': self.delta}

        try:
            epsilon = self.privacy_engine.get_epsilon_spent()
            return {'epsilon': float(epsilon), 'delta': self.delta}
        except Exception as e:
            self.logger.warning(
                f"Could not compute epsilon: {e}. "
                f"Using dummy value."
            )
            return {'epsilon': float('inf'), 'delta': self.delta}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 40, patience: int = 10,
            output_dir: str = './results') -> Dict[str, Any]:
        """
        Train model with differential privacy.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            output_dir: Directory to save results

        Returns:
            Dictionary with training results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()

        self.privacy_engine, train_loader = self._setup_privacy_engine(
            train_loader
        )

        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        start_time = time.time()

        self.logger.info(
            f"Starting DP training with "
            f"noise_multiplier={self.noise_multiplier}, "
            f"max_grad_norm={self.max_grad_norm}"
        )

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, criterion)
            val_metrics = self._validate_epoch(val_loader, criterion)

            privacy = self._get_privacy_spent()
            epsilon = privacy.get('epsilon', float('inf'))

            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Epsilon: {epsilon:.4f}"
            )

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                epochs_without_improvement = 0

                checkpoint_path = Path(output_dir) / 'best_model.pt'
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(
                    f"Model saved with val_acc={best_val_acc:.4f}"
                )
            else:
                epochs_without_improvement += 1

            if self.target_epsilon is not None:
                if epsilon >= self.target_epsilon:
                    self.logger.info(
                        f"Target epsilon {self.target_epsilon} reached "
                        f"(current: {epsilon:.4f}). Stopping."
                    )
                    break

            if epochs_without_improvement >= patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        elapsed = time.time() - start_time

        checkpoint_path = Path(output_dir) / 'best_model.pt'
        if checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        privacy_final = self._get_privacy_spent()

        results = {
            'total_epochs': epoch + 1,
            'best_epoch': best_epoch + 1,
            'best_val_acc': float(best_val_acc),
            'final_train_loss': float(train_metrics['loss']),
            'final_train_acc': float(train_metrics['accuracy']),
            'final_val_loss': float(val_metrics['loss']),
            'final_val_acc': float(val_metrics['accuracy']),
            'training_time_seconds': elapsed,
            'final_epsilon': float(privacy_final.get('epsilon', float('inf'))),
            'delta': float(privacy_final.get('delta', self.delta)),
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm
        }

        results_file = Path(output_dir) / 'training_history.json'
        with open(results_file, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history,
                'final_results': results
            }, f, indent=2)

        self.logger.info(
            f"Training completed in {elapsed:.1f}s\n"
            f"  Final epsilon: {results['final_epsilon']:.4f}\n"
            f"  Best val accuracy: {results['best_val_acc']:.4f}"
        )

        return results

    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_x.size(0)

        accuracy = total_correct / total_samples

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        from sklearn.metrics import (
            precision_score, recall_score, f1_score, confusion_matrix
        )

        n_classes = len(np.unique(all_targets))
        average_type = 'binary' if n_classes == 2 else 'weighted'

        precision = precision_score(
            all_targets, all_predictions, average=average_type, zero_division=0
        )
        recall = recall_score(
            all_targets, all_predictions, average=average_type, zero_division=0
        )
        f1 = f1_score(
            all_targets, all_predictions, average=average_type, zero_division=0
        )

        conf_matrix = confusion_matrix(all_targets, all_predictions)
        class_names = [f"class_{i}" for i in range(n_classes)]

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': class_names,
        }

        if self.privacy_engine is not None:
            privacy = self._get_privacy_spent()
            results['final_epsilon'] = float(
                privacy.get('epsilon', float('inf'))
            )
            results['delta'] = float(privacy.get('delta', self.delta))

        return results


if __name__ == "__main__":
    print("DP Trainer module (requires Opacus)")
    print("Install with: pip install opacus")