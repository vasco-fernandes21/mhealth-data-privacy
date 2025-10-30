# src/training/dp_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

from src.training.base_trainer import BaseTrainer
from src.privacy.dp_utils import DPConfig, setup_privacy_engine, get_epsilon
from src.utils.logging_utils import get_logger


class DPTrainer(BaseTrainer):
    """Training with Differential Privacy (Opacus)."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cuda'):
        super().__init__(model, config, device)
        self.privacy_engine = None
        self.dp_config = DPConfig(config)
        self.privacy_budget_history = []
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer (Adam) and loss."""
        cfg = self.config['training']
        
        # ✅ Adam for DP (better than SGD with Opacus)
        lr = float(cfg['learning_rate'])
        weight_decay = float(cfg.get('weight_decay', 1e-4))
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, 
            weight_decay=weight_decay
        )
        
        # Loss
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch with DP."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Forward + Backward
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # PrivacyEngine handles clipping + noise
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        return total_loss / total, correct / total
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, patience: int = 8,
            output_dir: str = None) -> Dict[str, Any]:
        """Train with DP."""
        
        self.setup_optimizer_and_loss()
        self._reset_training_state()
        
        # Setup PrivacyEngine
        self.model, self.optimizer, self.privacy_engine = \
            setup_privacy_engine(
                self.model, self.optimizer, train_loader,
                self.dp_config, device=str(self.device)
            )
        
        import time
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Get epsilon
            epsilon = get_epsilon(
                self.privacy_engine,
                float(self.dp_config.target_delta)
            )
            self.privacy_budget_history.append(epsilon)
            
            # Store + log
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self._log_epoch_dp(epoch, train_loss, train_acc,
                             val_loss, val_acc, epsilon)
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                if output_dir:
                    from pathlib import Path
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(
                        f"{output_dir}/best_model.pth"
                    )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    break
        
        elapsed = time.time() - start_time
        
        if output_dir:
            from pathlib import Path
            best_path = Path(output_dir) / 'best_model.pth'
            if best_path.exists():
                self.load_checkpoint(str(best_path))
        
        final_epsilon = get_epsilon(
            self.privacy_engine,
            float(self.dp_config.target_delta)
        )
        
        return {
            'total_epochs': epoch,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'final_epsilon': float(final_epsilon),
            'history': self.history,
            'privacy_budget_history': self.privacy_budget_history
        }
    
    def evaluate_full(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, per-class metrics,
            confusion matrix and class names.
        """
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(y_true, y_pred, labels=unique_labels, zero_division=0)
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
            'class_names': self.config['dataset'].get('class_names', []),
            'final_epsilon': float(
                get_epsilon(self.privacy_engine, float(self.dp_config.target_delta))
            )
        }
        
        return metrics
    
    def _log_epoch_dp(self, epoch: int, train_loss: float,
                      train_acc: float, val_loss: float,
                      val_acc: float, epsilon: float) -> None:
        """Log with DP metrics."""
        print(
            f"Epoch {epoch:03d}: "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"ε={epsilon:.4f}"
        )