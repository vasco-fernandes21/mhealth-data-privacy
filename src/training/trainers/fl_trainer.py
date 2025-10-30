# src/training/fl_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

from src.training.base_trainer import BaseTrainer
from src.privacy.fl_client import FLClient
from src.privacy.fl_server import FLServer


class FLTrainer(BaseTrainer):
    """Federated Learning trainer."""
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup loss (clients manage optimizers)."""
        cfg = self.config['training']
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
    
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
            'class_names': self.config['dataset'].get('class_names', [])
        }
        
        return metrics
    
    def _update_history(self, epoch: int, train_loss: float, train_acc: float,
                       val_loss: float, val_acc: float) -> None:
        """Update training history."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
    
    def fit(self, train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 8,
            output_dir: str = None) -> Dict[str, Any]:
        """Train with FL."""
        
        self.setup_optimizer_and_loss()
        self._reset_training_state()
        
        # Setup clients + server
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
        
        print(f"\n🚀 Starting FL training with {len(clients)} clients, {epochs} global rounds", flush=True)
        print(f"   Local epochs per round: {self.config['federated_learning'].get('local_epochs', 1)}", flush=True)
        print(f"   Patience: {patience}\n", flush=True)
        
        for round_num in range(1, epochs + 1):
            # Train round
            train_metrics = server.train_round()
            train_loss = train_metrics['avg_loss']
            train_acc = train_metrics['avg_accuracy']
            
            # Validate
            val_metrics = server.evaluate_on_clients(val_loaders)
            val_loss = val_metrics['avg_loss']
            val_acc = val_metrics['avg_accuracy']
            
            # Store + log
            self._update_history(round_num, train_loss, train_acc,
                               val_loss, val_acc)
            print(
                f"Round {round_num:03d}: "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True
            )
            
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
                    print(f"\n⏹️  Early stopping triggered after {round_num} rounds (patience={patience})", flush=True)
                    break
        
        elapsed = time.time() - start_time
        
        if round_num == epochs:
            print(f"\n✅ Training completed: {round_num} rounds in {elapsed:.1f}s", flush=True)
        else:
            print(f"\n✅ Training stopped early: {round_num} rounds in {elapsed:.1f}s", flush=True)
        
        print(f"   Best validation accuracy: {self.best_val_acc:.4f}\n", flush=True)
        
        if output_dir:
            from pathlib import Path
            best_path = Path(output_dir) / 'best_model.pth'
            if best_path.exists():
                self.load_checkpoint(str(best_path))
        
        return {
            'total_rounds': round_num,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': len(clients)
        }