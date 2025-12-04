"""
Centralized Training: Baseline and DP (non-federated).
Integrates BaselineTrainer and DPTrainer logic.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .base import BaseTrainer


class BaselineTrainer(BaseTrainer):
    """Baseline centralized trainer without differential privacy."""
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer, loss function, and scheduler."""
        cfg = self.config['training']
        dataset_cfg = self.config.get('dataset', {})
        
        # Optimizer Setup
        lr = float(cfg.get('learning_rate', 0.0005))
        weight_decay = float(cfg.get('weight_decay', 0.0001))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Loss with Class Weights
        class_weights = None
        if dataset_cfg.get('use_class_weights', False):
            weights_dict = dataset_cfg.get('class_weights', {})
            if weights_dict:
                n_classes = dataset_cfg.get('n_classes', 2)
                cw = torch.zeros(n_classes, dtype=torch.float32)
                for k, v in weights_dict.items():
                    cw[int(k)] = float(v)
                class_weights = cw.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=float(cfg.get('label_smoothing', 0.05))
        ).to(self.device)
        
        # Scheduler
        epochs = int(cfg.get('epochs', 40))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        self.scheduler.step()
        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0
    
    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        """Execute baseline training."""
        self.setup_optimizer_and_loss()
        epochs = self.config['training']['epochs']
        
        self.log(f"Starting Baseline Training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc = self.validate(val_loader)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(t_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_loss'].append(v_loss)
            self.history['val_acc'].append(v_acc)
            
            # Update UI
            progress = int((epoch / epochs) * 100)
            self.callback(progress=progress, metrics={
                "accuracy": v_acc, "loss": v_loss, "epsilon": 0.0, "epoch": epoch
            })
            
            if epoch % 5 == 0:
                self.log(f"Epoch {epoch}/{epochs}: Train Acc {t_acc:.4f}, Val Acc {v_acc:.4f}")
        
        return {"final_acc": v_acc, "epochs": epochs}
    
    def evaluate_full(self, test_loader) -> Dict[str, Any]:
        """Full evaluation with fairness metrics."""
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = self.model(bx)
                y_true.extend(by.cpu().numpy())
                y_pred.extend(torch.max(out, 1)[1].cpu().numpy())
        
        # Calculate Fairness (Recall of minority class - typically class 1)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get minority class (class with fewer samples)
        unique, counts = np.unique(y_true, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[minority_class], zero_division=0
        )
        
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "minority_recall": float(rec[0]) if len(rec) > 0 else 0.0
        }


class DPTrainer(BaselineTrainer):
    """Differential Privacy Trainer using Opacus."""
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup optimizer, loss, and PrivacyEngine."""
        super().setup_optimizer_and_loss()
        self.privacy_engine = PrivacyEngine()
    
    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        """Execute DP training with PrivacyEngine."""
        self.setup_optimizer_and_loss()
        epochs = self.config['training']['epochs']
        dp_cfg = self.config['differential_privacy']
        
        # Attach Privacy Engine (matching paper: max_grad_norm=5.0, poisson_sampling=True)
        try:
            self.model, self.optimizer, train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=dp_cfg['noise_multiplier'],
                max_grad_norm=dp_cfg.get('max_grad_norm', 5.0),  # Paper uses 5.0
                poisson_sampling=dp_cfg.get('poisson_sampling', True),  # Paper uses True
                grad_sample_mode=dp_cfg.get('grad_sample_mode', 'hooks')
            )
        except Exception as e:
            self.log(f"Warning: poisson_sampling failed, using uniform: {e}")
            # Fallback to uniform sampling (only if poisson fails)
            self.model, self.optimizer, train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=dp_cfg['noise_multiplier'],
                max_grad_norm=dp_cfg.get('max_grad_norm', 5.0),
                poisson_sampling=False,
                grad_sample_mode=dp_cfg.get('grad_sample_mode', 'hooks')
            )
        
        self.log(f"DP Training Started (Sigma={dp_cfg['noise_multiplier']})")
        
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc = self.validate(val_loader)
            
            # Get epsilon
            epsilon = 0.0
            try:
                epsilon = float(self.privacy_engine.get_epsilon(delta=dp_cfg.get('delta', 1e-5)))
            except:
                pass
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(t_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_loss'].append(v_loss)
            self.history['val_acc'].append(v_acc)
            
            # Update UI
            progress = int((epoch / epochs) * 100)
            self.callback(progress=progress, metrics={
                "accuracy": v_acc, "loss": v_loss, "epsilon": epsilon, "epoch": epoch
            })
            
            if epoch % 5 == 0:
                self.log(f"Epoch {epoch}/{epochs}: Val Acc {v_acc:.4f}, ε={epsilon:.2f}")
        
        # Final epsilon
        final_epsilon = 0.0
        try:
            final_epsilon = float(self.privacy_engine.get_epsilon(delta=dp_cfg.get('delta', 1e-5)))
        except:
            pass
        
        return {"final_acc": v_acc, "final_epsilon": final_epsilon, "epochs": epochs}
    
    def evaluate_full(self, test_loader) -> Dict[str, Any]:
        """Full evaluation (same as baseline)."""
        return super().evaluate_full(test_loader)
