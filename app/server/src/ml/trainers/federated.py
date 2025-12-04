"""Federated Learning Trainer: FL and FL+DP modes."""

import time
import copy
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .base import BaseTrainer
from ...ml.models import get_model


class FLClient:
    """Federated Learning client."""

    def __init__(self, client_id, model, data, config, device, use_dp=False):
        self.id = client_id
        self.model = model.to(device)
        self.X, self.y = data
        self.config = config
        self.device = torch.device(device)
        self.use_dp = use_dp
        self.privacy_engine = None
        
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X).float(), 
            torch.from_numpy(self.y).long()
        )
        bs = config['training']['batch_size']
        self.loader = torch.utils.data.DataLoader(
            ds, batch_size=min(bs, len(ds)), shuffle=True
        )
        
        self._init_optimizer()
        self._init_criterion()
        
        if self.use_dp:
            self._setup_dp()
    
    def _init_optimizer(self):
        cfg = self.config['training']
        lr = float(cfg.get('learning_rate', 0.0005))
        weight_decay = float(cfg.get('weight_decay', 0.0001))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def _init_criterion(self):
        dataset_cfg = self.config.get('dataset', {})
        training_cfg = self.config.get('training', {})
        
        class_weights = None
        if dataset_cfg.get('use_class_weights', False):
            weights_dict = dataset_cfg.get('class_weights', {})
            if weights_dict:
                n_classes = dataset_cfg.get('n_classes', 2)
                cw = torch.zeros(n_classes, dtype=torch.float32)
                for k, v in weights_dict.items():
                    cw[int(k)] = float(v)
                class_weights = cw.to(self.device)
        
        label_smoothing = float(training_cfg.get('label_smoothing', 0.05))
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        ).to(self.device)
    
    def _setup_dp(self):
        dp_cfg = self.config['differential_privacy']
        self.privacy_engine = PrivacyEngine()
        
        try:
            self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=dp_cfg['noise_multiplier'],
                max_grad_norm=dp_cfg.get('max_grad_norm', 5.0),  # Paper uses 5.0
                poisson_sampling=dp_cfg.get('poisson_sampling', True),  # Paper uses True
                grad_sample_mode=dp_cfg.get('grad_sample_mode', 'hooks')
            )
        except Exception as e:
            self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=dp_cfg['noise_multiplier'],
                max_grad_norm=dp_cfg.get('max_grad_norm', 5.0),
                poisson_sampling=False,
                grad_sample_mode=dp_cfg.get('grad_sample_mode', 'hooks')
            )
    
    def train_round(self, global_weights):
        is_dp_model = hasattr(self.model, '_module')
        
        adjusted_weights = {}
        for k, v in global_weights.items():
            if is_dp_model and not k.startswith('_module.'):
                adjusted_weights[f'_module.{k}'] = v
            elif not is_dp_model and k.startswith('_module.'):
                adjusted_weights[k.replace('_module.', '')] = v
            else:
                adjusted_weights[k] = v
        try:
            self.model.load_state_dict(adjusted_weights, strict=True)
        except RuntimeError as e:
            print(f"[WARN] Strict loading failed for client {self.id}, trying non-strict. Error: {e}")
            self.model.load_state_dict(adjusted_weights, strict=False)

        self.model.train()
        
        loss_accum = 0.0
        total_samples = 0
        local_epochs = self.config['federated_learning']['local_epochs']
        
        for _ in range(local_epochs):
            for bx, by in self.loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
                
                loss_accum += loss.item() * bx.size(0)
                total_samples += bx.size(0)
        
        if hasattr(self.model, '_module'):
            state = self.model._module.state_dict()
        else:
            state = self.model.state_dict()
            
        epsilon = 0.0
        if self.use_dp and self.privacy_engine is not None:
            try:
                epsilon = float(self.privacy_engine.get_epsilon(delta=self.config['differential_privacy'].get('delta', 1e-5)))
            except:
                pass
        
        avg_loss = loss_accum / total_samples if total_samples > 0 else 0.0
        
        return state, {"loss": avg_loss, "epsilon": epsilon}


class FederatedTrainer(BaseTrainer):
    """Federated Learning trainer for FL and FL+DP modes."""

    def __init__(self, n_clients, use_dp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = n_clients
        self.use_dp = use_dp
        self.clients = []
    
    def setup_clients(self, train_data_full):
        X_full, y_full, subjects = train_data_full
        unique_subs = np.unique(subjects)
        sub_splits = np.array_split(unique_subs, self.n_clients)
        
        self.clients = []
        for i, subs in enumerate(sub_splits):
            mask = np.isin(subjects, subs)
            if mask.sum() == 0:
                continue
            
            # Instantiate fresh model per client
            c_model = get_model(
                self.config['dataset']['input_dim'], 
                self.config['dataset']['n_classes']
            )
            
            client = FLClient(
                client_id=i,
                model=c_model,
                data=(X_full[mask], y_full[mask]),
                config=self.config,
                device=str(self.device),
                use_dp=self.use_dp
            )
            self.clients.append(client)
    
    def aggregate(self, client_weights):
        if not client_weights:
            return None
        
        avg_weights = copy.deepcopy(client_weights[0])
        for k in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[k] += client_weights[i][k]
            avg_weights[k] = torch.div(avg_weights[k], len(client_weights))
        return avg_weights
    
    def fit(self, train_data, val_loader) -> Dict[str, Any]:
        """
        Federated training with round-based early stopping.
        Mirrors the FLTrainer structure from the offline experiments.
        """
        self._reset_training_state()
        self.setup_clients(train_data)

        fl_cfg = self.config.get("federated_learning", {})
        rounds = int(self.config["training"].get("epochs", fl_cfg.get("global_rounds", 40)))
        local_epochs = int(fl_cfg.get("local_epochs", 1))
        patience = int(self.config["training"].get("early_stopping_patience", 10))

        mode = "FL+DP" if self.use_dp else "FL"
        self.log(f"Protocol: {mode} | Clients: {len(self.clients)} | Rounds: {rounds}")

        start_time = time.time()
        best_epoch = 0
        last_round = 0
        last_train_loss = 0.0
        last_train_acc = 0.0

        max_epsilon = 0.0

        for r in range(1, rounds + 1):
            local_weights: List[Dict[str, torch.Tensor]] = []
            total_loss = 0.0

            for i, client in enumerate(self.clients):
                weights, metrics = client.train_round(self.model.state_dict())
                local_weights.append(weights)
                total_loss += metrics["loss"]
                if metrics["epsilon"] > max_epsilon:
                    max_epsilon = metrics["epsilon"]

                if len(self.clients) <= 10 or i < 3:
                    self.log(f"Client {i + 1}: loss={metrics['loss']:.4f}")

            if not local_weights:
                self.log("No client updates collected; stopping.")
                break

            global_weights = self.aggregate(local_weights)
            if global_weights:
                self.model.load_state_dict(global_weights)

            v_loss, v_acc = self.validate(val_loader)
            avg_loss = total_loss / len(self.clients) if self.clients else 0.0

            self.history["epoch"].append(r)
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(v_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            last_round = r
            last_train_loss = avg_loss
            last_train_acc = v_acc

            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                self.epochs_no_improve = 0
                best_epoch = r
                self.best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    self.log(
                        f"Early stopping at round {r} "
                        f"(patience={patience}, no improvement for {self.epochs_no_improve} rounds)"
                    )
                    break

            progress = int((r / rounds) * 100)
            self.callback(
                progress=progress,
                metrics={
                    "accuracy": v_acc,
                    "loss": avg_loss,
                    "epsilon": max_epsilon,
                    "round": r,
                },
            )

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        elapsed = time.time() - start_time

        model_size_bytes = sum(p.numel() * 4 for p in self.model.parameters())

        return {
            "total_epochs": last_round,
            "best_epoch": best_epoch or last_round,
            "epochs_no_improve": self.epochs_no_improve,
            "training_time_seconds": elapsed,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "n_clients": len(self.clients),
            "local_epochs": local_epochs,
            "final_train_loss": last_train_loss,
            "final_train_acc": last_train_acc,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            "final_val_acc": self.history["val_acc"][-1] if self.history["val_acc"] else 0.0,
            "communication": {
                "total_rounds": last_round,
                "model_size_bytes": model_size_bytes,
                "total_communication_bytes": model_size_bytes * last_round * len(self.clients),
                "communication_per_round_bytes": model_size_bytes * len(self.clients),
            },
            "final_acc": self.best_val_acc,
            "final_epsilon": max_epsilon,
            "rounds": last_round,
        }

    def evaluate_full(self, test_loader) -> Dict[str, Any]:
        """Full evaluation with metrics, aligned with offline FL experiments."""
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = self.model(bx)
                y_true.extend(by.cpu().numpy())
                y_pred.extend(torch.max(out, 1)[1].cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_labels = np.unique(y_true)

        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        unique, counts = np.unique(y_true, return_counts=True)
        minority_idx = int(np.argmin(counts))
        minority_class = unique[minority_idx]
        _, minority_rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[minority_class], zero_division=0
        )

        class_names = self.config["dataset"].get("class_names", [])

        class_distribution = {
            str(class_names[i]) if i < len(class_names) else str(int(lbl)): int(cnt)
            for i, (lbl, cnt) in enumerate(zip(unique, counts))
        }
        total_samples = int(counts.sum())
        imbalance_ratio = (
            float(counts.max() / counts.min()) if counts.min() > 0 else 1.0
        )
        minority_name = (
            class_names[minority_idx]
            if minority_idx < len(class_names)
            else str(int(minority_class))
        )
        majority_idx = int(np.argmax(counts))
        majority_name = (
            class_names[majority_idx]
            if majority_idx < len(class_names)
            else str(int(unique[majority_idx]))
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=unique_labels,
                )
            ),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "minority_recall": float(minority_rec[0]) if len(minority_rec) > 0 else 0.0,
            "class_imbalance": {
                "class_distribution": class_distribution,
                "imbalance_ratio": imbalance_ratio,
                "total_samples": total_samples,
                "minority_class": minority_name,
                "majority_class": majority_name,
            },
        }
