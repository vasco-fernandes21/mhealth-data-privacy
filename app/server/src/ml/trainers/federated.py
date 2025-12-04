"""
Federated Learning Trainer: FL and FL+DP modes.
Integrates FLClient and FLServer logic.
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict, List, Any, Optional
from opacus import PrivacyEngine

from .base import BaseTrainer
from ...ml.models import get_model


class FLClient:
    """Federated Learning Client (local training)."""
    
    def __init__(self, client_id, model, data, config, device, use_dp=False):
        self.id = client_id
        self.model = model.to(device)
        self.X, self.y = data
        self.config = config
        self.device = torch.device(device)
        self.use_dp = use_dp
        self.privacy_engine = None
        
        # Setup Data Loader
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X).float(), 
            torch.from_numpy(self.y).long()
        )
        bs = config['training']['batch_size']
        self.loader = torch.utils.data.DataLoader(
            ds, batch_size=min(bs, len(ds)), shuffle=True
        )
        
        # Init Optimizer/Loss
        self._init_optimizer()
        self._init_criterion()
        
        if self.use_dp:
            self._setup_dp()
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        cfg = self.config['training']
        lr = float(cfg.get('learning_rate', 0.0005))
        weight_decay = float(cfg.get('weight_decay', 0.0001))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def _init_criterion(self):
        """Initialize loss function with class weights if needed."""
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
        """Setup PrivacyEngine for DP."""
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
            # Fallback to uniform sampling (only if poisson fails)
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
        """Train locally for one round and return updated weights."""
        # --- FIX: Handle Opacus '_module.' prefix mismatch ---
        # Se o modelo local tem DP (está envolvido), mas os pesos globais não têm prefixo
        is_dp_model = hasattr(self.model, '_module')
        
        adjusted_weights = {}
        for k, v in global_weights.items():
            if is_dp_model and not k.startswith('_module.'):
                adjusted_weights[f'_module.{k}'] = v
            elif not is_dp_model and k.startswith('_module.'):
                adjusted_weights[k.replace('_module.', '')] = v
            else:
                adjusted_weights[k] = v
        # -----------------------------------------------------

        # Load global state safely
        try:
            self.model.load_state_dict(adjusted_weights, strict=True)
        except RuntimeError as e:
            # Fallback: Tenta carregar ignorando erros menores se a estrutura base for compatível
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
        
        # Return weights (clean) and metrics
        # Always return CLEAN weights (without _module prefix) to server
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
    """Federated Learning Trainer for FL and FL+DP modes."""
    
    def __init__(self, n_clients, use_dp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = n_clients
        self.use_dp = use_dp
        self.clients = []
    
    def setup_clients(self, train_data_full):
        """Setup clients with subject-based data splitting."""
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
        """Simple FedAvg aggregation."""
        if not client_weights:
            return None
        
        avg_weights = copy.deepcopy(client_weights[0])
        for k in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[k] += client_weights[i][k]
            avg_weights[k] = torch.div(avg_weights[k], len(client_weights))
        return avg_weights
    
    def fit(self, train_data, val_loader) -> Dict[str, Any]:
        """Execute federated training."""
        self.setup_clients(train_data)
        rounds = self.config['training']['epochs']  # In FL context, epochs = rounds
        
        mode = "FL+DP" if self.use_dp else "FL"
        self.log(f"Protocol: {mode} | Clients: {len(self.clients)}")
        
        for r in range(1, rounds + 1):
            local_weights = []
            max_epsilon = 0.0
            total_loss = 0.0
            
            # Add a visual separator in logs for major rounds
            if r == 1 or r % 10 == 0:
                self.log(f"--- Global Round {r} Started ---")
            
            # Client Training Round
            for i, client in enumerate(self.clients):
                w, metrics = client.train_round(self.model.state_dict())
                local_weights.append(w)
                if metrics['epsilon'] > max_epsilon:
                    max_epsilon = metrics['epsilon']
                total_loss += metrics['loss']
                
                # LOG ACTIVITY FOR UI (Verbose Mode for Simulation feel)
                # Only log first few clients to avoid UI lag if >50 clients
                if len(self.clients) <= 10 or i < 3: 
                    self.log(f"   > Node {i+1}: Local Update Compute Complete (Loss: {metrics['loss']:.3f})")
            
            if len(self.clients) > 10:
                self.log(f"   > ... and {len(self.clients) - 3} other nodes.")
            
            if not local_weights:
                self.log("Warning: No client weights collected")
                break
            
            # Server Aggregation
            self.log(f"   > Server: Aggregating {len(local_weights)} models (FedAvg)...")
            global_weights = self.aggregate(local_weights)
            if global_weights:
                self.model.load_state_dict(global_weights)
            
            # Global Validation
            v_loss, v_acc = self.validate(val_loader)
            
            # Update history
            self.history['epoch'].append(r)
            self.history['val_loss'].append(v_loss)
            self.history['val_acc'].append(v_acc)
            
            avg_loss = total_loss / len(self.clients) if self.clients else 0.0
            
            # Update Progress Bar & Metrics for UI
            progress = int((r / rounds) * 100)
            self.callback(progress=progress, metrics={
                "accuracy": v_acc, 
                "loss": avg_loss, 
                "epsilon": max_epsilon,
                "round": r
            })
            
            self.log(f"✅ Round {r}: Global Acc {v_acc:.4f} | ε {max_epsilon:.2f}")
        
        return {
            "final_acc": v_acc, 
            "final_epsilon": max_epsilon,
            "rounds": rounds
        }
    
    def evaluate_full(self, test_loader) -> Dict[str, Any]:
        """Full evaluation on global model."""
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = self.model(bx)
                y_true.extend(by.cpu().numpy())
                y_pred.extend(torch.max(out, 1)[1].cpu().numpy())
        
        # Calculate Fairness (Recall of minority class)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get minority class
        unique, counts = np.unique(y_true, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[minority_class], zero_division=0
        )
        
        from sklearn.metrics import accuracy_score
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "minority_recall": float(rec[0]) if len(rec) > 0 else 0.0
        }
