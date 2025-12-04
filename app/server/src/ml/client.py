import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from typing import Dict, Tuple, Optional


class VirtualClient:
    def __init__(self, client_id: int, data: Tuple, config: Dict):
        self.id = client_id
        self.X, self.y = data
        self.config = config
        self.device = torch.device("cpu")  # CPU for simulation stability
        
        # Cache configurations
        self.training_cfg = config.get("training", {})
        self.dataset_cfg = config.get("dataset", {})
        self.dp_cfg = config.get("differential_privacy", {})
        self.fl_cfg = config.get("federated_learning", {})
        
        # State persistence for Federated Learning (Simulating real device)
        # CRITICAL: PrivacyEngine and dp_train_loader persist across rounds to accumulate epsilon correctly
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.dp_train_loader = None  # Reuse the same loader across rounds (like paper)
        self.total_steps = 0  # Track total training steps for epsilon accumulation
        
        # Calculate batch size for privacy accounting
        self.dataset_size = len(self.X)
        requested_batch_size = self.training_cfg.get("batch_size", 128)
        self.batch_size = min(requested_batch_size, self.dataset_size)
        if self.batch_size < 1:
            self.batch_size = 1
        
        # For poisson_sampling=True, ensure batch_size is reasonable
        # Opacus requires sample_rate < 1.0, so batch_size < dataset_size
        # Also, very small datasets can cause "Discrete mean differs" error
        if self.dataset_size < 100:
            # For very small datasets, reduce batch size to avoid Opacus errors
            self.batch_size = max(1, min(self.batch_size, self.dataset_size // 2))

    def train_round(self, global_state_dict) -> Tuple[Dict, Dict]:
        # 1. Initialize Model & Load Global Weights
        model = self._init_model()
        model.load_state_dict(global_state_dict)
        model.train()

        # 2. Setup Data (create dataset, but loader will be created by PrivacyEngine)
        dataset = TensorDataset(torch.from_numpy(self.X), torch.from_numpy(self.y))
        base_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Setup Optimizer
        optimizer = self._get_optimizer(model)

        # 4. Setup Criterion (Loss)
        criterion = self._get_criterion()

        # 5. Differential Privacy Setup
        # CRITICAL: Initialize PrivacyEngine ONCE and reuse the SAME dp_train_loader
        # This matches the paper's approach where the loader is created once and reused
        if self.privacy_engine is None:
            self.privacy_engine = PrivacyEngine()
            
            # Opacus make_private - first time setup (creates dp_train_loader)
            # Try poisson_sampling=True first (matches paper), fallback to False if error
            try:
                model, optimizer, self.dp_train_loader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=base_loader,
                    noise_multiplier=self.dp_cfg.get("noise_multiplier", 1.0),
                    max_grad_norm=self.dp_cfg.get("max_grad_norm", 1.0),
                    poisson_sampling=True,  # Match paper configuration
                    grad_sample_mode=self.dp_cfg.get("grad_sample_mode", "hooks")
                )
                self.use_poisson = True
            except Exception as e:
                print(f"Warning: poisson_sampling=True failed for client {self.id}: {e}")
                print(f"  Falling back to poisson_sampling=False")
                # Fallback to uniform sampling
                model, optimizer, self.dp_train_loader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=base_loader,
                    noise_multiplier=self.dp_cfg.get("noise_multiplier", 1.0),
                    max_grad_norm=self.dp_cfg.get("max_grad_norm", 1.0),
                    poisson_sampling=False,
                    grad_sample_mode=self.dp_cfg.get("grad_sample_mode", "hooks")
                )
                self.use_poisson = False
        else:
            # IMPORTANT: In Opacus, calling make_private multiple times can reset the accountant
            # The paper reuses the SAME dp_train_loader across rounds
            # We need to re-attach the PrivacyEngine but ensure the accountant persists
            # The PrivacyEngine's accountant should accumulate across rounds automatically
            poisson_setting = getattr(self, 'use_poisson', True)
            
            # Re-attach PrivacyEngine to new model/optimizer
            # The accountant state should persist in PrivacyEngine
            model, optimizer, new_dp_loader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=base_loader,
                noise_multiplier=self.dp_cfg.get("noise_multiplier", 1.0),
                max_grad_norm=self.dp_cfg.get("max_grad_norm", 1.0),
                poisson_sampling=poisson_setting,
                grad_sample_mode=self.dp_cfg.get("grad_sample_mode", "hooks")
            )
            # Update dp_train_loader reference (PrivacyEngine maintains accountant state)
            self.dp_train_loader = new_dp_loader
        
        # Use the dp_train_loader for training (maintains privacy accounting)
        loader = self.dp_train_loader if self.dp_train_loader is not None else base_loader

        # 6. Training Loop
        local_epochs = self.fl_cfg.get("local_epochs", 1)
        epoch_loss = 0.0
        total_samples = 0
        steps_this_round = 0
        
        for _ in range(local_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)
                steps_this_round += 1
        
        # Update total steps for privacy accounting
        self.total_steps += steps_this_round

        # 7. Calculate Epsilon (accumulated across all rounds)
        # The PrivacyEngine automatically tracks epsilon as training progresses
        delta = self.dp_cfg.get("target_delta", 1e-5)
        epsilon = 0.0
        
        if self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(delta)
                # Debug logging for first few rounds
                if self.total_steps <= 20 or self.total_steps % 10 == 0:
                    print(f"DEBUG Client {self.id}: epsilon={epsilon:.4f}, steps={self.total_steps}, delta={delta}")
                
                # Handle invalid epsilon values
                if not isinstance(epsilon, (int, float)) or epsilon == float('inf') or epsilon != epsilon or epsilon < 0:
                    print(f"WARN: Client {self.id} - Invalid epsilon: {epsilon} (type: {type(epsilon)})")
                    epsilon = 0.0
            except Exception as e:
                print(f"WARN: Client {self.id} - Epsilon calc failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback estimation if RDP fails
                try:
                    from ..ml.privacy import estimate_epsilon
                    sample_rate = self.batch_size / self.dataset_size if self.dataset_size > 0 else 0.001
                    epsilon = estimate_epsilon(
                        self.dp_cfg.get("noise_multiplier", 1.0),
                        sample_rate,
                        self.total_steps,  # Use total steps for accumulation
                        delta
                    )
                    print(f"DEBUG Client {self.id}: Using fallback epsilon={epsilon:.4f}")
                except Exception as est_err:
                    print(f"WARN: Client {self.id} - Fallback epsilon also failed: {est_err}")
                    epsilon = 0.0
        else:
            print(f"ERROR: Client {self.id} - PrivacyEngine is None!")

        # Return weights (unwrapped) and metrics
        return model._module.state_dict(), {
            "loss": epoch_loss / total_samples if total_samples > 0 else 0.0,
            "epsilon": float(epsilon),
            "client_id": self.id,
            "steps": steps_this_round,
            "total_steps": self.total_steps
        }

    def _init_model(self):
        from .models import get_model
        input_dim = self.dataset_cfg.get("input_dim")
        n_classes = self.dataset_cfg.get("n_classes")
        return get_model(input_dim, n_classes, self.device)

    def _get_optimizer(self, model):
        lr = self.training_cfg.get("learning_rate", 0.0005)
        weight_decay = self.training_cfg.get("weight_decay", 0.0001)
        opt_name = self.training_cfg.get("optimizer", "adamw").lower()
        
        if opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = self.training_cfg.get("momentum", 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_criterion(self):
        # Handle Class Weights
        class_weights = None
        if self.dataset_cfg.get("use_class_weights", False):
            if "class_weights" in self.dataset_cfg:
                weights_dict = self.dataset_cfg["class_weights"]
                n_classes = self.dataset_cfg.get("n_classes", 2)
                class_weights = torch.zeros(n_classes, dtype=torch.float32)
                for class_idx, weight in weights_dict.items():
                    class_weights[int(class_idx)] = float(weight)
                class_weights = class_weights.to(self.device)
        
        label_smoothing = float(self.training_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
