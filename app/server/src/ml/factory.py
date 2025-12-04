"""
Factory Pattern: Determines training mode and creates appropriate trainer.
Maps UI sliders (clients, sigma) to training algorithms.
Integrates YAML configurations with user inputs.
"""
import copy
import torch
import numpy as np
from typing import Dict, Any
from .trainers.centralized import BaselineTrainer, DPTrainer
from .trainers.federated import FederatedTrainer
from .models import get_model
from ..data.loader import data_loader
from ..core.dataset_configs import get_config


def get_trainer_mode(clients: int, sigma: float) -> str:
    """
    Determine training mode based on configuration.
    
    Args:
        clients: Number of federated clients (0 = centralized)
        sigma: Noise multiplier (0 = no DP)
    
    Returns:
        Mode string: BASELINE_CENTRALIZED, DP_CENTRALIZED, FEDERATED_LEARNING, FEDERATED_LEARNING_DP
    """
    if clients == 0 and sigma == 0:
        return "BASELINE_CENTRALIZED"
    elif clients == 0 and sigma > 0:
        return "DP_CENTRALIZED"
    elif clients > 0 and sigma == 0:
        return "FEDERATED_LEARNING"
    else:  # clients > 0 and sigma > 0
        return "FEDERATED_LEARNING_DP"


def build_config(dataset_name: str, clients: int, sigma: float, train_y: np.ndarray = None) -> Dict[str, Any]:
    """
    Build full configuration from dataset defaults and user selections.
    Merges YAML configs with runtime parameters.
    Computes class_weights dynamically from training data (matching paper).
    """
    # Get base config from dataset_configs (which loads from YAML and computes class_weights)
    base_cfg = get_config(dataset_name, clients, sigma, train_y=train_y)
    
    # Ensure DP config is properly set (use values from config, not hardcoded)
    if 'differential_privacy' not in base_cfg:
        base_cfg['differential_privacy'] = {}
    
    base_cfg['differential_privacy']['enabled'] = sigma > 0
    base_cfg['differential_privacy']['noise_multiplier'] = sigma
    base_cfg['differential_privacy']['delta'] = 1e-5
    # max_grad_norm and poisson_sampling come from config (5.0 and True per paper)
    
    # Ensure FL config
    if clients > 0:
        base_cfg['federated_learning']['n_clients'] = clients
        base_cfg['federated_learning']['enabled'] = True
    
    # Ensure epochs is always 40 (no early stopping in MVP)
    base_cfg['training']['epochs'] = 40
    
    return base_cfg


def create_trainer(job_id: str, config: Dict[str, Any], callback: Any) -> Any:
    """
    Factory function to create appropriate trainer based on configuration.
    
    Args:
        job_id: Unique job identifier
        config: Training configuration dict with keys: dataset, clients, sigma, batch_size, epochs
        callback: Function to call for progress updates (progress, metrics, log)
    
    Returns:
        Trainer instance (BaselineTrainer, DPTrainer, or FederatedTrainer)
    """
    # Set seed FIRST for reproducibility (matching paper: seed=42)
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 1. Prepare Data
    dataset_name = config["dataset"]
    
    # Load separate train/val/test from files (matching paper exactly)
    from pathlib import Path
    from ..core.config import settings
    
    dataset_dir = settings.DATA_DIR / dataset_name.lower()
    X_train = np.load(dataset_dir / 'X_train.npy', allow_pickle=False).astype(np.float32)
    y_train = np.load(dataset_dir / 'y_train.npy', allow_pickle=False).astype(np.int64)
    X_val = np.load(dataset_dir / 'X_val.npy', allow_pickle=False).astype(np.float32)
    y_val = np.load(dataset_dir / 'y_val.npy', allow_pickle=False).astype(np.int64)
    X_test = np.load(dataset_dir / 'X_test.npy', allow_pickle=False).astype(np.float32)
    y_test = np.load(dataset_dir / 'y_test.npy', allow_pickle=False).astype(np.int64)
    
    # For FL: need train+val combined with subjects
    data_meta = data_loader.load_dataset(dataset_name)
    train_data_full = data_meta['train']  # Tuple (X, y, subjects) - already combined
    
    # For validation: use VAL set (not test) - matching paper
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(), 
        torch.from_numpy(y_val).long()
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    
    # For final evaluation: use TEST set (matching paper)
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # 2. Build Full Config (with class_weights computed from TRAIN data only - matching paper)
    # For centralized: use X_train only
    # For FL: use train+val combined (already in train_data_full)
    if config["clients"] > 0:
        # FL: use combined train+val
        X_train_full, y_train_full, _ = train_data_full
    else:
        # Centralized: use train only (matching paper)
        X_train_full, y_train_full = X_train, y_train
    
    full_config = build_config(dataset_name, config["clients"], config["sigma"], train_y=y_train_full)
    
    # 3. Instantiate Model
    model = get_model(data_meta['input_dim'], data_meta['n_classes'])
    
    # 4. Select Strategy
    if config["clients"] > 0:
        # Federated (DP handled inside via use_dp flag)
        trainer = FederatedTrainer(
            model=model,
            config=full_config,
            n_clients=config["clients"],
            use_dp=(config["sigma"] > 0),
            device="cpu",  # CPU for stability
            callback=callback
        )
        # FL needs the full tuple (X, y, subjects) to split
        trainer.train_data_full = train_data_full
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader  # For final evaluation
    
    elif config["sigma"] > 0:
        # Centralized DP - use TRAIN only (not train+val)
        trainer = DPTrainer(
            model=model, 
            config=full_config, 
            device="cpu",
            callback=callback
        )
        # Create standard loader from TRAIN only (matching paper)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        trainer.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=config.get("batch_size", 128), shuffle=True
        )
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader  # For final evaluation
    
    else:
        # Baseline - use TRAIN only (not train+val)
        trainer = BaselineTrainer(
            model=model, 
            config=full_config, 
            device="cpu",
            callback=callback
        )
        # Create standard loader from TRAIN only (matching paper)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        trainer.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=config.get("batch_size", 128), shuffle=True
        )
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader  # For final evaluation
    
    return trainer
