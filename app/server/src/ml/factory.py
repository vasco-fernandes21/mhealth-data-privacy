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


def build_config(
    dataset_name: str,
    clients: int,
    sigma: float,
    train_y: np.ndarray = None,
    max_grad_norm: float | None = None,
    use_class_weights: bool | None = None,
) -> Dict[str, Any]:
    """
    Build full configuration from dataset defaults and user selections.
    """
    base_cfg = get_config(dataset_name, clients, sigma, train_y=train_y)
    
    if 'differential_privacy' not in base_cfg:
        base_cfg['differential_privacy'] = {}
    
    base_cfg['differential_privacy']['enabled'] = sigma > 0
    base_cfg['differential_privacy']['noise_multiplier'] = sigma
    base_cfg['differential_privacy']['delta'] = 1e-5
    if max_grad_norm is not None:
        base_cfg['differential_privacy']['max_grad_norm'] = float(max_grad_norm)
    
    if use_class_weights is not None:
        base_cfg.setdefault("dataset", {})
        base_cfg["dataset"]["use_class_weights"] = bool(use_class_weights)
    
    if clients > 0:
        base_cfg['federated_learning']['n_clients'] = clients
        base_cfg['federated_learning']['enabled'] = True
    
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
    import random
    seed = config.get("seed", 42)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    dataset_name = config["dataset"]
    
    import sys
    from pathlib import Path
    from ..core.config import settings
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from preprocessing.sleep_edf import load_windowed_sleep_edf
    from preprocessing.wesad import load_processed_wesad
    
    dataset_dir = settings.DATA_DIR / dataset_name.lower()
    
    if dataset_name.lower() == 'sleep-edf':
        data_tuple = load_windowed_sleep_edf(str(dataset_dir))
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects_all = data_tuple
        subjects_train = np.load(dataset_dir / 'subjects_train.npy', allow_pickle=True)
        subjects_val = np.load(dataset_dir / 'subjects_val.npy', allow_pickle=True)
        subjects_test = np.load(dataset_dir / 'subjects_test.npy', allow_pickle=True)
    elif dataset_name.lower() == 'wesad':
        data_tuple = load_processed_wesad(str(dataset_dir))
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = data_tuple
        subjects_train = np.load(dataset_dir / 'subjects_train.npy', allow_pickle=True)
        subjects_val = np.load(dataset_dir / 'subjects_val.npy', allow_pickle=True)
        subjects_test = np.load(dataset_dir / 'subjects_test.npy', allow_pickle=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    data_meta = data_loader.load_dataset(dataset_name)
    train_data_full = data_meta['train']
    
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(), 
        torch.from_numpy(y_val).long()
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    
    if config["clients"] > 0:
        X_train_full, y_train_full = X_train, y_train
    else:
        X_train_full, y_train_full = X_train, y_train
    
    full_config = build_config(
        dataset_name,
        config["clients"],
        config["sigma"],
        train_y=y_train_full,
        max_grad_norm=config.get("max_grad_norm"),
        use_class_weights=config.get("use_class_weights"),
    )
    
    # 3. Instantiate Model
    model = get_model(data_meta['input_dim'], data_meta['n_classes'])
    
    if config["clients"] > 0:
        trainer = FederatedTrainer(
            model=model,
            config=full_config,
            n_clients=config["clients"],
            use_dp=(config["sigma"] > 0),
            device="cpu",
            callback=callback
        )
        trainer.train_data_full = (X_train, y_train, np.zeros(len(X_train)))
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
    
    elif config["sigma"] > 0:
        trainer = DPTrainer(
            model=model, 
            config=full_config, 
            device="cpu",
            callback=callback
        )
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        trainer.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=config.get("batch_size", 128), shuffle=True
        )
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
    
    else:
        # Baseline - use TRAIN only (not train+val)
        trainer = BaselineTrainer(
            model=model, 
            config=full_config, 
            device="cpu",
            callback=callback
        )
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        trainer.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=config.get("batch_size", 128), shuffle=True
        )
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
    
    return trainer
