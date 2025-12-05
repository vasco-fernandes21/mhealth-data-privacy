import numpy as np
from pathlib import Path
import yaml

def compute_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights (same formula as paper preprocessing)."""
    class_counts = np.bincount(y.astype(int))
    n_classes = len(class_counts)
    n_samples = len(y)
    
    weights = {}
    for c in range(n_classes):
        if class_counts[c] > 0:
            weights[c] = n_samples / (n_classes * class_counts[c])
        else:
            weights[c] = 1.0
    
    return weights

def load_yaml_config(dataset_name: str) -> dict:
    yaml_path = Path(__file__).parent.parent.parent.parent / "src" / "configs" / "datasets" / f"{dataset_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

WESAD_CONFIG = {
    "dataset": {
        "name": "wesad",
        "input_dim": 140,
        "n_classes": 2,
        "class_names": ["non-stress", "stress"],
        "use_class_weights": True
    },
    "model": {
        "hidden_dims": [256, 128],
        "dropout": 0.3
    },
    "training": {
        "batch_size": 128,
        "epochs": 40,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "momentum": 0.9,
        "scheduler": "cosine",
        "label_smoothing": 0.05,
        "gradient_clipping": True,
        "early_stopping_patience": 20,
        "num_workers": 0
    },
    "federated_learning": {
        "enabled": True,
        "n_clients": 10,
        "global_rounds": 40,
        "local_epochs": 1,
        "aggregation_method": "fedavg",
        "client_sampling_fraction": 1.0,
        "validation_frequency": 5
    },
    "differential_privacy": {
        "enabled": True,
        "target_delta": 1e-5,
        "noise_multiplier": 0.5,  # Default, can be overridden by sigma from UI
        "max_grad_norm": 5.0,
        "accounting_method": "rdp",
        "poisson_sampling": True,
        "grad_sample_mode": "hooks"
    }
}

SLEEP_EDF_CONFIG = {
    "dataset": {
        "name": "sleep-edf",
        "input_dim": 24,
        "n_classes": 5,
        "class_names": ["W", "N1", "N2", "N3", "R"],
        "use_class_weights": True
    },
    "model": {
        "hidden_dims": [256, 128],
        "dropout": 0.5
    },
    "training": {
        "batch_size": 128,
        "epochs": 40,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "label_smoothing": 0.05,
        "gradient_clipping": False,
        "early_stopping_patience": 20,
        "num_workers": 0
    },
    "federated_learning": {
        "enabled": True,
        "n_clients": 10,
        "global_rounds": 40,
        "local_epochs": 1,
        "aggregation_method": "fedavg",
        "client_sampling_fraction": 1.0,
        "validation_frequency": 5
    },
    "differential_privacy": {
        "enabled": True,
        "target_delta": 1e-5,
        "noise_multiplier": 0.5,
        "max_grad_norm": 5.0,
        "accounting_method": "rdp",
        "poisson_sampling": True,
        "grad_sample_mode": "hooks"
    }
}

def get_config(
    dataset_name: str,
    n_clients: int,
    sigma: float,
    train_y: np.ndarray = None,
) -> dict:
    """
    Get configuration for dataset, overriding with UI parameters.
    """
    try:
        yaml_cfg = load_yaml_config(dataset_name)
        if dataset_name.lower() == "wesad":
            config = WESAD_CONFIG.copy()
        elif dataset_name.lower() == "sleep-edf":
            config = SLEEP_EDF_CONFIG.copy()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if 'training' in yaml_cfg:
            config['training'].update(yaml_cfg['training'])
        if 'model' in yaml_cfg:
            config['model'].update(yaml_cfg['model'])
    except FileNotFoundError:
        if dataset_name.lower() == "wesad":
            config = WESAD_CONFIG.copy()
        elif dataset_name.lower() == "sleep-edf":
            config = SLEEP_EDF_CONFIG.copy()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if train_y is not None and config["dataset"].get("use_class_weights", False):
        weights_dict = compute_class_weights(train_y)
        config["dataset"]["class_weights"] = {
            str(k): float(v) for k, v in weights_dict.items()
        }
    
    config["federated_learning"]["n_clients"] = n_clients
    config["differential_privacy"]["noise_multiplier"] = sigma
    config["differential_privacy"]["enabled"] = sigma > 0
    
    config["training"]["epochs"] = 40
    
    return config

