"""
Dataset configurations matching the paper's experimental setup.
"""

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
        "max_grad_norm": 1.0,  # Standard for DP-SGD
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
        "max_grad_norm": 1.0,
        "accounting_method": "rdp",
        "poisson_sampling": True,
        "grad_sample_mode": "hooks"
    }
}

def get_config(dataset_name: str, n_clients: int, sigma: float) -> dict:
    """Get configuration for dataset, overriding with UI parameters."""
    if dataset_name.lower() == "wesad":
        config = WESAD_CONFIG.copy()
    elif dataset_name.lower() == "sleep-edf":
        config = SLEEP_EDF_CONFIG.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Override with UI parameters
    config["federated_learning"]["n_clients"] = n_clients
    config["differential_privacy"]["noise_multiplier"] = sigma
    
    return config

