#!/usr/bin/env python3
"""
Train Federated Learning model.

Usage:
    python scripts/train_fl.py --dataset wesad --n_clients 1 --seed 42
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import setup_logging, get_logger, ExperimentLogger
from src.models.model_factory import create_model
from src.training.trainers.fl_trainer import FLTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_processed_wesad_temporal
from src.preprocessing.common import get_subject_splits
from torch.utils.data import TensorDataset, DataLoader


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs) -> dict:
    """Merge configuration dictionaries."""
    merged = {}
    for cfg in configs:
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = merge_configs(merged[k], v)
                else:
                    merged[k] = v
    return merged


def load_data(dataset: str, data_dir: str):
    """Load dataset."""
    data_path = Path(data_dir) / dataset
    
    if dataset == 'sleep-edf':
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects_train = \
            load_windowed_sleep_edf(str(data_path))
    elif dataset == 'wesad':
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = \
            load_processed_wesad_temporal(str(data_path))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, info


def create_client_dataloaders(X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             n_clients: int,
                             batch_size: int = 32) -> tuple:
    """
    Create dataloaders for each client (simulate data partitioning).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_clients: Number of clients
        batch_size: Batch size
    
    Returns:
        (train_loaders, val_loaders, client_ids)
    """
    n_samples = len(X_train)
    samples_per_client = n_samples // n_clients
    
    train_loaders = []
    val_loaders = []
    client_ids = []
    
    for client_id in range(n_clients):
        # Split training data
        start_idx = client_id * samples_per_client
        end_idx = (
            start_idx + samples_per_client
            if client_id < n_clients - 1
            else n_samples
        )
        
        X_client_train = X_train[start_idx:end_idx]
        y_client_train = y_train[start_idx:end_idx]
        
        # For validation, use a small subset for each client
        val_start = client_id * (len(X_val) // n_clients)
        val_end = val_start + (len(X_val) // n_clients)
        
        X_client_val = X_val[val_start:val_end]
        y_client_val = y_val[val_start:val_end]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(X_client_train, dtype=torch.float32),
            torch.tensor(y_client_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_client_val, dtype=torch.float32),
            torch.tensor(y_client_val, dtype=torch.long)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        client_ids.append(f"client_{client_id:02d}")
    
    return train_loaders, val_loaders, client_ids


def print_config(config: dict, n_clients: int) -> None:
    """Print configuration for verification."""
    print("\n" + "="*70, flush=True)
    print("CONFIGURATION VERIFICATION", flush=True)
    print("="*70, flush=True)
    
    print("\nTRAINING (should match WESAD baseline):", flush=True)
    print(f"  Batch size: {config['training']['batch_size']}", flush=True)
    print(f"  Learning rate: {config['training']['learning_rate']}", flush=True)
    print(f"  Optimizer: {config['training']['optimizer']}", flush=True)
    print(f"  Weight decay: {config['training']['weight_decay']}", flush=True)
    print(
        f"  LR Scheduler: {config['training'].get('lr_scheduler', 'none')}",
        flush=True
    )
    print(
        f"  Warmup epochs: {config['training'].get('warmup_epochs', 0)}",
        flush=True
    )
    print(
        f"  Early stopping patience: "
        f"{config['training']['early_stopping_patience']}",
        flush=True
    )
    print(
        f"  Gradient clipping: {config['training']['gradient_clipping']}",
        flush=True
    )
    
    print("\nFEDERATED LEARNING:", flush=True)
    print(f"  Number of clients: {n_clients}", flush=True)
    print(
        f"  Global rounds: {config['federated_learning']['global_rounds']}",
        flush=True
    )
    print(
        f"  Local epochs: {config['federated_learning']['local_epochs']}",
        flush=True
    )
    print(
        f"  Local batch size: "
        f"{config['federated_learning']['local_batch_size']}",
        flush=True
    )
    print(
        f"  Aggregation: {config['federated_learning']['aggregation_method']}",
        flush=True
    )
    
    print("\nMODEL:", flush=True)
    print(
        f"  Architecture: {config['model']['architecture']}", flush=True
    )
    print(f"  LSTM units: {config['model']['lstm_units']}", flush=True)
    print(f"  Dropout: {config['model']['dropout']}", flush=True)
    print(f"  Loss: {config['training']['loss']}", flush=True)
    print("="*70 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Train with Federated Learning'
    )
    parser.add_argument(
        '--dataset', choices=['sleep-edf', 'wesad'],
        default='wesad'
    )
    parser.add_argument(
        '--n_clients', type=int, default=1,
        help='Number of FL clients'
    )
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    # Unbuffered output
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

    args = parser.parse_args()
    
    # Device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    set_reproducible(seed=args.seed, device=device, verbose=True)
    
    # Setup logging
    setup_logging(output_dir=args.output_dir, level='INFO')
    logger = get_logger(__name__)
    
    experiment_name = f"fl_{args.dataset}_clients{args.n_clients}"
    with ExperimentLogger(experiment_name, args.output_dir) as exp_logger:
        exp_logger.info(
            f"Training {args.dataset} with FL ({args.n_clients} clients)"
        )
        
        # Load all configs
        default_cfg = load_config(
            Path(args.config_dir) / 'training_defaults.yaml'
        )
        privacy_cfg = load_config(
            Path(args.config_dir) / 'privacy_defaults.yaml'
        )
        
        config_mapping = {
            'sleep-edf': 'sleep_edf.yaml',
            'wesad': 'wesad.yaml'
        }
        config_filename = config_mapping.get(
            args.dataset, f'{args.dataset}.yaml'
        )
        dataset_cfg = load_config(
            Path(args.config_dir) / config_filename
        )
        
        # Merge: training defaults → privacy defaults → dataset-specific
        config = merge_configs(default_cfg, privacy_cfg, dataset_cfg)
        
        exp_logger.info("Configuration loaded")
        
        # Print config for verification
        print_config(config, args.n_clients)
        
        # Load data
        exp_logger.info(f"Loading {args.dataset} data...")
        X_train, X_val, X_test, y_train, y_val, y_test, info = \
            load_data(args.dataset, args.data_dir)
        exp_logger.info(
            f"Data loaded: Train={X_train.shape}, Val={X_val.shape}, "
            f"Test={X_test.shape}"
        )
        
        # Create client dataloaders
        exp_logger.info(f"Creating {args.n_clients} clients...")
        train_loaders, val_loaders, client_ids = (
            create_client_dataloaders(
                X_train, y_train, X_val, y_val,
                n_clients=args.n_clients,
                batch_size=config['training']['batch_size']
            )
        )
        exp_logger.info(f"Clients created with IDs: {client_ids}")
        
        # Create model
        model = create_model(args.dataset, config, device=device)
        exp_logger.info(f"Model created: {model.__class__.__name__}")
        
        # Create trainer
        trainer = FLTrainer(model, config, device=device)
        exp_logger.info("FL Trainer created")
        
        # Train
        exp_logger.info("Starting FL training...")
        
        # Get patience from config (not hardcoded!)
        patience = config['training'].get('early_stopping_patience', 30)
        global_rounds = config['federated_learning']['global_rounds']
        
        results = trainer.fit(
            train_loaders,
            val_loaders,
            client_ids=client_ids,
            epochs=global_rounds,
            patience=patience,
            output_dir=str(Path(args.output_dir) / 'fl' / args.dataset)
        )
        
        exp_logger.info(
            f"Training completed in {results['training_time_seconds']:.1f}s"
        )
        exp_logger.info(
            f"Best validation accuracy: {results['best_val_acc']:.4f}"
        )
        
        # Save results
        output_path = Path(args.output_dir) / 'fl' / args.dataset
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        exp_logger.info(f"Results saved to {output_path / 'results.json'}")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())