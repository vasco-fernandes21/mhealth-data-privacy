#!/usr/bin/env python3
"""
Train Federated Learning + Differential Privacy model.

Usage:
    python scripts/train_fl_dp.py --dataset sleep-edf --epsilon 1.0 --n_clients 5
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
from src.training.trainers.fl_dp_trainer import FLDPTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_augmented_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs) -> dict:
    """Merge configurations."""
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
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = \
            load_windowed_sleep_edf(str(data_path))
    elif dataset == 'wesad':
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = \
            load_augmented_wesad_temporal(str(data_path))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, info


def create_client_dataloaders(X_train, y_train, X_val, y_val,
                             n_clients: int, batch_size: int = 32) -> tuple:
    """Create dataloaders for FL clients."""
    n_samples = len(X_train)
    samples_per_client = n_samples // n_clients
    
    train_loaders = []
    val_loaders = []
    client_ids = []
    
    for client_id in range(n_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < n_clients - 1 else n_samples
        
        X_client_train = X_train[start_idx:end_idx]
        y_client_train = y_train[start_idx:end_idx]
        
        val_start = client_id * (len(X_val) // n_clients)
        val_end = val_start + (len(X_val) // n_clients)
        
        X_client_val = X_val[val_start:val_end]
        y_client_val = y_val[val_start:val_end]
        
        train_dataset = TensorDataset(
            torch.tensor(X_client_train, dtype=torch.float32),
            torch.tensor(y_client_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_client_val, dtype=torch.float32),
            torch.tensor(y_client_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True,
                                 drop_last=True)  # Important for DP
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        client_ids.append(f"client_{client_id:02d}")
    
    return train_loaders, val_loaders, client_ids


def main():
    parser = argparse.ArgumentParser(description='Train FL+DP')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad'],
                       default='sleep-edf')
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_reproducible(seed=args.seed, device=device, verbose=True)
    
    setup_logging(output_dir=args.output_dir, level='INFO')
    logger = get_logger(__name__)
    
    experiment_name = f"fl_dp_{args.dataset}_eps{args.epsilon}"
    with ExperimentLogger(experiment_name, args.output_dir) as exp_logger:
        exp_logger.info(f"Training {args.dataset} with FL+DP (ε={args.epsilon})")
        
        # Load and merge configs
        default_cfg = load_config(Path(args.config_dir) / 'training_defaults.yaml')
        privacy_cfg = load_config(Path(args.config_dir) / 'privacy_defaults.yaml')
        dataset_cfg = load_config(Path(args.config_dir) / f'{args.dataset}.yaml')
        
        config = merge_configs(default_cfg, privacy_cfg, dataset_cfg)
        config['differential_privacy']['enabled'] = True
        config['differential_privacy']['target_epsilon'] = args.epsilon
        
        exp_logger.info(f"Configuration loaded with ε={args.epsilon}")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, info = \
            load_data(args.dataset, args.data_dir)
        
        # Create client loaders
        train_loaders, val_loaders, client_ids = create_client_dataloaders(
            X_train, y_train, X_val, y_val,
            n_clients=args.n_clients
        )
        exp_logger.info(f"Created {args.n_clients} FL clients with DP enabled")
        
        # Create and train
        model = create_model(args.dataset, config, device=device)
        trainer = FLDPTrainer(model, config, device=device)
        
        exp_logger.info("Starting FL+DP training...")
        results = trainer.fit(
            train_loaders, val_loaders,
            client_ids=client_ids,
            epochs=config['federated_learning']['global_rounds'],
            patience=8,
            output_dir=str(Path(args.output_dir) / 'fl_dp' / f'epsilon_{args.epsilon}' / args.dataset)
        )
        
        exp_logger.info(f"Completed in {results['training_time_seconds']:.1f}s")
        
        # Save results
        output_path = Path(args.output_dir) / 'fl_dp' / f'epsilon_{args.epsilon}' / args.dataset
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())