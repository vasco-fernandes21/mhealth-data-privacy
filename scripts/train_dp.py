#!/usr/bin/env python3
"""
Train model with Differential Privacy using Opacus.

Usage:
    python scripts/train_dp.py --dataset sleep-edf --epsilon 1.0
    python scripts/train_dp.py --dataset wesad --epsilon 5.0
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import setup_logging, get_logger, ExperimentLogger
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.training.trainers.dp_trainer import DPTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_augmented_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


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


def get_model(dataset: str, config: dict, device: str):
    """Create model based on dataset."""
    if dataset == 'sleep-edf':
        return SleepEDFModel(config, device=device)
    elif dataset == 'wesad':
        return WESADModel(config, device=device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


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


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size: int, num_workers: int = 4,
                       drop_last: bool = True):
    """Create data loaders (with drop_last for DP fixed batch size)."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=drop_last  # IMPORTANT for DP
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader


def train_with_dp(dataset: str, epsilon: float, data_dir: str,
                 config_dir: str, output_dir: str, seed: int, device: str):
    """Train with DP."""
    
    set_reproducible(seed=seed, device=device, verbose=True)
    
    experiment_name = f"dp_{dataset}_eps{epsilon}"
    with ExperimentLogger(experiment_name, output_dir) as logger:
        logger.info(f"Training {dataset} with DP (ε={epsilon})")
        logger.info(f"Seed: {seed}, Device: {device}")
        
        # Load configs
        default_cfg = load_config(Path(config_dir) / 'training_defaults.yaml')
        privacy_cfg = load_config(Path(config_dir) / 'privacy_defaults.yaml')
        dataset_cfg = load_config(Path(config_dir) / f'{dataset}.yaml')
        
        # Merge and override epsilon
        config = merge_configs(default_cfg, privacy_cfg, dataset_cfg)
        config['differential_privacy']['enabled'] = True
        config['differential_privacy']['target_epsilon'] = epsilon
        
        logger.info(f"Configuration loaded with ε={epsilon}")
        
        # Load data
        logger.info(f"Loading {dataset} data...")
        X_train, X_val, X_test, y_train, y_val, y_test, info = \
            load_data(dataset, data_dir)
        logger.info(f"Data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # Create dataloaders (with drop_last for DP)
        batch_size = config['training']['batch_size']
        num_workers = config['training'].get('num_workers', 4)
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size, num_workers, drop_last=True
        )
        logger.info(f"Data loaders created (batch_size={batch_size}, drop_last=True for DP)")
        
        # Create model
        model = get_model(dataset, config, device)
        logger.info(f"Model: {model.__class__.__name__}")
        
        # Create trainer
        trainer = DPTrainer(model, config, device=device)
        logger.info("DP Trainer created")
        
        # Train
        logger.info("Starting DP training...")
        results = trainer.fit(
            train_loader,
            val_loader,
            epochs=config['training']['epochs'],
            patience=config['training'].get('early_stopping_patience', 8),
            output_dir=str(Path(output_dir) / 'dp' / f'epsilon_{epsilon}' / dataset)
        )
        
        logger.info(f"Training completed in {results['training_time_seconds']:.1f}s")
        logger.info(f"Final epsilon: {results['final_epsilon']:.4f}")
        logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate_full(test_loader)
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-score: {test_metrics['f1_score']:.4f}")
        
        # Combine results
        final_results = {
            **results,
            **test_metrics,
            'epsilon': epsilon
        }
        
        # Save results
        output_path = Path(output_dir) / 'dp' / f'epsilon_{epsilon}' / dataset
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path / 'results.json'}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description='Train with Differential Privacy')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad'],
                       default='sleep-edf', help='Dataset')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Privacy budget epsilon')
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Setup logging
    setup_logging(output_dir=args.output_dir, level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*70)
    logger.info("DP TRAINING - DIFFERENTIAL PRIVACY")
    logger.info("="*70)
    
    train_with_dp(args.dataset, args.epsilon, args.data_dir,
                 args.config_dir, args.output_dir, args.seed, device)
    
    logger.info("\n" + "="*70)
    logger.info("✅ DP TRAINING COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    sys.exit(main())