#!/usr/bin/env python3
"""
Train baseline model (without privacy).

Usage:
    python scripts/train_baseline.py --dataset sleep-edf
    python scripts/train_baseline.py --dataset wesad
    python scripts/train_baseline.py --dataset all
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible, set_all_seeds
from src.utils.logging_utils import setup_logging, get_logger, ExperimentLogger
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_augmented_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs) -> dict:
    """Merge multiple config dictionaries."""
    merged = {}
    for cfg in configs:
        merged.update(cfg)
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
                       batch_size: int, num_workers: int = 4):
    """Create data loaders."""
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
        drop_last=False
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


def train_dataset(dataset: str, data_dir: str, config_dir: str, 
                 output_dir: str, seed: int, device: str):
    """Train baseline for a dataset."""
    
    # Setup
    set_reproducible(seed=seed, device=device, verbose=True)
    
    experiment_name = f"baseline_{dataset}"
    with ExperimentLogger(experiment_name, output_dir) as logger:
        logger.info(f"Training {dataset} baseline")
        logger.info(f"Seed: {seed}")
        logger.info(f"Device: {device}")
        
        # Load configs
        default_cfg = load_config(Path(config_dir) / 'training_defaults.yaml')
        dataset_cfg = load_config(Path(config_dir) / f'{dataset}.yaml')
        config = merge_configs(default_cfg, dataset_cfg)
        
        logger.info(f"Configuration loaded")
        
        # Load data
        logger.info(f"Loading {dataset} data...")
        X_train, X_val, X_test, y_train, y_val, y_test, info = \
            load_data(dataset, data_dir)
        logger.info(f"Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # Create dataloaders
        batch_size = config['training']['batch_size']
        num_workers = config['training'].get('num_workers', 4)
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size, num_workers
        )
        logger.info(f"Data loaders created: batch_size={batch_size}")
        
        # Create model
        model = get_model(dataset, config, device)
        logger.info(f"Model created: {model.__class__.__name__}")
        model.print_model_summary()
        
        # Create trainer
        trainer = BaselineTrainer(model, config, device=device)
        logger.info("Trainer created")
        
        # Train
        logger.info("Starting training...")
        results = trainer.fit(
            train_loader,
            val_loader,
            epochs=config['training']['epochs'],
            patience=config['training'].get('early_stopping_patience', 8),
            output_dir=str(Path(output_dir) / 'baseline' / dataset)
        )
        
        logger.info(f"Training completed in {results['training_time_seconds']:.1f}s")
        logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate_full(test_loader)
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-score: {test_metrics['f1_score']:.4f}")
        
        # Combine results
        final_results = {
            **results,
            **test_metrics
        }
        
        # Save results
        output_path = Path(output_dir) / 'baseline' / dataset
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path / 'results.json'}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad', 'all'],
                       default='all', help='Dataset to train on')
    parser.add_argument('--data_dir', default='./data/processed',
                       help='Path to processed data')
    parser.add_argument('--config_dir', default='./src/configs',
                       help='Path to config files')
    parser.add_argument('--output_dir', default='./results',
                       help='Path to save results')
    parser.add_argument('--device', default=None,
                       help='Device (cuda, cpu, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Setup logging
    setup_logging(output_dir=args.output_dir, level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*70)
    logger.info("BASELINE TRAINING - NO PRIVACY")
    logger.info("="*70)
    
    # Train
    datasets = ['sleep-edf', 'wesad'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        logger.info(f"\nTraining {dataset}...")
        train_dataset(dataset, args.data_dir, args.config_dir,
                     args.output_dir, args.seed, device)
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL BASELINES TRAINED")
    logger.info("="*70)


if __name__ == "__main__":
    sys.exit(main())