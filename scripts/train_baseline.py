#!/usr/bin/env python3
"""
Train baseline model (without privacy).

Supports multiple seeds for statistical significance.

Usage:
    python scripts/train_baseline.py --dataset wesad --seeds 3
    python scripts/train_baseline.py --dataset sleep-edf --seeds 3
    python scripts/train_baseline.py --dataset all --seeds 3
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import torch
import numpy as np
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import setup_logging, get_logger, ExperimentLogger
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_augmented_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(target: dict, source: dict) -> dict:
    """Deep merge source into target dictionary.
    
    Recursively merges nested dictionaries instead of overwriting.
    """
    for key, value in source.items():
        if (key in target and 
            isinstance(target[key], dict) and 
            isinstance(value, dict)):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def merge_configs(*configs) -> dict:
    """Deep merge multiple config dictionaries."""
    merged = {}
    for cfg in configs:
        if cfg is not None:
            deep_merge(merged, deepcopy(cfg))
    
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
        (X_train, X_val, X_test, y_train, y_val, y_test,
         scaler, info, subjects_train) = load_windowed_sleep_edf(str(data_path))
    elif dataset == 'wesad':
        (X_train, X_val, X_test, y_train, y_val, y_test,
         label_encoder, info) = load_augmented_wesad_temporal(str(data_path))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, info


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size: int, num_workers: int = 4):
    """Create data loaders."""
    
    # ✅ Calculate flags correctly
    persistent_workers = num_workers > 0
    pin_memory = num_workers > 0

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
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader, test_loader


def train_dataset_with_seed(dataset: str, seed: int, data_dir: str,
                            config_dir: str, output_dir: str,
                            device: str) -> dict:
    """Train baseline for a single seed."""
    
    set_reproducible(seed=seed, device=device, verbose=False)
    
    # Load configs with deep merge
    default_cfg = load_config(Path(config_dir) / 'training_defaults.yaml')
    
    # Map dataset names to config file names
    config_mapping = {
        'sleep-edf': 'sleep_edf.yaml',
        'wesad': 'wesad.yaml'
    }
    config_filename = config_mapping.get(dataset, f'{dataset}.yaml')
    dataset_cfg = load_config(Path(config_dir) / config_filename)
    config = merge_configs(default_cfg, dataset_cfg)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, info = \
        load_data(dataset, data_dir)
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 0)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size, num_workers
    )
    
    # Create model
    model = get_model(dataset, config, device)
    
    # Create trainer
    trainer = BaselineTrainer(model, config, device=device)
    
    # Train
    results = trainer.fit(
        train_loader,
        val_loader,
        epochs=config['training']['epochs'],
        patience=config['training'].get('early_stopping_patience', 10),
        output_dir=str(Path(output_dir) / 'baseline' / dataset / f'seed_{seed}')
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_full(test_loader)
    
    # Combine results
    final_results = {
        'seed': seed,
        **results,
        **test_metrics
    }
    
    return final_results


def train_dataset(dataset: str, data_dir: str, config_dir: str,
                 output_dir: str, device: str, num_seeds: int = 3,
                 base_seed: int = 42):
    """Train baseline for a dataset with multiple seeds."""
    
    experiment_name = f"baseline_{dataset}"
    with ExperimentLogger(experiment_name, output_dir) as logger:
        logger.info(f"Training {dataset} baseline with {num_seeds} seeds")
        logger.info(f"Device: {device}")
        
        # Generate seeds
        seeds = [base_seed + i * 100 for i in range(num_seeds)]
        logger.info(f"Seeds: {seeds}")
        
        all_results = []
        
        # Train with each seed
        for seed in seeds:
            logger.info(f"\nTraining with seed {seed}...")
            
            results = train_dataset_with_seed(
                dataset, seed, data_dir, config_dir,
                output_dir, device
            )
            all_results.append(results)
            
            logger.info(
                f"  Accuracy: {results['accuracy']:.4f}, "
                f"F1: {results['f1_score']:.4f}"
            )
        
        # Aggregate statistics
        logger.info(f"\n{'='*70}")
        logger.info("BASELINE RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        
        accuracy_scores = [r['accuracy'] for r in all_results]
        precision_scores = [r['precision'] for r in all_results]
        recall_scores = [r['recall'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        
        summary = {
            'dataset': dataset,
            'num_seeds': num_seeds,
            'seeds': seeds,
            'accuracy': {
                'mean': float(np.mean(accuracy_scores)),
                'std': float(np.std(accuracy_scores)),
                'values': accuracy_scores
            },
            'precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores)),
                'values': precision_scores
            },
            'recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores)),
                'values': recall_scores
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'values': f1_scores
            },
            'all_runs': all_results
        }
        
        # Log summary
        logger.info(
            f"Accuracy:  {summary['accuracy']['mean']:.4f} ± "
            f"{summary['accuracy']['std']:.4f}"
        )
        logger.info(
            f"Precision: {summary['precision']['mean']:.4f} ± "
            f"{summary['precision']['std']:.4f}"
        )
        logger.info(
            f"Recall:    {summary['recall']['mean']:.4f} ± "
            f"{summary['recall']['std']:.4f}"
        )
        logger.info(
            f"F1-Score:  {summary['f1_score']['mean']:.4f} ± "
            f"{summary['f1_score']['std']:.4f}"
        )
        
        # Save summary
        output_path = Path(output_dir) / 'baseline' / dataset
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSummary saved to {output_path / 'summary.json'}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument(
        '--dataset',
        choices=['sleep-edf', 'wesad', 'all'],
        default='all',
        help='Dataset to train on'
    )
    parser.add_argument(
        '--data_dir',
        default='./data/processed',
        help='Path to processed data'
    )
    parser.add_argument(
        '--config_dir',
        default='./src/configs',
        help='Path to config files'
    )
    parser.add_argument(
        '--output_dir',
        default='./results',
        help='Path to save results'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device (cuda, cpu, auto)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=3,
        help='Number of random seeds to try'
    )
    parser.add_argument(
        '--base_seed',
        type=int,
        default=42,
        help='Base seed for reproducibility'
    )
    
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
    logger.info(f"Number of seeds: {args.seeds}")
    logger.info(f"Device: {device}")
    
    # Train
    datasets = (
        ['sleep-edf', 'wesad']
        if args.dataset == 'all'
        else [args.dataset]
    )
    
    all_summaries = {}
    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {dataset}...")
        logger.info(f"{'='*70}")
        
        summary = train_dataset(
            dataset, args.data_dir, args.config_dir,
            args.output_dir, device, num_seeds=args.seeds,
            base_seed=args.base_seed
        )
        all_summaries[dataset] = summary
    
    # Global summary
    logger.info("\n" + "="*70)
    logger.info("✅ ALL BASELINES TRAINED")
    logger.info("="*70)
    
    for dataset, summary in all_summaries.items():
        logger.info(
            f"\n{dataset}:"
            f" Accuracy={summary['accuracy']['mean']:.4f}±"
            f"{summary['accuracy']['std']:.4f}, "
            f"F1={summary['f1_score']['mean']:.4f}±"
            f"{summary['f1_score']['std']:.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())