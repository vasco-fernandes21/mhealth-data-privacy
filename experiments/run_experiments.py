#!/usr/bin/env python3
"""
Experiment Runner - Optimized for MLP + Features
With integrated data caching for faster experiment runs
"""

import sys
import yaml
import json
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import time
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import get_logger
from src.models import UnifiedMLPModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer

from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_processed_wesad


# ================================================================
# DATA CACHE - Prevents reloading data for each run
# ================================================================
class DataCache:
    """Global data cache - loads data once, reuses N times."""
    
    _cache: Dict[str, Tuple] = {}
    
    @classmethod
    def load_data(cls, dataset: str, data_dir: Path) -> Tuple:
        """Load dataset with caching."""
        if dataset in cls._cache:
            print(f"✓ Using cached {dataset}", flush=True)
            return cls._cache[dataset]
        
        print(f"Loading {dataset}...", flush=True)
        data_path = data_dir / dataset
        
        try:
            if dataset == 'sleep-edf':
                data = load_windowed_sleep_edf(str(data_path))
            elif dataset == 'wesad':
                data = load_processed_wesad(str(data_path))
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            # Cacheia para futuras chamadas
            cls._cache[dataset] = data
            return data
        
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset}: {e}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear cache."""
        cls._cache.clear()
        print("✓ Cache cleared", flush=True)
    
    @classmethod
    def get_stats(cls) -> Dict[str, str]:
        """Get cache statistics."""
        stats = {}
        for dataset, data in cls._cache.items():
            if isinstance(data, tuple) and len(data) >= 6:
                X_train, X_val, X_test = data[0], data[1], data[2]
                size_mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes) / 1e6
                stats[dataset] = f"{size_mb:.1f} MB"
        return stats


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_seed_everywhere(seed: int, device: str) -> None:
    """Set seed EVERYWHERE before any random operation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size=64, device='cpu'):
    """Create dataloaders for standard training."""
    from torch.utils.data import TensorDataset, DataLoader

    pin_memory = device == 'cuda'

    def make_loader(X, y, shuffle):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=pin_memory
        )

    return (
        make_loader(X_train, y_train, True),
        make_loader(X_val, y_val, False),
        make_loader(X_test, y_test, False)
    )


def create_fl_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                         n_clients, batch_size=64, device='cpu'):
    """Create dataloaders for federated learning."""
    from torch.utils.data import TensorDataset, DataLoader

    pin_memory = device == 'cuda'
    train_loaders = []
    val_loaders = []

    n_train = len(X_train)
    n_val = len(X_val)
    train_per_client = n_train // n_clients
    val_per_client = n_val // n_clients

    for i in range(n_clients):
        start_train = i * train_per_client
        end_train = (start_train + train_per_client 
                    if i < n_clients - 1 else n_train)

        X_client = torch.from_numpy(
            X_train[start_train:end_train].astype(np.float32)
        )
        y_client = torch.from_numpy(
            y_train[start_train:end_train].astype(np.int64)
        )
        train_dataset = TensorDataset(X_client, y_client)
        train_loaders.append(DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_memory
        ))

        start_val = i * val_per_client
        end_val = start_val + val_per_client if i < n_clients - 1 else n_val

        X_val_client = torch.from_numpy(
            X_val[start_val:end_val].astype(np.float32)
        )
        y_val_client = torch.from_numpy(
            y_val[start_val:end_val].astype(np.int64)
        )
        val_dataset = TensorDataset(X_val_client, y_val_client)
        val_loaders.append(DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory
        ))

    X_test_t = torch.from_numpy(X_test.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )

    return train_loaders, val_loaders, test_loader


# ================================================================
# EXPERIMENT RUNNER
# ================================================================

class ExperimentRunner:
    """Experiment runner for MLP + Features with data caching."""

    def __init__(self, scenarios_dir='experiments/scenarios',
                 data_dir='./data/processed',
                 config_dir='./src/configs',
                 results_dir='./results'):
        self.scenarios_dir = Path(scenarios_dir)
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results = []
        self.logger = get_logger(__name__)
        self.data_cache = DataCache()

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config not found: {config_path}")
            raise

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge configs."""
        if override is None:
            return deepcopy(base) if base else {}
        if base is None:
            return deepcopy(override) if override else {}
        
        result = deepcopy(base)
        for key, value in override.items():
            if (key in result and isinstance(result[key], dict) and
                    isinstance(value, dict)):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _get_config(self, dataset: str, method: str, hyperparams=None):
        """Get merged config with proper defaults."""
        dataset_cfg = self._load_config(
            self.config_dir / 'datasets' / f'{dataset}.yaml'
        )
        method_cfg = self._load_config(
            self.config_dir / 'methods' / f'{method}.yaml'
        )
        
        if dataset_cfg is None:
            raise ValueError(f"Dataset config not found: {dataset}")
        if method_cfg is None:
            raise ValueError(f"Method config not found: {method}")
        
        # Baseline uses dataset training config
        if method == 'baseline':
            config = deepcopy(dataset_cfg)
        else:
            config = self._merge_configs(dataset_cfg, method_cfg)

        # Apply hyperparameter overrides
        if hyperparams:
            for section, params in hyperparams.items():
                config[section] = self._merge_configs(
                    config.get(section, {}), params
                )
        
        return config

    def load_scenario(self, scenario_name: str) -> Dict:
        """Load scenario YAML."""
        with open(self.scenarios_dir / f'{scenario_name}.yaml', 'r') as f:
            return yaml.safe_load(f)

    def _load_data(self, dataset: str) -> Tuple:
        """Load dataset using cache."""
        return self.data_cache.load_data(dataset, self.data_dir)

    def run_experiment(self, exp_name: str, exp_config: Dict,
                      device: str) -> Dict:
        """Execute single experiment."""
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']

        print(f"\n{'='*60}")
        print(f"{exp_name} | {dataset} | {method} | seed={seed}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Set seed first
            set_seed_everywhere(seed=seed, device=device)
            
            config = self._get_config(
                dataset, method, exp_config.get('hyperparameters')
            )

            X_train, X_val, X_test, y_train, y_val, y_test = (
                self._load_data(dataset)[:6]
            )

            print(f"Data: train={X_train.shape} val={X_val.shape} "
                  f"test={X_test.shape}")

            config['dataset']['input_dim'] = X_train.shape[1]

            model = UnifiedMLPModel(config, device=device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params:,} params")

            output_dir = (self.results_dir / method / dataset /
                         f'seed_{seed}')
            output_dir.mkdir(parents=True, exist_ok=True)

            batch_size = config['training'].get('batch_size', 64)
            epochs = config['training'].get('epochs', 40)
            patience = config['training'].get('early_stopping_patience', 10)

            # Train
            if method == 'baseline':
                train_loader, val_loader, test_loader = create_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, device
                )
                trainer = BaselineTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loader, val_loader, epochs=epochs,
                    patience=patience, output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)

            elif method == 'dp':
                train_loader, val_loader, test_loader = create_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, device
                )
                trainer = DPTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loader, val_loader, epochs=epochs,
                    patience=patience, output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)

            elif method == 'fl':
                n_clients = config['federated_learning'].get(
                    'n_clients', 5
                )
                train_loaders, val_loaders, test_loader = (
                    create_fl_dataloaders(
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        n_clients, batch_size, device
                    )
                )
                client_ids = [f"client_{i:02d}" for i in range(n_clients)]
                trainer = FLTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loaders, val_loaders, client_ids=client_ids,
                    epochs=config['federated_learning'].get(
                        'global_rounds', 40
                    ),
                    patience=patience, output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)

            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = time.time() - start_time

            results_data = {
                'experiment_info': {
                    'name': exp_name,
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                },
                'training_metrics': {
                    'total_epochs': training_results.get('total_epochs', 0),
                    'best_val_acc': training_results.get('best_val_acc', 0),
                    'training_time_seconds': training_results.get(
                        'training_time_seconds', 0
                    ),
                },
                'test_metrics': {
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'f1_score': test_metrics.get('f1_score', 0),
                },
                'timing': {
                    'total_time_seconds': elapsed,
                }
            }

            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"✓ {test_metrics.get('accuracy', 0):.4f} acc "
                  f"| {test_metrics.get('f1_score', 0):.4f} f1 "
                  f"| {elapsed:.1f}s")

            result = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': True,
                'accuracy': test_metrics.get('accuracy', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'time_seconds': elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Failed: {e}")
            self.logger.error(f"{exp_name} failed: {e}", exc_info=True)

            result = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': False,
                'error': str(e),
                'time_seconds': elapsed,
            }

        self.results.append(result)
        return result

    def run_all(self, experiments: Dict, device: str):
        """Run all experiments."""
        total = len(experiments)
        print(f"\n{'='*60}")
        print(f"Running {total} experiments on {device.upper()}")
        print(f"{'='*60}")

        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"\n[{idx}/{total}]", flush=True)
            self.run_experiment(exp_name, exp_config, device)

        return self.results

    def save_results(self, output_file='experiments/results_log.json'):
        """Save results summary."""
        if not self.results:
            print("No results to save")
            return

        successful = [r for r in self.results if r['success']]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            'average_accuracy': (np.mean([r['accuracy'] for r in successful])
                                if successful else 0),
            'average_f1': (np.mean([r['f1_score'] for r in successful])
                          if successful else 0),
            'total_time_hours': sum(r['time_seconds']
                                   for r in self.results) / 3600,
            'results': self.results
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Success: {summary['successful']}/{summary['total_experiments']}")
        print(f"Avg Accuracy: {summary['average_accuracy']:.4f}")
        print(f"Avg F1: {summary['average_f1']:.4f}")
        print(f"Time: {summary['total_time_hours']:.2f}h")
        print(f"Saved: {output_file}\n")
        
        # Show cache stats
        cache_stats = self.data_cache.get_stats()
        if cache_stats:
            print("Cache Statistics:")
            for dataset, size in cache_stats.items():
                print(f"  {dataset}: {size}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment runner for MLP + Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiments.py --scenario baseline
  python experiments/run_experiments.py --scenario baseline \\
    --n_experiments 3 --auto
  python experiments/run_experiments.py --scenario all --datasets wesad \\
    --auto
        """
    )

    parser.add_argument('--scenario', 
                       choices=['baseline', 'dp', 'fl', 'all'],
                       default='baseline', help='Scenario to run')
    parser.add_argument('--device', default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--datasets',
                       help='Datasets (comma-separated: sleep-edf,wesad)')
    parser.add_argument('--n_experiments', type=int,
                       help='Limit number of experiments')
    parser.add_argument('--auto', action='store_true', 
                       help='Skip confirmation')
    parser.add_argument('--output_file', 
                       default='experiments/results_log.json')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else args.device
    print(f"\nDevice: {device}\n")

    if device == 'cuda':
        torch.cuda.empty_cache()

    runner = ExperimentRunner()
    scenarios = ['baseline', 'dp', 'fl'] if args.scenario == 'all' else [
        args.scenario
    ]

    all_experiments = {}
    for scenario in scenarios:
        try:
            scenario_data = runner.load_scenario(scenario)
            experiments = scenario_data.get('experiments', {})
            enabled = {
                name: config for name, config in experiments.items()
                if isinstance(config, dict) and config.get('enabled', True)
            }
            all_experiments.update(enabled)
        except FileNotFoundError:
            print(f"Warning: Scenario {scenario} not found")

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
        all_experiments = {
            name: config for name, config in all_experiments.items()
            if config.get('dataset') in datasets
        }

    if args.n_experiments:
        all_experiments = dict(
            list(all_experiments.items())[:args.n_experiments]
        )

    if not all_experiments:
        print("Error: No experiments found")
        return 1

    print(f"{'='*60}")
    print(f"Will run {len(all_experiments)} experiments:")
    for exp_name in list(all_experiments.keys())[:5]:
        print(f"  - {exp_name}")
    if len(all_experiments) > 5:
        print(f"  ... and {len(all_experiments) - 5} more")
    print(f"{'='*60}")

    if not args.auto:
        response = input("\nProceed? (y/n): ").lower().strip()
        if response != 'y':
            return 0

    start_time = time.time()
    try:
        runner.run_all(all_experiments, device)
    finally:
        # Always clear cache at the end
        runner.data_cache.clear()
    
    elapsed = time.time() - start_time

    runner.save_results(args.output_file)
    print(f"Total time: {elapsed/60:.1f} minutes\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())