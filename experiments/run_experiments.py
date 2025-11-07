#!/usr/bin/env python3
"""
Optimized experiment runner for Baseline, DP, FL, DP+FL training.
Works with pre-computed windows and HDF5 for maximum speed.
"""

import sys
import yaml
import json
import argparse
import torch
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import time
from copy import deepcopy
import numpy as np
from collections import defaultdict
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import get_logger
from src.models import UnifiedLSTMModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer

from src.preprocessing.sleep_edf import (
    load_windowed_sleep_edf,
    preprocess_sleep_edf
)
from src.preprocessing.wesad import load_processed_wesad_temporal


def get_auto_device() -> str:
    """Detect device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and \
         torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_name(device: str) -> str:
    """Get human-readable device name."""
    if device == 'auto':
        device = get_auto_device()

    if device == 'cuda' and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    elif device == 'mps' and hasattr(torch.backends, "mps") and \
         torch.backends.mps.is_available():
        return "mps"
    else:
        return device


class OptimizedDataLoaderFactory:
    """Factory for creating optimized dataloaders."""

    @staticmethod
    def create_standard_dataloaders(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        batch_size: int = 128,
        num_workers: int = 2
    ) -> Tuple:
        """Create optimized dataloaders for baseline/DP methods."""
        from torch.utils.data import Dataset, DataLoader

        class OptimizedDataset(Dataset):
            """Optimized dataset with contiguous arrays."""

            def __init__(self, X: np.ndarray, y: np.ndarray):
                self.X = np.ascontiguousarray(X, dtype=np.float32)
                self.y = np.ascontiguousarray(y, dtype=np.int64)

            def __len__(self) -> int:
                return len(self.X)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                return torch.from_numpy(self.X[idx]), self.y[idx]

        print("Creating optimized DataLoaders (standard)...")

        train_dataset = OptimizedDataset(X_train, y_train)
        val_dataset = OptimizedDataset(X_val, y_val)
        test_dataset = OptimizedDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )

        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader)} batches")
        print(f"  Test:  {len(test_loader)} batches\n")

        return train_loader, val_loader, test_loader

    @staticmethod
    def create_fl_dataloaders(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        n_clients: int,
        batch_size: int = 128,
        num_workers: int = 2
    ) -> Tuple[List, List]:
        """Create optimized FL dataloaders (partitioned by client)."""
        from torch.utils.data import Dataset, DataLoader

        class OptimizedDataset(Dataset):
            """Optimized dataset with contiguous arrays."""

            def __init__(self, X: np.ndarray, y: np.ndarray):
                self.X = np.ascontiguousarray(X, dtype=np.float32)
                self.y = np.ascontiguousarray(y, dtype=np.int64)

            def __len__(self) -> int:
                return len(self.X)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                return torch.from_numpy(self.X[idx]), self.y[idx]

        print(f"Creating FL DataLoaders for {n_clients} clients...")

        n_samples_train = len(X_train)
        n_samples_val = len(X_val)
        samples_per_client_train = n_samples_train // n_clients
        samples_per_client_val = n_samples_val // n_clients

        train_loaders = []
        val_loaders = []

        for client_id in range(n_clients):
            # Train split
            start_idx = client_id * samples_per_client_train
            end_idx = (start_idx + samples_per_client_train
                      if client_id < n_clients - 1 else n_samples_train)

            X_client_train = X_train[start_idx:end_idx]
            y_client_train = y_train[start_idx:end_idx]

            # Val split
            val_start = client_id * samples_per_client_val
            val_end = (val_start + samples_per_client_val
                      if client_id < n_clients - 1 else n_samples_val)

            X_client_val = X_val[val_start:val_end]
            y_client_val = y_val[val_start:val_end]

            # Create datasets
            train_dataset = OptimizedDataset(X_client_train, y_client_train)
            val_dataset = OptimizedDataset(X_client_val, y_client_val)

            # Create loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=False
            )

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        avg_batches = sum(len(l) for l in train_loaders) // n_clients
        print(f"FL DataLoaders created: {n_clients} clients, "
              f"avg_train_batches={avg_batches}\n")

        return train_loaders, val_loaders


class OptimizedExperimentRunner:
    """Optimized experiment runner with HDF5 + fast DataLoaders."""

    def __init__(self,
                 scenarios_dir: str = 'experiments/scenarios',
                 data_dir: str = './data/processed',
                 config_dir: str = './src/configs',
                 results_dir: str = './results'):
        self.scenarios_dir = Path(scenarios_dir)
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results = []
        self.logger = get_logger(__name__)
        self._data_cache = {}

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge override into base."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and \
               isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _get_config(self, dataset: str, method: str,
                    exp_hyperparams: Dict = None) -> dict:
        """Load and merge configs: dataset + method + experiment."""
        dataset_cfg = self._load_config(
            self.config_dir / 'datasets' / f'{dataset}.yaml'
        )
        method_cfg = self._load_config(
            self.config_dir / 'methods' / f'{method}.yaml'
        )

        config = self._deep_merge(dataset_cfg, method_cfg)

        if exp_hyperparams:
            for section, params in exp_hyperparams.items():
                config[section] = self._deep_merge(
                    config.get(section, {}), params
                )

        return config

    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load scenario YAML."""
        scenario_file = self.scenarios_dir / f'{scenario_name}.yaml'
        if not scenario_file.exists():
            raise FileNotFoundError(
                f"Scenario not found: {scenario_file}"
            )

        with open(scenario_file, 'r') as f:
            return yaml.safe_load(f)

    def filter_experiments(self, experiments: Dict,
                          tags: List[str] = None,
                          keywords: str = None,
                          datasets: List[str] = None,
                          methods: List[str] = None,
                          epsilon: float = None,
                          clients: int = None) -> Dict:
        """Filter experiments by criteria."""
        filtered = {}

        for exp_name, exp_config in experiments.items():
            if not exp_config.get('enabled', True):
                continue

            if tags and not any(tag in exp_config.get('tags', [])
                               for tag in tags):
                continue

            if keywords and keywords.lower() not in exp_name.lower():
                continue

            if datasets and exp_config.get('dataset') not in datasets:
                continue

            if methods and exp_config.get('method') not in methods:
                continue

            if epsilon is not None:
                hp = exp_config.get('hyperparameters', {})
                if hp.get('differential_privacy', {}).get(
                    'target_epsilon') != epsilon:
                    continue

            if clients is not None:
                hp = exp_config.get('hyperparameters', {})
                if hp.get('federated_learning', {}).get(
                    'n_clients') != clients:
                    continue

            filtered[exp_name] = exp_config

        return filtered

    def _load_data_cached(self, dataset: str) -> Tuple:
        """Load dataset with caching (auto-preprocess if needed)."""
        if dataset in self._data_cache:
            print(f"Using cached {dataset} data\n")
            return self._data_cache[dataset]

        data_path = self.data_dir / dataset

        if dataset == 'sleep-edf':
            hdf5_path = data_path / 'sleep_edf_data.h5'

            if not hdf5_path.exists():
                required_files = [
                    'X_train.npy', 'y_train.npy',
                    'X_val.npy', 'y_val.npy',
                    'X_test.npy', 'y_test.npy',
                    'preprocessing_info.pkl', 'scaler.pkl'
                ]

                all_exist = all(
                    (data_path / f).exists() for f in required_files
                )

                if not all_exist:
                    print(f"Data not found. Preprocessing {dataset}...\n")
                    raw_path = Path('./data/raw/sleep-edf')
                    if not raw_path.exists():
                        raise FileNotFoundError(
                            f"Raw data not found: {raw_path}"
                        )

                    preprocess_sleep_edf(
                        data_dir=str(raw_path),
                        output_dir=str(data_path),
                        force_reprocess=False
                    )

            data = load_windowed_sleep_edf(str(data_path))
            self._data_cache[dataset] = data
            return data

        elif dataset == 'wesad':
            data = load_processed_wesad_temporal(str(data_path))
            self._data_cache[dataset] = data
            return data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _get_output_dir_name(self, method: str, dataset: str, seed: int,
                            hyperparameters: Dict = None) -> str:
        """Generate unique output directory name."""
        base_name = f'seed_{seed}'

        if method == 'baseline':
            return base_name

        suffix_parts = []

        if hyperparameters:
            dp_cfg = hyperparameters.get('differential_privacy', {})
            if dp_cfg.get('noise_multiplier') is not None:
                sigma = dp_cfg['noise_multiplier']
                sigma_str = f'{sigma:.1f}'.replace('.', '_')
                suffix_parts.append(f'sigma{sigma_str}')

            fl_cfg = hyperparameters.get('federated_learning', {})
            if fl_cfg.get('n_clients') is not None:
                n_clients = fl_cfg['n_clients']
                suffix_parts.append(f'clients{n_clients}')

        if suffix_parts:
            return f"{base_name}_{'_'.join(suffix_parts)}"

        return base_name

    def run_experiment(self, exp_name: str, exp_config: Dict,
                      device: str) -> Dict:
        """Execute single experiment (optimized)."""
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']

        print(f"\n{'='*70}")
        print(f"{exp_name} | {dataset} | {method} | seed={seed}")
        print(f"{'='*70}")

        actual_device = get_auto_device() if device == 'auto' else device
        device_name = get_device_name(actual_device)
        print(f"Device: {device_name}\n")

        start_time = time.time()

        try:
            set_reproducible(seed=seed, device=actual_device, verbose=False)

            config = self._get_config(
                dataset, method,
                exp_config.get('hyperparameters')
            )

            num_workers = min(config['training'].get('num_workers', 2), 4)
            print(f"Config: batch_size={config['training'].get('batch_size')}, "
                  f"lr={config['training'].get('learning_rate')}, "
                  f"epochs={config['training'].get('epochs')}, "
                  f"num_workers={num_workers}\n")

            if method == 'dp':
                dp_cfg = config.get('differential_privacy', {})
                print(f"DP Config: σ={dp_cfg.get('noise_multiplier', 'N/A')}, "
                      f"ε_target={dp_cfg.get('target_epsilon', 'N/A')}\n")

            print("Loading data (optimized)...")
            data = self._load_data_cached(dataset)
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, \
                info, subjects = data

            print(f"Shapes: train={X_train.shape}, val={X_val.shape}, "
                  f"test={X_test.shape}")
            print(f"Memory: train={X_train.nbytes/(1024**2):.1f}MB\n")

            print("Creating model...")
            config['dataset']['input_dim'] = X_train.shape[1]
            model = UnifiedLSTMModel(config, device=actual_device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params:,} parameters\n")

            output_dir_name = self._get_output_dir_name(
                method, dataset, seed, exp_config.get('hyperparameters')
            )
            output_dir = (self.results_dir / method / dataset /
                         output_dir_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            result = self._run_method_optimized(
                method, model, config, X_train, y_train,
                X_val, y_val, X_test, y_test, actual_device, output_dir,
                num_workers
            )

            training_elapsed = time.time() - start_time
            test_metrics = result['test_metrics']

            results_file = output_dir / 'results.json'
            training_history = result['training_results'].get('history', {})

            detailed_results = {
                'experiment_info': {
                    'name': exp_name,
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'hyperparameters': exp_config.get('hyperparameters', {})
                },
                'training_metrics': {
                    'total_epochs': result['training_results'].get(
                        'total_epochs',
                        result['training_results'].get('total_rounds', 0)
                    ),
                    'best_val_acc': result['training_results'].get(
                        'best_val_acc', 0
                    ),
                    'training_time_seconds': result['training_results'].get(
                        'training_time_seconds', 0
                    ),
                    'final_train_loss': result['training_results'].get(
                        'final_train_loss',
                        training_history.get('train_loss', [0])[-1]
                        if training_history.get('train_loss') else 0
                    ),
                    'final_train_acc': result['training_results'].get(
                        'final_train_acc',
                        training_history.get('train_acc', [0])[-1]
                        if training_history.get('train_acc') else 0
                    ),
                    'final_val_loss': result['training_results'].get(
                        'final_val_loss',
                        training_history.get('val_loss', [0])[-1]
                        if training_history.get('val_loss') else 0
                    ),
                    'final_val_acc': result['training_results'].get(
                        'final_val_acc',
                        training_history.get('val_acc', [0])[-1]
                        if training_history.get('val_acc') else 0
                    )
                },
                'test_metrics': {
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'f1_score': test_metrics.get('f1_score', 0),
                    'precision_per_class': test_metrics.get(
                        'precision_per_class', []
                    ),
                    'recall_per_class': test_metrics.get(
                        'recall_per_class', []
                    ),
                    'f1_per_class': test_metrics.get('f1_per_class', []),
                    'confusion_matrix': test_metrics.get(
                        'confusion_matrix', []
                    ),
                    'class_names': test_metrics.get('class_names', [])
                },
                'training_history': training_history
            }

            if method == 'dp':
                detailed_results['privacy_metrics'] = {
                    'final_epsilon': test_metrics.get('final_epsilon', 0),
                    'target_epsilon': config.get(
                        'differential_privacy', {}
                    ).get('target_epsilon', 0),
                    'target_delta': config.get(
                        'differential_privacy', {}
                    ).get('target_delta', 0),
                    'noise_multiplier': config.get(
                        'differential_privacy', {}
                    ).get('noise_multiplier', 0),
                    'max_grad_norm': config.get(
                        'differential_privacy', {}
                    ).get('max_grad_norm', 0),
                    'privacy_budget_history': result['training_results'].get(
                        'privacy_budget_history', []
                    )
                }
            elif method == 'fl':
                detailed_results['federated_metrics'] = {
                    'n_clients': result['training_results'].get(
                        'n_clients', 0
                    ),
                    'total_rounds': result['training_results'].get(
                        'total_rounds', 0
                    ),
                    'local_epochs_per_round': config.get(
                        'federated_learning', {}
                    ).get('local_epochs', 1)
                }

            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)

            total_elapsed = time.time() - start_time

            print(f"✅ Completed in {total_elapsed:.1f}s")
            print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"   F1-Score: {test_metrics.get('f1_score', 0):.4f}")

            if method == 'dp' and 'final_epsilon' in test_metrics:
                print(f"   Final ε: {test_metrics['final_epsilon']:.4f}")

            print()

            final_results = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': True,
                'time_seconds': total_elapsed,
                'accuracy': test_metrics.get('accuracy', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'results_file': str(results_file),
                'timestamp': datetime.now().isoformat()
            }

            self.results.append(final_results)

            return final_results

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: {e}\n")
            import traceback
            traceback.print_exc()

            final_results = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': False,
                'time_seconds': elapsed,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            self.results.append(final_results)

            return final_results

    def _run_method_optimized(self, method, model, config, X_train, y_train,
                             X_val, y_val, X_test, y_test, device, output_dir,
                             num_workers) -> Dict:
        """Run training method with optimized dataloaders."""
        batch_size = config['training']['batch_size']
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        if method == 'baseline':
            train_loader, val_loader, test_loader = \
                OptimizedDataLoaderFactory.create_standard_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, num_workers
                )

            trainer = BaselineTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=str(output_dir)
            )
            test_metrics = trainer.evaluate_full(test_loader)

        elif method == 'dp':
            train_loader, val_loader, test_loader = \
                OptimizedDataLoaderFactory.create_standard_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, num_workers
                )

            trainer = DPTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=str(output_dir)
            )
            test_metrics = trainer.evaluate_full(test_loader)

        elif method == 'fl':
            n_clients = config['federated_learning'].get('n_clients', 5)
            train_loaders, val_loaders = \
                OptimizedDataLoaderFactory.create_fl_dataloaders(
                    X_train, y_train, X_val, y_val,
                    n_clients, batch_size, num_workers
                )

            _, _, test_loader = \
                OptimizedDataLoaderFactory.create_standard_dataloaders(
                    X_test, y_test, X_test, y_test, X_test, y_test,
                    batch_size, num_workers
                )

            client_ids = [f"client_{i:02d}" for i in range(n_clients)]
            trainer = FLTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loaders, val_loaders, client_ids=client_ids,
                epochs=config['federated_learning'].get(
                    'global_rounds', 100
                ),
                patience=patience, output_dir=str(output_dir)
            )
            test_metrics = trainer.evaluate_full(test_loader)

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'training_results': training_results,
            'test_metrics': test_metrics
        }

    def run_all(self, experiments: Dict, device: str) -> List[Dict]:
        """Run all experiments with optimizations."""
        total = len(experiments)
        print(f"\nRunning {total} experiments with optimizations...\n")

        for idx, (exp_name, exp_config) in enumerate(
            experiments.items(), 1
        ):
            print(f"[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)

            if idx % 3 == 0:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()

        return self.results

    def save_results(self,
                    output_file: str = 'experiments/results_log.json'):
        """Save summary results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results: {summary['successful']}/{summary['total_experiments']} "
              f"successful")
        print(f"Saved to: {output_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run optimized experiments (Baseline, DP, FL, DP+FL)'
    )

    parser.add_argument('--scenario',
                       choices=['baseline', 'dp', 'fl', 'dp_fl', 'all'],
                       default='baseline')
    parser.add_argument('--tags', type=str,
                       help='Filter by tags (comma-separated)')
    parser.add_argument('--keywords', type=str,
                       help='Filter by keywords')
    parser.add_argument('--datasets', type=str,
                       help='Filter by datasets (comma-separated)')
    parser.add_argument('--methods', type=str,
                       help='Filter by methods (comma-separated)')
    parser.add_argument('--epsilon', type=float,
                       help='Filter by epsilon (DP)')
    parser.add_argument('--clients', type=int,
                       help='Filter by n_clients (FL)')
    parser.add_argument('--n_experiments', type=int, default=None,
                       help='Limit number of experiments')
    parser.add_argument('--output_file',
                       default='experiments/results_log.json')
    parser.add_argument('--scenarios_dir',
                       default='experiments/scenarios')
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--device', default=None,
                       help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--auto', action='store_true',
                       help='Skip confirmation')

    args = parser.parse_args()

    if args.device is None or args.device == 'auto':
        device = get_auto_device()
    else:
        device = args.device

    print(f"Device: {get_device_name(device)}")
    print(f"Optimizations: HDF5, optimized DataLoaders, caching\n")

    runner = OptimizedExperimentRunner(
        scenarios_dir=args.scenarios_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )

    scenarios = ['baseline', 'dp', 'fl', 'dp_fl'] \
        if args.scenario == 'all' else [args.scenario]
    all_experiments = {}

    for scenario in scenarios:
        try:
            scenario_data = runner.load_scenario(scenario)
            all_experiments.update(scenario_data['experiments'])
        except FileNotFoundError:
            continue

    if not all_experiments:
        print("No experiments found")
        return 1

    print(f"Loaded {len(all_experiments)} experiments\n")

    tags = args.tags.split(',') if args.tags else None
    datasets = args.datasets.split(',') if args.datasets else None
    methods = args.methods.split(',') if args.methods else None

    filtered = runner.filter_experiments(
        all_experiments,
        tags=tags,
        keywords=args.keywords,
        datasets=datasets,
        methods=methods,
        epsilon=args.epsilon,
        clients=args.clients
    )

    if args.n_experiments:
        filtered = dict(list(filtered.items())[:args.n_experiments])

    if not filtered:
        print("No experiments matched")
        return 1

    print(f"{'='*70}")
    print(f"Will run: {len(filtered)} experiments (OPTIMIZED)")
    print(f"{'='*70}\n")

    if not args.auto:
        if input("Proceed? (y/n): ").lower() != 'y':
            return 0

    start_time = time.time()
    runner.run_all(filtered, device)
    elapsed = time.time() - start_time
    runner.save_results(args.output_file)

    print(f"Total time: {elapsed/3600:.2f} hours (optimized!)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())