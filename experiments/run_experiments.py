#!/usr/bin/env python3
"""
Simplified experiment runner 
"""

import sys
import yaml
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time
from copy import deepcopy
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import get_logger
from src.models import UnifiedLSTMModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer

from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_processed_wesad_temporal


def get_device() -> str:
    """Simple device detection."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def create_simple_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                             batch_size=128):
    """Create simple, fast dataloaders."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Simple conversion
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64))
    )
    
    # Simple dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_fl_dataloaders(X_train, y_train, X_val, y_val, n_clients, 
                         batch_size=128):
    """Create FL dataloaders (simple partition)."""
    from torch.utils.data import TensorDataset, DataLoader
    
    train_loaders = []
    val_loaders = []
    
    # Simple split by samples
    n_train = len(X_train)
    n_val = len(X_val)
    train_per_client = n_train // n_clients
    val_per_client = n_val // n_clients
    
    for i in range(n_clients):
        # Train split
        start = i * train_per_client
        end = start + train_per_client if i < n_clients - 1 else n_train
        X_client = X_train[start:end]
        y_client = y_train[start:end]
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_client.astype(np.float32)),
            torch.from_numpy(y_client.astype(np.int64))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
        
        # Val split
        val_start = i * val_per_client
        val_end = val_start + val_per_client if i < n_clients - 1 else n_val
        X_val_client = X_val[val_start:val_end]
        y_val_client = y_val[val_start:val_end]
        
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_client.astype(np.float32)),
            torch.from_numpy(y_val_client.astype(np.int64))
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders


class SimpleExperimentRunner:
    """Simple, fast experiment runner."""
    
    def __init__(self, scenarios_dir='experiments/scenarios',
                 data_dir='./data/processed',
                 config_dir='./src/configs',
                 results_dir='./results'):
        self.scenarios_dir = Path(scenarios_dir)
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results = []
        self._data_cache = {}

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _merge_configs(self, base: dict, override: dict) -> dict:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _get_config(self, dataset: str, method: str, exp_hyperparams=None):
        dataset_cfg = self._load_config(self.config_dir / 'datasets' / f'{dataset}.yaml')
        method_cfg = self._load_config(self.config_dir / 'methods' / f'{method}.yaml')
        
        config = self._merge_configs(dataset_cfg, method_cfg)
        
        if exp_hyperparams:
            for section, params in exp_hyperparams.items():
                config[section] = self._merge_configs(config.get(section, {}), params)
        
        return config

    def load_scenario(self, scenario_name: str):
        scenario_file = self.scenarios_dir / f'{scenario_name}.yaml'
        with open(scenario_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_data(self, dataset: str):
        if dataset in self._data_cache:
            print(f"✅ {dataset} (cached)", flush=True)
            return self._data_cache[dataset]
        
        print(f"📥 Loading {dataset}...", flush=True)
        sys.stdout.flush()
        
        data_path = self.data_dir / dataset
        
        if dataset == 'sleep-edf':
            print(f"   Reading HDF5...", flush=True)
            sys.stdout.flush()
            data = load_windowed_sleep_edf(str(data_path))
        elif dataset == 'wesad':
            print(f"   Reading NPZ...", flush=True)
            sys.stdout.flush()
            data = load_processed_wesad_temporal(str(data_path))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self._data_cache[dataset] = data
        print(f"✅ {dataset} loaded", flush=True)
        sys.stdout.flush()
        return data

    def run_experiment(self, exp_name: str, exp_config: Dict, device: str):
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        
        print(f"\n{'='*50}")
        print(f"{exp_name} | {dataset} | {method} | seed={seed}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            set_reproducible(seed=seed, device=device, verbose=False)
            
            config = self._get_config(dataset, method, exp_config.get('hyperparameters'))
            
            # Load data
            data = self._load_data(dataset)
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects = data
            
            print(f"Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            
            # Create model
            config['dataset']['input_dim'] = X_train.shape[1]
            model = UnifiedLSTMModel(config, device=device)
            
            # Output directory
            output_dir = self.results_dir / method / dataset / f'seed_{seed}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run training
            batch_size = config['training']['batch_size']
            epochs = config['training']['epochs']
            patience = config['training'].get('early_stopping_patience', 10)
            
            if method == 'baseline':
                train_loader, val_loader, test_loader = create_simple_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size
                )
                trainer = BaselineTrainer(model, config, device=device)
                training_results = trainer.fit(train_loader, val_loader, 
                                             epochs=epochs, patience=patience,
                                             output_dir=str(output_dir))
                test_metrics = trainer.evaluate_full(test_loader)
                
            elif method == 'dp':
                train_loader, val_loader, test_loader = create_simple_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size
                )
                trainer = DPTrainer(model, config, device=device)
                training_results = trainer.fit(train_loader, val_loader,
                                             epochs=epochs, patience=patience,
                                             output_dir=str(output_dir))
                test_metrics = trainer.evaluate_full(test_loader)
                
            elif method == 'fl':
                n_clients = config['federated_learning'].get('n_clients', 5)
                train_loaders, val_loaders = create_fl_dataloaders(
                    X_train, y_train, X_val, y_val, n_clients, batch_size
                )
                _, _, test_loader = create_simple_dataloaders(
                    X_test, y_test, X_test, y_test, X_test, y_test, batch_size
                )
                
                client_ids = [f"client_{i:02d}" for i in range(n_clients)]
                trainer = FLTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loaders, val_loaders, client_ids=client_ids,
                    epochs=config['federated_learning'].get('global_rounds', 100),
                    patience=patience, output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed = time.time() - start_time
            
            # Save results
            results_data = {
                'experiment_info': {
                    'name': exp_name,
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat()
                },
                'test_metrics': test_metrics,
                'training_results': training_results
            }
            
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"✅ Completed in {elapsed:.1f}s")
            print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"   F1-Score: {test_metrics.get('f1_score', 0):.4f}")
            
            result = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': True,
                'time_seconds': elapsed,
                'accuracy': test_metrics.get('accuracy', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: {e}")
            result = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': False,
                'error': str(e),
                'time_seconds': elapsed,
                'timestamp': datetime.now().isoformat()
            }
        
        self.results.append(result)
        return result

    def run_all(self, experiments: Dict, device: str):
        total = len(experiments)
        print(f"\nRunning {total} experiments...")
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"\n[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)
            
            # Cleanup every few experiments
            if idx % 3 == 0:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        return self.results

    def save_results(self, output_file='experiments/results_log.json'):
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'results': self.results
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults: {summary['successful']}/{summary['total_experiments']} successful")
        print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Simple experiment runner')
    parser.add_argument('--scenario', choices=['baseline', 'dp', 'fl', 'dp_fl'], default='baseline')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--datasets', help='Filter datasets (comma-separated)')
    parser.add_argument('--n_experiments', type=int, help='Limit experiments')
    parser.add_argument('--auto', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else args.device
    print(f"Device: {device}")
    
    runner = SimpleExperimentRunner()
    
    # Load scenario
    scenario_data = runner.load_scenario(args.scenario)
    experiments = scenario_data['experiments']
    
    # Filter datasets if specified
    if args.datasets:
        datasets = args.datasets.split(',')
        experiments = {
            name: config for name, config in experiments.items()
            if config.get('dataset') in datasets
        }
    
    # Limit experiments if specified
    if args.n_experiments:
        experiments = dict(list(experiments.items())[:args.n_experiments])
    
    print(f"Will run: {len(experiments)} experiments")
    
    if not args.auto:
        if input("Proceed? (y/n): ").lower() != 'y':
            return 0
    
    start_time = time.time()
    runner.run_all(experiments, device)
    elapsed = time.time() - start_time
    runner.save_results()
    
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())