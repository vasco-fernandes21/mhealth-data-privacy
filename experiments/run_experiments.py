#!/usr/bin/env python3
"""
Optimized experiment runner for Baseline, DP, FL, DP+FL training.
Works with pre-computed windows for fast loading.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import get_logger
from src.models import UnifiedLSTMModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf, \
    preprocess_sleep_edf
from src.preprocessing.wesad import load_processed_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader


def get_auto_device() -> str:
    """
    Detect device automatically with hierarchy: cuda → mps → cpu.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and \
         torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical summary for a list of values."""
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'values': []
        }
    
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'values': [float(v) for v in values]
    }


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


class ExperimentRunner:
    """Optimized experiment runner."""
    
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
    
    # ===== CONFIG MANAGEMENT =====
    
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
    
    # ===== SCENARIO LOADING =====
    
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
    
    # ===== DATA LOADING =====
    
    def _load_data(self, dataset: str) -> Tuple:
        """Load dataset (auto-preprocess if needed)."""
        data_path = self.data_dir / dataset

        if dataset == 'sleep-edf':
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
            
            return load_windowed_sleep_edf(str(data_path))
        
        elif dataset == 'wesad':
            return load_processed_wesad_temporal(str(data_path))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # ===== DATALOADER CREATION =====
    
    def _create_dataloaders(self, X_train, y_train, X_val, y_val,
                           X_test, y_test, batch_size: int,
                           is_dp: bool = False,
                           num_workers: int = 0) -> Tuple:
        """Create dataloaders from numpy arrays."""
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=is_dp,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"DataLoaders created: num_workers={num_workers}, "
              f"train_batches={len(train_loader)}, "
              f"val_batches={len(val_loader)}, "
              f"test_batches={len(test_loader)}\n")

        return train_loader, val_loader, test_loader
    
    def _create_fl_dataloaders(self, X_train, y_train, X_val, y_val,
                              n_clients: int, batch_size: int,
                              is_dp: bool = False,
                              num_workers: int = 0) -> Tuple[List, List]:
        """Create FL dataloaders (partitioned by client)."""
        n_samples = len(X_train)
        samples_per_client = n_samples // n_clients
        
        train_loaders = []
        val_loaders = []
        
        for client_id in range(n_clients):
            start_idx = client_id * samples_per_client
            end_idx = (start_idx + samples_per_client
                      if client_id < n_clients - 1 else n_samples)
            
            X_client_train = X_train[start_idx:end_idx]
            y_client_train = y_train[start_idx:end_idx]
            
            val_start = client_id * (len(X_val) // n_clients)
            val_end = val_start + (len(X_val) // n_clients)
            
            X_client_val = X_val[val_start:val_end]
            y_client_val = y_val[val_start:val_end]
            
            train_ds = TensorDataset(
                torch.tensor(X_client_train, dtype=torch.float32),
                torch.tensor(y_client_train, dtype=torch.long)
            )
            val_ds = TensorDataset(
                torch.tensor(X_client_val, dtype=torch.float32),
                torch.tensor(y_client_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=is_dp,
                num_workers=num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
        
        print(f"FL DataLoaders created: {n_clients} clients, "
              f"num_workers={num_workers}, "
              f"avg_train_batches={sum(len(l) for l in train_loaders) // n_clients}, "
              f"avg_val_batches={sum(len(l) for l in val_loaders) // n_clients}\n")
        
        return train_loaders, val_loaders
    
    # ===== EXPERIMENT EXECUTION =====
    
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
        """Execute single experiment."""
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
            
            num_workers = config['training'].get('num_workers', 0)
            print(f"Config: batch_size={config['training'].get('batch_size')}, "
                  f"lr={config['training'].get('learning_rate')}, "
                  f"epochs={config['training'].get('epochs')}, "
                  f"num_workers={num_workers}\n")

            if method == 'dp':
                dp_cfg = config.get('differential_privacy', {})
                print(f"DP Config: σ={dp_cfg.get('noise_multiplier', 'N/A')}, "
                      f"ε_target={dp_cfg.get('target_epsilon', 'N/A')}\n")
            
            print("Loading data...")
            data = self._load_data(dataset)
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, \
                info, subjects = data
            
            print(f"Shapes: train={X_train.shape}, val={X_val.shape}, "
                  f"test={X_test.shape}\n")
            
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
            
            result = self._run_method(
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
            elif method == 'dp_fl':
                detailed_results['privacy_metrics'] = {
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
                    ).get('max_grad_norm', 0)
                }
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
    
    def _run_method(self, method, model, config, X_train, y_train,
                   X_val, y_val, X_test, y_test, device, output_dir,
                   num_workers) -> Dict:
        """Run training method."""
        batch_size = config['training']['batch_size']
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        if method == 'baseline':
            train_loader, val_loader, test_loader = \
                self._create_dataloaders(X_train, y_train, X_val, y_val,
                                        X_test, y_test, batch_size,
                                        num_workers=num_workers)

            trainer = BaselineTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=str(output_dir)
            )
            test_metrics = trainer.evaluate_full(test_loader)

        elif method == 'dp':
            train_loader, val_loader, test_loader = \
                self._create_dataloaders(X_train, y_train, X_val, y_val,
                                        X_test, y_test, batch_size,
                                        is_dp=True,
                                        num_workers=num_workers)

            trainer = DPTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=str(output_dir)
            )
            test_metrics = trainer.evaluate_full(test_loader)
        
        elif method == 'fl':
            n_clients = config['federated_learning'].get('n_clients', 5)
            train_loaders, val_loaders = \
                self._create_fl_dataloaders(X_train, y_train, X_val, y_val,
                                           n_clients, batch_size,
                                           num_workers=num_workers)

            test_ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            print(f"Test DataLoader created: num_workers={num_workers}, "
                  f"test_batches={len(test_loader)}\n")
            
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
        
        elif method == 'dp_fl':
            n_clients = config['federated_learning'].get('n_clients', 5)
            train_loaders, val_loaders = \
                self._create_fl_dataloaders(X_train, y_train, X_val, y_val,
                                           n_clients, batch_size,
                                           is_dp=True,
                                           num_workers=num_workers)

            test_ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            print(f"Test DataLoader created: num_workers={num_workers}, "
                  f"test_batches={len(test_loader)}\n")
            
            client_ids = [f"client_{i:02d}" for i in range(n_clients)]
            from src.training.trainers.fl_dp_trainer import FLDPTrainer
            trainer = FLDPTrainer(model, config, device=device)
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
        """Run all experiments."""
        total = len(experiments)
        print(f"\nRunning {total} experiments...\n")
        
        for idx, (exp_name, exp_config) in enumerate(
            experiments.items(), 1
        ):
            print(f"[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)
        
        return self.results
    
    def _load_experiment_results(
        self, results_file: str
    ) -> Optional[Dict]:
        """Load results from experiment file."""
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(
                f"Could not load {results_file}: {e}"
            )
            return None
    
    def _aggregate_baseline_results(self, dataset: str) -> Dict:
        """Aggregate baseline results by dataset."""
        dataset_dir = self.results_dir / 'baseline' / dataset
        if not dataset_dir.exists():
            return {}
        
        results = []
        for seed_dir in dataset_dir.iterdir():
            if seed_dir.is_dir():
                results_file = seed_dir / 'results.json'
                if results_file.exists():
                    result = self._load_experiment_results(
                        str(results_file)
                    )
                    if result:
                        results.append(result)
        
        if not results:
            return {}
        
        aggregated = {
            'method': 'baseline',
            'dataset': dataset,
            'n_experiments': len(results),
            'aggregated_metrics': {
                'accuracy': compute_statistics([
                    r['test_metrics']['accuracy'] for r in results
                ]),
                'f1_score': compute_statistics([
                    r['test_metrics']['f1_score'] for r in results
                ]),
                'precision': compute_statistics([
                    r['test_metrics']['precision'] for r in results
                ]),
                'recall': compute_statistics([
                    r['test_metrics']['recall'] for r in results
                ]),
                'best_val_acc': compute_statistics([
                    r['training_metrics']['best_val_acc'] for r in results
                ]),
                'training_time_seconds': compute_statistics([
                    r['training_metrics']['training_time_seconds']
                    for r in results
                ]),
                'total_epochs': compute_statistics([
                    r['training_metrics']['total_epochs'] for r in results
                ])
            }
        }
        
        return aggregated
    
    def _aggregate_dp_results(self, dataset: str) -> Dict:
        """Aggregate DP results by noise level."""
        dataset_dir = self.results_dir / 'dp' / dataset
        if not dataset_dir.exists():
            return {}
        
        noise_groups = defaultdict(list)
        
        for config_dir in dataset_dir.iterdir():
            if config_dir.is_dir():
                results_file = config_dir / 'results.json'
                if results_file.exists():
                    result = self._load_experiment_results(
                        str(results_file)
                    )
                    if result:
                        noise_mult = result.get(
                            'privacy_metrics', {}
                        ).get('noise_multiplier', 0)
                        noise_key = f'{noise_mult:.1f}'.replace('.', '_')
                        noise_groups[noise_key].append(result)
        
        if not noise_groups:
            return {}
        
        aggregated = {
            'method': 'dp',
            'dataset': dataset,
            'noise_levels': {}
        }
        
        for noise_key, results in noise_groups.items():
            aggregated['noise_levels'][noise_key] = {
                'n_experiments': len(results),
                'privacy': {
                    'epsilon': compute_statistics([
                        r['privacy_metrics']['final_epsilon']
                        for r in results
                    ]),
                    'noise_multiplier': results[0][
                        'privacy_metrics'
                    ]['noise_multiplier']
                },
                'utility': {
                    'accuracy': compute_statistics([
                        r['test_metrics']['accuracy'] for r in results
                    ]),
                    'f1_score': compute_statistics([
                        r['test_metrics']['f1_score'] for r in results
                    ]),
                    'precision': compute_statistics([
                        r['test_metrics']['precision'] for r in results
                    ]),
                    'recall': compute_statistics([
                        r['test_metrics']['recall'] for r in results
                    ])
                }
            }
        
        return aggregated
    
    def _aggregate_fl_results(self, dataset: str) -> Dict:
        """Aggregate FL results by n_clients."""
        dataset_dir = self.results_dir / 'fl' / dataset
        if not dataset_dir.exists():
            return {}
        
        client_groups = defaultdict(list)
        
        for config_dir in dataset_dir.iterdir():
            if config_dir.is_dir():
                results_file = config_dir / 'results.json'
                if results_file.exists():
                    result = self._load_experiment_results(
                        str(results_file)
                    )
                    if result:
                        n_clients = result.get(
                            'federated_metrics', {}
                        ).get('n_clients', 0)
                        client_key = str(n_clients)
                        client_groups[client_key].append(result)
        
        if not client_groups:
            return {}
        
        aggregated = {
            'method': 'fl',
            'dataset': dataset,
            'client_configurations': {}
        }
        
        for client_key, results in client_groups.items():
            aggregated['client_configurations'][client_key] = {
                'n_experiments': len(results),
                'utility': {
                    'accuracy': compute_statistics([r['test_metrics']['accuracy'] for r in results]),
                    'f1_score': compute_statistics([r['test_metrics']['f1_score'] for r in results]),
                    'precision': compute_statistics([r['test_metrics']['precision'] for r in results]),
                    'recall': compute_statistics([r['test_metrics']['recall'] for r in results])
                },
                'efficiency': {
                    'training_time_seconds': compute_statistics([r['training_metrics']['training_time_seconds'] for r in results]),
                    'total_rounds': compute_statistics([r['federated_metrics']['total_rounds'] for r in results]),
                    'n_clients': results[0]['federated_metrics']['n_clients']
                },
                'training': {
                    'best_val_acc': compute_statistics([r['training_metrics']['best_val_acc'] for r in results])
                }
            }
        
        # Scalability analysis
        sorted_clients = sorted(client_groups.keys(), key=int)
        aggregated['scalability_analysis'] = {
            'n_clients': [int(k) for k in sorted_clients],
            'accuracy': [aggregated['client_configurations'][k]['utility']['accuracy']['mean'] for k in sorted_clients],
            'f1_score': [aggregated['client_configurations'][k]['utility']['f1_score']['mean'] for k in sorted_clients],
            'training_time': [aggregated['client_configurations'][k]['efficiency']['training_time_seconds']['mean'] for k in sorted_clients]
        }
        
        return aggregated
    
    def _aggregate_dp_fl_results(self, dataset: str) -> Dict:
        """Aggregate DP+FL results."""
        dataset_dir = self.results_dir / 'dp_fl' / dataset
        if not dataset_dir.exists():
            return {}
        
        results = []
        for config_dir in dataset_dir.iterdir():
            if config_dir.is_dir():
                results_file = config_dir / 'results.json'
                if results_file.exists():
                    result = self._load_experiment_results(
                        str(results_file)
                    )
                    if result:
                        results.append(result)
        
        if not results:
            return {}
        
        aggregated = {
            'method': 'dp_fl',
            'dataset': dataset,
            'n_experiments': len(results),
            'aggregated_metrics': {
                'accuracy': compute_statistics([r['test_metrics']['accuracy'] for r in results]),
                'f1_score': compute_statistics([r['test_metrics']['f1_score'] for r in results]),
                'precision': compute_statistics([r['test_metrics']['precision'] for r in results]),
                'recall': compute_statistics([r['test_metrics']['recall'] for r in results])
            },
            'privacy_metrics': {
                'noise_multiplier': compute_statistics([r['privacy_metrics']['noise_multiplier'] for r in results if 'privacy_metrics' in r])
            },
            'federated_metrics': {
                'n_clients': compute_statistics([r['federated_metrics']['n_clients'] for r in results if 'federated_metrics' in r]),
                'total_rounds': compute_statistics([r['federated_metrics']['total_rounds'] for r in results if 'federated_metrics' in r])
            }
        }
        
        return aggregated
    
    def create_summaries(self):
        """Create aggregated summary files."""
        methods = ['baseline', 'dp', 'fl', 'dp_fl']
        datasets = set()
        
        for method in methods:
            method_dir = self.results_dir / method
            if method_dir.exists():
                for dataset_dir in method_dir.iterdir():
                    if dataset_dir.is_dir():
                        datasets.add(dataset_dir.name)
        
        for dataset in datasets:
            for method in methods:
                if method == 'baseline':
                    summary = self._aggregate_baseline_results(dataset)
                elif method == 'dp':
                    summary = self._aggregate_dp_results(dataset)
                elif method == 'fl':
                    summary = self._aggregate_fl_results(dataset)
                elif method == 'dp_fl':
                    summary = self._aggregate_dp_fl_results(dataset)
                else:
                    continue
                
                if summary:
                    summary_file = self.results_dir / method / \
                        f'{dataset}_summary.json'
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    self.logger.info(f"Created summary: {summary_file}")
    
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
        
        print(f"\nCreating aggregated summaries...")
        self.create_summaries()
        
        print(f"\n{'='*70}")
        print(f"Results: {summary['successful']}/{summary['total_experiments']} successful")
        print(f"Saved to: {output_path}")
        print(f"Summaries created in: {self.results_dir}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments (Baseline, DP, FL, DP+FL)'
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
    
    print(f"Device: {get_device_name(device)}\n")
    
    runner = ExperimentRunner(
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
    print(f"Will run: {len(filtered)} experiments")
    print(f"{'='*70}\n")
    
    if not args.auto:
        if input("Proceed? (y/n): ").lower() != 'y':
            return 0
    
    start_time = time.time()
    runner.run_all(filtered, device)
    elapsed = time.time() - start_time
    runner.save_results(args.output_file)
    
    print(f"Total time: {elapsed/3600:.2f} hours\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())