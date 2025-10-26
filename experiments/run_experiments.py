#!/usr/bin/env python3
"""
Master script to run experiments from YAML configuration files.
Supports: Baseline, DP, FL, DP+FL training.

Usage:
    python experiments/run_experiments.py --scenario baseline
    python experiments/run_experiments.py --scenario dp --tags tier1
    python experiments/run_experiments.py --scenario fl --datasets wesad
    python experiments/run_experiments.py --scenario all
"""

import sys
import yaml
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time
from copy import deepcopy
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import setup_logging, get_logger
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer
from src.training.trainers.fl_dp_trainer import FLDPTrainer
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import (
    load_processed_wesad_temporal,
    apply_normalization,
    compute_normalization_stats
)
from torch.utils.data import TensorDataset, DataLoader


class ExperimentRunner:
    """Run experiments with support for Baseline, DP, FL, DP+FL."""
    
    def __init__(self, scenarios_dir: str = 'experiments/scenarios', 
                 data_dir: str = './data/processed',
                 config_dir: str = './src/configs',
                 results_dir: str = './results'):
        self.scenarios_dir = Path(scenarios_dir)
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results = []
        self.logger = get_logger(__name__)
        
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load experiment scenario from YAML."""
        scenario_file = self.scenarios_dir / f'{scenario_name}.yaml'
        
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")
        
        with open(scenario_file, 'r') as f:
            return yaml.safe_load(f)
    
    def filter_experiments(self, experiments: Dict, 
                          tags: List[str] = None,
                          keywords: str = None,
                          epsilon: float = None,
                          clients: int = None,
                          datasets: List[str] = None,
                          methods: List[str] = None) -> Dict:
        """Filter experiments based on criteria."""
        filtered = {}
        
        for exp_name, exp_config in experiments.items():
            if not exp_config.get('enabled', True):
                continue
            
            if tags:
                exp_tags = exp_config.get('tags', [])
                if not any(tag in exp_tags for tag in tags):
                    continue
            
            if keywords and keywords.lower() not in exp_name.lower():
                continue
            
            if datasets and exp_config.get('dataset') not in datasets:
                continue
            
            if methods and exp_config.get('method') not in methods:
                continue
            
            if epsilon is not None:
                hp = exp_config.get('hyperparameters', {})
                dp_cfg = hp.get('differential_privacy', {})
                if dp_cfg.get('target_epsilon') != epsilon:
                    continue
            
            if clients is not None:
                hp = exp_config.get('hyperparameters', {})
                fl_cfg = hp.get('federated_learning', {})
                if fl_cfg.get('n_clients') != clients:
                    continue
            
            filtered[exp_name] = exp_config
        
        return filtered
    
    # =====================================================================
    # DATA LOADING
    # =====================================================================
    
    def _load_data(self, dataset: str) -> Tuple:
        """Load dataset."""
        data_path = self.data_dir / dataset
        
        if dataset == 'sleep-edf':
            (X_train, X_val, X_test, y_train, y_val, y_test,
             scaler, info, subjects_train) = load_windowed_sleep_edf(
                 str(data_path))
        elif dataset == 'wesad':
            (X_train, X_val, X_test, y_train, y_val, y_test,
             label_encoder, info) = load_processed_wesad_temporal(
                 str(data_path))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, info
    
    # DATALOADER CREATION
    # =====================================================================
    # =====================================================================
    
    def _create_standard_dataloaders(self, X_train, y_train, X_val, y_val,
                                     X_test, y_test, batch_size: int) -> Tuple:
        """Create standard dataloaders (Baseline, DP)."""
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader
    
    def _create_dp_dataloaders(self, X_train, y_train, X_val, y_val,
                               X_test, y_test, batch_size: int) -> Tuple:
        """Create DP-compatible dataloaders (fixed batch size)."""
        
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
        
        # DP requires: fixed batch size, drop_last=True
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_fl_dataloaders(self, X_train, y_train, X_val, y_val,
                               n_clients: int, batch_size: int,
                               is_dp: bool = False) -> Tuple[List, List]:
        """Create FL dataloaders (partitioned by client).
        
        Returns:
            (train_loaders, val_loaders)
        """
        n_samples = len(X_train)
        samples_per_client = n_samples // n_clients
        
        train_loaders = []
        val_loaders = []
        
        for client_id in range(n_clients):
            # Partition training data
            start_idx = client_id * samples_per_client
            end_idx = (
                start_idx + samples_per_client
                if client_id < n_clients - 1
                else n_samples
            )
            
            X_client_train = X_train[start_idx:end_idx]
            y_client_train = y_train[start_idx:end_idx]
            
            # Partition validation data
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
            
            # DP requires drop_last=True
            drop_last = is_dp
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
        
        return train_loaders, val_loaders
    
    # =====================================================================
    # MODEL CREATION
    # =====================================================================
    
    def _get_model(self, dataset: str, config: dict, device: str):
        """Create model based on dataset."""
        if dataset == 'sleep-edf':
            return SleepEDFModel(config, device=device)
        elif dataset == 'wesad':
            return WESADModel(config, device=device)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # =====================================================================
    # CONFIG MANAGEMENT
    # =====================================================================
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge override into base dictionary."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and \
               isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _apply_experiment_hyperparameters(self, config: dict,
                                         exp_config: dict) -> dict:
        """Apply experiment-specific hyperparameters."""
        if 'hyperparameters' not in exp_config:
            return config
        
        hp = exp_config['hyperparameters']
        config = deepcopy(config)
        
        # Model hyperparameters
        if 'model' in hp:
            config['model'] = self._deep_merge(
                config.get('model', {}), hp['model']
            )
        
        # Training hyperparameters
        if 'training' in hp:
            config['training'] = self._deep_merge(
                config.get('training', {}), hp['training']
            )
        
        # Top-level training parameters
        training_params = ['epochs', 'batch_size', 'learning_rate',
                          'early_stopping_patience', 'optimizer',
                          'weight_decay']
        for param in training_params:
            if param in hp:
                if 'training' not in config:
                    config['training'] = {}
                config['training'][param] = hp[param]
        
        # FL hyperparameters
        if 'federated_learning' in hp:
            config['federated_learning'] = self._deep_merge(
                config.get('federated_learning', {}),
                hp['federated_learning']
            )
        
        # DP hyperparameters
        if 'differential_privacy' in hp:
            config['differential_privacy'] = self._deep_merge(
                config.get('differential_privacy', {}),
                hp['differential_privacy']
            )
        
        return config
    
    # =====================================================================
    # EXPERIMENT EXECUTION
    # =====================================================================
    
    def run_experiment(self, exp_name: str, exp_config: Dict,
                      device: str) -> Dict:
        """Execute single experiment (dispatches to correct trainer)."""
        
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        
        print(f"\n{'='*70}")
        print(f"{exp_name}")
        print(f"   Dataset: {dataset}, Method: {method}, Seed: {seed}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success = False
        final_results = None
        
        try:
            set_reproducible(seed=seed, device=device, verbose=False)
            
            # Load configs
            default_cfg = self._load_config(
                self.config_dir / 'training_defaults.yaml')
            
            config_mapping = {
                'sleep-edf': 'sleep_edf.yaml',
                'wesad': 'wesad.yaml'
            }
            config_filename = config_mapping.get(
                dataset, f'{dataset}.yaml'
            )
            dataset_cfg = self._load_config(
                self.config_dir / config_filename
            )
            
            # Merge configs
            config = self._deep_merge(default_cfg, dataset_cfg)
            config = self._apply_experiment_hyperparameters(
                config, exp_config
            )
            
            # Log configuration
            print("Applied configuration:")
            print(f"   Batch size: {config['training'].get('batch_size', 'N/A')}")
            print(f"   Learning rate: {config['training'].get('learning_rate', 'N/A')}")
            print(f"   Epochs: {config['training'].get('epochs', 'N/A')}\n")
            
            # Load data
            print("Loading data...")
            X_train, X_val, X_test, y_train, y_val, y_test, info = \
                self._load_data(dataset)
            print(f"Data loaded: train={X_train.shape}, val={X_val.shape}, "
                  f"test={X_test.shape}")
            
            # Normalize
            norm_stats = compute_normalization_stats(X_train)
            X_train = apply_normalization(X_train, norm_stats)
            X_val = apply_normalization(X_val, norm_stats)
            X_test = apply_normalization(X_test, norm_stats)
            
            # Create model
            print("Creating model...")
            model = self._get_model(dataset, config, device)
            
            # ===================================================================
            # ROUTE BY METHOD
            # ===================================================================
            
            if method == 'baseline':
                result = self._run_baseline(
                    model, config, X_train, y_train, X_val, y_val,
                    X_test, y_test, device
                )
            
            elif method == 'dp':
                result = self._run_dp(
                    model, config, X_train, y_train, X_val, y_val,
                    X_test, y_test, device
                )
            
            elif method == 'fl':
                result = self._run_fl(
                    model, config, X_train, y_train, X_val, y_val,
                    X_test, y_test, device
                )
            
            elif method == 'dp_fl':
                result = self._run_dp_fl(
                    model, config, X_train, y_train, X_val, y_val,
                    X_test, y_test, device
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # ===================================================================
            # SAVE RESULTS
            # ===================================================================
            
            training_results = result['training_results']
            test_metrics = result['test_metrics']
            
            elapsed = time.time() - start_time
            success = True
            
            # Detailed results
            detailed_results = {
                'seed': seed,
                'method': method,
                'dataset': dataset,
                'total_epochs': training_results.get('total_epochs', 0),
                'training_time_seconds': training_results.get(
                    'training_time_seconds', elapsed),
                'best_val_acc': training_results.get('best_val_acc', 0),
                'final_train_loss': training_results.get(
                    'final_train_loss', 0),
                'final_train_acc': training_results.get(
                    'final_train_acc', 0),
                'final_val_loss': training_results.get(
                    'final_val_loss', 0),
                'final_val_acc': training_results.get(
                    'final_val_acc', 0),
                'accuracy': test_metrics.get('accuracy', 0),
                'precision': test_metrics.get('precision', 0),
                'recall': test_metrics.get('recall', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'confusion_matrix': test_metrics.get(
                    'confusion_matrix', []),
                'class_names': test_metrics.get('class_names', []),
                'config_used': {
                    'batch_size': config['training'].get('batch_size'),
                    'learning_rate': config['training'].get(
                        'learning_rate'),
                    'epochs': config['training'].get('epochs'),
                }
            }
            
            # Add method-specific results
            if method == 'dp':
                detailed_results['final_epsilon'] = training_results.get(
                    'final_epsilon', 0)
                detailed_results['target_epsilon'] = training_results.get(
                    'target_epsilon', 0)
            
            if method in ['fl', 'dp_fl']:
                detailed_results['n_clients'] = training_results.get(
                    'n_clients', 0)
            
            # Save to file
            output_dir = (self.results_dir / method / dataset /
                         f'seed_{seed}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
            
            # Summary result
            final_results = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': success,
                'time_seconds': elapsed,
                'accuracy': test_metrics.get('accuracy', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'results_file': str(results_file),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Completed in {elapsed:.1f}s")
            print(f"   Accuracy: {final_results['accuracy']:.4f}")
            print(f"   F1-Score: {final_results['f1_score']:.4f}")
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Failed: {e}")
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
    
    def _run_baseline(self, model, config, X_train, y_train, X_val, y_val,
                      X_test, y_test, device) -> Dict:
        """Run Baseline experiment."""
        
        batch_size = config['training']['batch_size']
        train_loader, val_loader, test_loader = \
            self._create_standard_dataloaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size
            )
        
        # Train
        trainer = BaselineTrainer(model, config, device=device)
        training_results = trainer.fit(
            train_loader, val_loader,
            epochs=config['training']['epochs'],
            patience=config['training'].get(
                'early_stopping_patience', 10),
            output_dir=None
        )
        
        # Evaluate
        test_metrics = trainer.evaluate_full(test_loader)
        
        return {
            'training_results': training_results,
            'test_metrics': test_metrics
        }
    
    def _run_dp(self, model, config, X_train, y_train, X_val, y_val,
                X_test, y_test, device) -> Dict:
        """Run DP experiment."""
        
        batch_size = config['training']['batch_size']
        train_loader, val_loader, test_loader = \
            self._create_dp_dataloaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size
            )
        
        # Train
        trainer = DPTrainer(model, config, device=device)
        training_results = trainer.fit(
            train_loader, val_loader,
            epochs=config['training']['epochs'],
            patience=config['training'].get(
                'early_stopping_patience', 10),
            output_dir=None
        )
        
        # Evaluate
        test_metrics = trainer.evaluate_full(test_loader)
        
        return {
            'training_results': training_results,
            'test_metrics': test_metrics
        }
    
    def _run_fl(self, model, config, X_train, y_train, X_val, y_val,
                X_test, y_test, device) -> Dict:
        """Run FL experiment."""
        
        # FL config
        n_clients = config['federated_learning'].get('n_clients', 5)
        batch_size = config['training']['batch_size']
        
        # Create FL dataloaders
        train_loaders, val_loaders = self._create_fl_dataloaders(
            X_train, y_train, X_val, y_val,
            n_clients=n_clients,
            batch_size=batch_size,
            is_dp=False
        )
        
        # Create test loader
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False)
        
        # Client IDs
        client_ids = [f"client_{i:02d}" for i in range(n_clients)]
        
        # Train
        trainer = FLTrainer(model, config, device=device)
        training_results = trainer.fit(
            train_loaders, val_loaders,
            client_ids=client_ids,
            epochs=config['federated_learning'].get('global_rounds', 100),
            patience=config['training'].get(
                'early_stopping_patience', 10),
            output_dir=None
        )
        
        # Evaluate
        test_metrics = trainer.evaluate_full(test_loader)
        
        return {
            'training_results': training_results,
            'test_metrics': test_metrics
        }
    
    def _run_dp_fl(self, model, config, X_train, y_train, X_val, y_val,
                   X_test, y_test, device) -> Dict:
        """Run DP+FL experiment."""
        
        # FL config
        n_clients = config['federated_learning'].get('n_clients', 5)
        batch_size = config['training']['batch_size']
        
        # Create FL+DP dataloaders
        train_loaders, val_loaders = self._create_fl_dataloaders(
            X_train, y_train, X_val, y_val,
            n_clients=n_clients,
            batch_size=batch_size,
            is_dp=True  # Use drop_last for DP
        )
        
        # Create test loader
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False)
        
        # Client IDs
        client_ids = [f"client_{i:02d}" for i in range(n_clients)]
        
        # Train
        trainer = FLDPTrainer(model, config, device=device)
        training_results = trainer.fit(
            train_loaders, val_loaders,
            client_ids=client_ids,
            epochs=config['federated_learning'].get('global_rounds', 100),
            patience=config['training'].get(
                'early_stopping_patience', 10),
            output_dir=None
        )
        
        # Evaluate
        test_metrics = trainer.evaluate_full(test_loader)
        
        return {
            'training_results': training_results,
            'test_metrics': test_metrics
        }
    
    def run_all(self, experiments: Dict, device: str) -> List[Dict]:
        """Run all filtered experiments."""
        total = len(experiments)
        print(f"\nRunning {total} experiments...\n")
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)
        
        self._create_aggregated_results()
        
        return self.results
    
    def _create_aggregated_results(self):
        """Create aggregated results by (dataset, method)."""
        from collections import defaultdict
        
        result_groups = defaultdict(list)
        for result in self.results:
            if result['success']:
                key = (result['dataset'], result['method'])
                result_groups[key].append(result)
        
        for (dataset, method), results in result_groups.items():
            if not results:
                continue
            
            # Load individual results
            all_runs = []
            for result in results:
                results_file = result.get('results_file')
                if results_file and Path(results_file).exists():
                    with open(results_file, 'r') as f:
                        run_data = json.load(f)
                        all_runs.append(run_data)
            
            if not all_runs:
                continue
            
            # Calculate metrics
            seeds = [run['seed'] for run in all_runs]
            accuracies = [run['accuracy'] for run in all_runs]
            precisions = [run['precision'] for run in all_runs]
            recalls = [run['recall'] for run in all_runs]
            f1_scores = [run['f1_score'] for run in all_runs]
            
            aggregated = {
                'dataset': dataset,
                'method': method,
                'num_seeds': len(seeds),
                'seeds': seeds,
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies)),
                    'values': accuracies
                },
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions)),
                    'min': float(np.min(precisions)),
                    'max': float(np.max(precisions)),
                    'values': precisions
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls)),
                    'min': float(np.min(recalls)),
                    'max': float(np.max(recalls)),
                    'values': recalls
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores)),
                    'values': f1_scores
                },
                'all_runs': all_runs
            }
            
            # Save
            output_file = (self.results_dir / method / dataset /
                          'aggregated_results.json')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            
            print(f"\nAggregated ({method}/{dataset}): "
                  f"acc={aggregated['accuracy']['mean']:.4f} ± "
                  f"{aggregated['accuracy']['std']:.4f}")
    
    def save_results(self, output_file: str = 'experiments/results_log.json'):
        """Save summary results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_time_hours': sum(r['time_seconds']
                                   for r in self.results) / 3600,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to {output_path}")
        print(f"   Total: {summary['total_experiments']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Time: {summary['total_time_hours']:.1f} hours")
        print(f"{'='*70}\n")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments (Baseline, DP, FL, DP+FL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiments.py --scenario baseline
  python experiments/run_experiments.py --scenario dp --tags tier1
  python experiments/run_experiments.py --scenario fl --datasets wesad
  python experiments/run_experiments.py --scenario all --auto
        """
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
                       help='Device (cuda, cpu, auto)')
    parser.add_argument('--auto', action='store_true',
                       help='Skip confirmation')
    args = parser.parse_args()
    
    # Device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Config dir: {args.config_dir}")
    
    # Runner
    runner = ExperimentRunner(
        scenarios_dir=args.scenarios_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )
    
    # Load scenarios
    if args.scenario == 'all':
        scenarios = ['baseline', 'dp', 'fl', 'dp_fl']
    else:
        scenarios = [args.scenario]
    
    all_experiments = {}
    for scenario in scenarios:
        try:
            scenario_data = runner.load_scenario(scenario)
            all_experiments.update(scenario_data['experiments'])
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not all_experiments:
        print("No experiments found. Exiting.")
        return
    
    print(f"Loaded {len(all_experiments)} experiments\n")
    
    # Filter
    tags = args.tags.split(',') if args.tags else None
    keywords = args.keywords
    datasets = args.datasets.split(',') if args.datasets else None
    methods = args.methods.split(',') if args.methods else None
    
    filtered_experiments = runner.filter_experiments(
        all_experiments,
        tags=tags,
        keywords=keywords,
        datasets=datasets,
        methods=methods,
        epsilon=args.epsilon,
        clients=args.clients
    )
    
    # Limit number of experiments
    if args.n_experiments:
        filtered_experiments = dict(list(filtered_experiments.items())[:args.n_experiments])
    
    if not filtered_experiments:
        print("No experiments matched")
        return 1
    
    print(f"\n{'='*70}")
    print(f"Experiments to run: {len(filtered_experiments)}")
    print(f"{'='*70}")
    for exp_name in list(filtered_experiments.keys())[:10]:
        print(f"  - {exp_name}")
    if len(filtered_experiments) > 10:
        print(f"  ... and {len(filtered_experiments) - 10} more")
    
    # Confirm
    if not args.auto:
        confirm = input("\nProceed? (y/n): ").lower()
        if confirm != 'y':
            print("Cancelled")
            return 0
    else:
        print("\nAuto mode: Starting execution...\n")
    
    # Run
    start_time = time.time()
    runner.run_all(filtered_experiments, device)
    elapsed = time.time() - start_time
    
    # Save results
    runner.save_results(args.output_file)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in runner.results if r['success'])
    print(f"Successful: {successful}/{len(runner.results)}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())