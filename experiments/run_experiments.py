#!/usr/bin/env python3
"""
Simplified experiment runner - OPTIMIZED
Fast and clean execution with proper data handling
"""

import sys
import yaml
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
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
                             batch_size=128, num_workers=0):
    """
    Create simple, fast dataloaders.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        batch_size (int, optional): Batch size for dataloaders. Default is 128.
        num_workers (int, optional): Number of worker processes for data loading. Default is 0.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors once
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def create_fl_dataloaders(X_train, y_train, X_val, y_val, n_clients, 
                         batch_size=128, num_workers=0):
    """Create FL dataloaders (simple subject-aware partition)."""
    from torch.utils.data import TensorDataset, DataLoader
    
    train_loaders = []
    val_loaders = []
    
    n_train = len(X_train)
    n_val = len(X_val)
    train_per_client = n_train // n_clients
    val_per_client = n_val // n_clients
    
    for i in range(n_clients):
        # Train split
        start = i * train_per_client
        end = start + train_per_client if i < n_clients - 1 else n_train
        
        X_client_t = torch.from_numpy(X_train[start:end].astype(np.float32))
        y_client_t = torch.from_numpy(y_train[start:end].astype(np.int64))
        
        train_dataset = TensorDataset(X_client_t, y_client_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loaders.append(train_loader)
        
        # Val split
        val_start = i * val_per_client
        val_end = val_start + val_per_client if i < n_clients - 1 else n_val
        
        X_val_client_t = torch.from_numpy(X_val[val_start:val_end].astype(np.float32))
        y_val_client_t = torch.from_numpy(y_val[val_start:val_end].astype(np.int64))
        
        val_dataset = TensorDataset(X_val_client_t, y_val_client_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders


class SimpleExperimentRunner:
    """Simple, fast experiment runner - OPTIMIZED."""
    
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
        self.logger = get_logger(__name__)

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge configs."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _get_config(self, dataset: str, method: str, exp_hyperparams=None):
        """Get merged config."""
        dataset_cfg = self._load_config(self.config_dir / 'datasets' / f'{dataset}.yaml')
        method_cfg = self._load_config(self.config_dir / 'methods' / f'{method}.yaml')
        
        config = self._merge_configs(dataset_cfg, method_cfg)
        
        if exp_hyperparams:
            for section, params in exp_hyperparams.items():
                config[section] = self._merge_configs(config.get(section, {}), params)
        
        return config

    def load_scenario(self, scenario_name: str) -> Dict:
        """Load scenario YAML."""
        scenario_file = self.scenarios_dir / f'{scenario_name}.yaml'
        with open(scenario_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_data(self, dataset: str) -> Tuple:
        """Load dataset with caching."""
        if dataset in self._data_cache:
            return self._data_cache[dataset]
        
        print(f"\n📥 Loading {dataset}...", flush=True)
        data_path = self.data_dir / dataset
        
        try:
            if dataset == 'sleep-edf':
                data = load_windowed_sleep_edf(str(data_path))
            elif dataset == 'wesad':
                data = load_processed_wesad_temporal(str(data_path))
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            self._data_cache[dataset] = data
            print(f"✅ {dataset} loaded and cached", flush=True)
            return data
        
        except Exception as e:
            print(f"❌ Failed to load {dataset}: {e}", flush=True)
            raise

    def run_experiment(self, exp_name: str, exp_config: Dict, device: str) -> Dict:
        """Execute single experiment."""
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        
        print(f"\n{'='*60}")
        print(f"🚀 {exp_name}")
        print(f"   Dataset: {dataset} | Method: {method} | Seed: {seed}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Setup
            set_reproducible(seed=seed, device=device, verbose=False)
            config = self._get_config(dataset, method, exp_config.get('hyperparameters'))
            
            # Load data
            data = self._load_data(dataset)
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects = data
            
            print(f"Loaded: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
            
            # Setup config - input_dim é número de canais
            config['dataset']['input_dim'] = X_train.shape[1]
            
            # Create model
            model = UnifiedLSTMModel(config, device=device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params:,} parameters")
            
            # Output directory
            output_dir = self.results_dir / method / dataset / f'seed_{seed}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get training config
            batch_size = config['training'].get('batch_size', 32)
            epochs = config['training'].get('epochs', 40)
            patience = config['training'].get('early_stopping_patience', 10)
            num_workers = config['training'].get('num_workers', 0)

            # Print / log effective training parameters so the user can see them
            self.logger.info(
                f"Training config:\n"
                f"  batch_size={batch_size}\n"
                f"  num_workers={num_workers}\n"
                f"  epochs={epochs}\n"
                f"  patience={patience}"
            )
            
            # Run training based on method
            if method == 'baseline':
                print("\n📊 Training: Baseline")
                train_loader, val_loader, test_loader = create_simple_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers=num_workers
                )
                trainer = BaselineTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loader, val_loader, 
                    epochs=epochs, patience=patience,
                    output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)
                
            elif method == 'dp':
                print("\n🔒 Training: Differential Privacy")
                train_loader, val_loader, test_loader = create_simple_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers=num_workers
                )
                trainer = DPTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loader, val_loader,
                    epochs=epochs, patience=patience,
                    output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)
                
            elif method == 'fl':
                print("\n🌐 Training: Federated Learning")
                n_clients = config['federated_learning'].get('n_clients', 5)
                print(f"   Clients: {n_clients}")
                
                train_loaders, val_loaders = create_fl_dataloaders(
                    X_train, y_train, X_val, y_val, n_clients, batch_size, num_workers=num_workers
                )
                
                _, _, test_loader = create_simple_dataloaders(
                    X_test, y_test, X_test, y_test, X_test, y_test, batch_size, num_workers=num_workers
                )
                
                client_ids = [f"client_{i:02d}" for i in range(n_clients)]
                trainer = FLTrainer(model, config, device=device)
                training_results = trainer.fit(
                    train_loaders, val_loaders, 
                    client_ids=client_ids,
                    epochs=config['federated_learning'].get('global_rounds', 100),
                    patience=patience, 
                    output_dir=str(output_dir)
                )
                test_metrics = trainer.evaluate_full(test_loader)
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed = time.time() - start_time
            
            # Save detailed results
            results_data = {
                'experiment_info': {
                    'name': exp_name,
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'hyperparameters': exp_config.get('hyperparameters', {})
                },
                'training_metrics': {
                    'total_epochs': training_results.get('total_epochs', 0),
                    'best_val_acc': training_results.get('best_val_acc', 0),
                    'training_time_seconds': training_results.get('training_time_seconds', 0),
                    'final_train_loss': training_results.get('final_train_loss', 0),
                    'final_train_acc': training_results.get('final_train_acc', 0),
                    'final_val_loss': training_results.get('final_val_loss', 0),
                    'final_val_acc': training_results.get('final_val_acc', 0),
                },
                'test_metrics': {
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'f1_score': test_metrics.get('f1_score', 0),
                    'confusion_matrix': test_metrics.get('confusion_matrix', []),
                    'class_names': test_metrics.get('class_names', [])
                },
                'timing': {
                    'total_time_seconds': elapsed,
                    'training_time_seconds': training_results.get('training_time_seconds', 0),
                    'evaluation_time_seconds': elapsed - training_results.get('training_time_seconds', 0)
                }
            }
            
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Print summary
            print(f"\n✅ Completed in {elapsed:.1f}s")
            print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"   F1-Score: {test_metrics.get('f1_score', 0):.4f}")
            print(f"   Best Val Acc: {training_results.get('best_val_acc', 0):.4f}")
            
            if method == 'dp' and 'final_epsilon' in test_metrics:
                print(f"   Final ε: {test_metrics['final_epsilon']:.4f}")
            
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
            print(f"\n❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"{exp_name} failed: {e}", exc_info=True)
            
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

    def run_all(self, experiments: Dict, device: str) -> List[Dict]:
        """Run all experiments."""
        total = len(experiments)
        print(f"\n{'='*60}")
        print(f"🚀 Running {total} experiments")
        print(f"{'='*60}")
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"\n[{idx}/{total}]", flush=True)
            self.run_experiment(exp_name, exp_config, device)
            
            # Perform memory cleanup every 3 experiments
            if idx % 3 == 0:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        return self.results

    def _create_case_key(self, result: Dict) -> str:
        """Create a unique key for grouping experiments by case (ignoring seed)."""
        dataset = result.get('dataset', 'unknown')
        method = result.get('method', 'unknown')
        
        # Try to extract hyperparameters from experiment name or load from file
        exp_name = result.get('name', '')
        hyperparams_str = ''
        
        # Try to load detailed results to get hyperparameters
        seed = result.get('seed', 'unknown')
        if result.get('success', False):
            results_file = self.results_dir / method / dataset / f'seed_{seed}' / 'results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        detailed = json.load(f)
                    hyperparams = detailed.get('experiment_info', {}).get('hyperparameters', {})
                    # Create a string representation of hyperparameters
                    hyperparams_str = json.dumps(hyperparams, sort_keys=True)
                except:
                    pass
        
        return f"{dataset}::{method}::{hyperparams_str}"
    
    def _load_detailed_results(self, method: str, dataset: str, seed: int) -> Optional[Dict]:
        """Load detailed results from JSON file."""
        results_file = self.results_dir / method / dataset / f'seed_{seed}' / 'results.json'
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _calculate_aggregated_metrics(self, results_group: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated metrics (mean, std, min, max) for a group of runs."""
        if not results_group:
            return {}
        
        # Load detailed results for all runs in the group
        detailed_results = []
        for result in results_group:
            if not result.get('success', False):
                continue
            
            dataset = result.get('dataset')
            method = result.get('method')
            seed = result.get('seed')
            
            detailed = self._load_detailed_results(method, dataset, seed)
            if detailed:
                detailed_results.append(detailed)
        
        if not detailed_results:
            return {}
        
        # Extract metrics from detailed results
        metrics_to_aggregate = {
            'test_accuracy': [],
            'test_f1_score': [],
            'test_precision': [],
            'test_recall': [],
            'best_val_acc': [],
            'training_time_seconds': [],
            'total_epochs': []
        }
        
        for detailed in detailed_results:
            test_metrics = detailed.get('test_metrics', {})
            training_metrics = detailed.get('training_metrics', {})
            
            if 'accuracy' in test_metrics:
                metrics_to_aggregate['test_accuracy'].append(test_metrics['accuracy'])
            if 'f1_score' in test_metrics:
                metrics_to_aggregate['test_f1_score'].append(test_metrics['f1_score'])
            if 'precision' in test_metrics:
                metrics_to_aggregate['test_precision'].append(test_metrics['precision'])
            if 'recall' in test_metrics:
                metrics_to_aggregate['test_recall'].append(test_metrics['recall'])
            if 'best_val_acc' in training_metrics:
                metrics_to_aggregate['best_val_acc'].append(training_metrics['best_val_acc'])
            if 'training_time_seconds' in training_metrics:
                metrics_to_aggregate['training_time_seconds'].append(training_metrics['training_time_seconds'])
            if 'total_epochs' in training_metrics:
                metrics_to_aggregate['total_epochs'].append(training_metrics['total_epochs'])
        
        # Calculate statistics for each metric
        aggregated = {}
        for metric_name, values in metrics_to_aggregate.items():
            if not values:
                continue
            
            values_array = np.array(values)
            aggregated[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'n_runs': len(values)
            }
        
        return aggregated
    
    def calculate_aggregated_results(self) -> Dict[str, Any]:
        """Calculate aggregated metrics across multiple runs for each case."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            return {}
        
        # Group results by case (dataset + method + hyperparameters)
        grouped_results = {}
        for result in successful_results:
            case_key = self._create_case_key(result)
            if case_key not in grouped_results:
                grouped_results[case_key] = []
            grouped_results[case_key].append(result)
        
        # Calculate aggregated metrics for each group
        aggregated_results = {}
        for case_key, results_group in grouped_results.items():
            # Get case info from first result
            first_result = results_group[0]
            dataset = first_result.get('dataset', 'unknown')
            method = first_result.get('method', 'unknown')
            
            # Load hyperparameters from first detailed result
            seed = first_result.get('seed')
            detailed = self._load_detailed_results(method, dataset, seed)
            hyperparams = {}
            if detailed:
                hyperparams = detailed.get('experiment_info', {}).get('hyperparameters', {})
            
            # Calculate aggregated metrics
            aggregated_metrics = self._calculate_aggregated_metrics(results_group)
            
            aggregated_results[case_key] = {
                'dataset': dataset,
                'method': method,
                'hyperparameters': hyperparams,
                'n_runs': len(results_group),
                'seeds': [r.get('seed') for r in results_group],
                'metrics': aggregated_metrics
            }
        
        return aggregated_results
    
    def save_results(self, output_file='experiments/results_log.json'):
        """Save results summary."""
        if not self.results:
            print("No results to save")
            return
        
        successful_results = [r for r in self.results if r['success']]
        
        # Calculate aggregated metrics
        aggregated_results = self.calculate_aggregated_results()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': len(successful_results),
            'failed': len(self.results) - len(successful_results),
            'average_accuracy': np.mean([r['accuracy'] for r in successful_results]) if successful_results else 0,
            'average_f1': np.mean([r['f1_score'] for r in successful_results]) if successful_results else 0,
            'total_time_hours': sum(r['time_seconds'] for r in self.results) / 3600,
            'results': self.results,
            'aggregated_results': aggregated_results
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"❌ Failed: {summary['failed']}/{summary['total_experiments']}")
        if successful_results:
            print(f"📈 Avg Accuracy: {summary['average_accuracy']:.4f}")
            print(f"📈 Avg F1-Score: {summary['average_f1']:.4f}")
        print(f"⏱️  Total time: {summary['total_time_hours']:.2f} hours")
        print(f"💾 Saved to: {output_file}")
        
        # Print aggregated results summary
        if aggregated_results:
            print(f"\n{'='*60}")
            print(f"📊 AGGREGATED METRICS (across multiple runs)")
            print(f"{'='*60}")
            for case_key, case_data in aggregated_results.items():
                dataset = case_data['dataset']
                method = case_data['method']
                n_runs = case_data['n_runs']
                metrics = case_data.get('metrics', {})
                
                print(f"\n{method.upper()} - {dataset} ({n_runs} runs)")
                if metrics:
                    if 'test_accuracy' in metrics:
                        acc = metrics['test_accuracy']
                        print(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f} "
                              f"(min: {acc['min']:.4f}, max: {acc['max']:.4f})")
                    if 'test_f1_score' in metrics:
                        f1 = metrics['test_f1_score']
                        print(f"  F1-Score:  {f1['mean']:.4f} ± {f1['std']:.4f} "
                              f"(min: {f1['min']:.4f}, max: {f1['max']:.4f})")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Simple experiment runner - OPTIMIZED',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiments.py --scenario baseline
  python experiments/run_experiments.py --scenario fl --datasets wesad
  python experiments/run_experiments.py --scenario dp --n_experiments 3
        """
    )
    
    parser.add_argument(
        '--scenario',
        choices=['baseline', 'dp', 'fl', 'dp_fl', 'all'],
        default='baseline',
        help='Scenario to run'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device (cuda/mps/cpu/auto)'
    )
    parser.add_argument(
        '--datasets',
        help='Filter datasets (comma-separated: sleep-edf,wesad)'
    )
    parser.add_argument(
        '--n_experiments',
        type=int,
        help='Limit number of experiments'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Skip confirmation'
    )
    parser.add_argument(
        '--output_file',
        default='experiments/results_log.json',
        help='Output results file'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device() if args.device == 'auto' else args.device
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 CUDA Device: {gpu_name}")
        print(f"💾 CUDA Memory: {gpu_memory:.1f}GB\n")
        
    # Create runner
    runner = SimpleExperimentRunner()
    
    # Load scenarios
    if args.scenario == 'all':
        scenarios = ['baseline', 'dp', 'fl']
    else:
        scenarios = [args.scenario]
    
    all_experiments = {}
    for scenario in scenarios:
        try:
            scenario_data = runner.load_scenario(scenario)
            
            # Validate scenario_data
            if scenario_data is None:
                print(f"⚠️  Scenario {scenario} returned None")
                continue
            
            # Get experiments dictionary
            experiments = scenario_data.get('experiments')
            if experiments is None:
                print(f"⚠️  Scenario {scenario} has no 'experiments' key")
                continue
            
            if not isinstance(experiments, dict):
                print(f"⚠️  Scenario {scenario} 'experiments' is not a dict: {type(experiments)}")
                continue
            
            # Filter only enabled experiments
            enabled_experiments = {
                name: config for name, config in experiments.items()
                if isinstance(config, dict) and config.get('enabled', True)  # Default to True if not specified
            }
            all_experiments.update(enabled_experiments)
        except FileNotFoundError:
            print(f"⚠️  Scenario not found: {scenario}")
            continue
        except Exception as e:
            print(f"⚠️  Error loading scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_experiments:
        print("❌ No experiments found")
        return 1
    
    # Filter experiments
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
        all_experiments = {
            name: config for name, config in all_experiments.items()
            if config.get('dataset') in datasets
        }
    
    if args.n_experiments:
        all_experiments = dict(list(all_experiments.items())[:args.n_experiments])
    
    if not all_experiments:
        print("❌ No experiments matched filters")
        return 1
    
    # Show summary
    print(f"\n{'='*60}")
    print(f"📋 Will run {len(all_experiments)} experiments:")
    for exp_name, exp_config in list(all_experiments.items())[:5]:
        print(f"   • {exp_name}: {exp_config['dataset']} / {exp_config['method']}")
    if len(all_experiments) > 5:
        print(f"   ... and {len(all_experiments) - 5} more")
    print(f"{'='*60}")
    
    # Confirmation
    if not args.auto:
        response = input("\n🤔 Proceed? (y/n): ").lower().strip()
        if response != 'y':
            print("⏹️  Cancelled")
            return 0
    
    # Run experiments
    start_time = time.time()
    runner.run_all(all_experiments, device)
    elapsed = time.time() - start_time
    
    # Save results
    runner.save_results(args.output_file)
    
    print(f"⏱️  Total execution time: {elapsed/60:.1f} minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())