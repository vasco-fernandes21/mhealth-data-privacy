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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_fl_dataloaders(X_train, y_train, X_val, y_val, n_clients, 
                         batch_size=128):
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
        
        # Val split
        val_start = i * val_per_client
        val_end = val_start + val_per_client if i < n_clients - 1 else n_val
        
        X_val_client_t = torch.from_numpy(X_val[val_start:val_end].astype(np.float32))
        y_val_client_t = torch.from_numpy(y_val[val_start:val_end].astype(np.int64))
        
        val_dataset = TensorDataset(X_val_client_t, y_val_client_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
            
            # Run training based on method
            if method == 'baseline':
                print("\n📊 Training: Baseline")
                train_loader, val_loader, test_loader = create_simple_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size
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
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size
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
                    X_train, y_train, X_val, y_val, n_clients, batch_size
                )
                
                _, _, test_loader = create_simple_dataloaders(
                    X_test, y_test, X_test, y_test, X_test, y_test, batch_size
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
            
            # Cleanup periodically
            if idx % 3 == 0:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        return self.results

    def save_results(self, output_file='experiments/results_log.json'):
        """Save results summary."""
        if not self.results:
            print("No results to save")
            return
        
        successful_results = [r for r in self.results if r['success']]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': len(successful_results),
            'failed': len(self.results) - len(successful_results),
            'average_accuracy': np.mean([r['accuracy'] for r in successful_results]) if successful_results else 0,
            'average_f1': np.mean([r['f1_score'] for r in successful_results]) if successful_results else 0,
            'total_time_hours': sum(r['time_seconds'] for r in self.results) / 3600,
            'results': self.results
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
            all_experiments.update(scenario_data.get('experiments', {}))
        except FileNotFoundError:
            print(f"⚠️  Scenario not found: {scenario}")
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