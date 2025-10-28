#!/usr/bin/env python3
"""
Optimized experiment runner for Baseline, DP, FL, DP+FL training.
Consolidated and simplified for maintainability.
"""

import sys
import yaml
import json
import argparse
import torch
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import get_logger
from src.models import UnifiedLSTMModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.training.trainers.fl_trainer import FLTrainer
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_processed_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader


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
            raise FileNotFoundError(f"Scenario not found: {scenario_file}")
        
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
        """Load dataset."""
        data_path = self.data_dir / dataset
        
        if dataset == 'sleep-edf':
            return load_windowed_sleep_edf(str(data_path))
        elif dataset == 'wesad':
            return load_processed_wesad_temporal(str(data_path))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # ===== DATALOADER CREATION =====
    
    def _create_dataloaders(self, X_train, y_train, X_val, y_val,
                           X_test, y_test, batch_size: int,
                           is_dp: bool = False, num_workers: int = 0) -> Tuple:
        """Create dataloaders."""
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
            train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=is_dp, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader
    
    def _create_fl_dataloaders(self, X_train, y_train, X_val, y_val,
                              n_clients: int, batch_size: int,
                              is_dp: bool = False, num_workers: int = 0) -> Tuple[List, List]:
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
            
            train_dataset = TensorDataset(
                torch.tensor(X_client_train, dtype=torch.float32),
                torch.tensor(y_client_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_client_val, dtype=torch.float32),
                torch.tensor(y_client_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, drop_last=is_dp, num_workers=num_workers
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers
            )
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
        
        return train_loaders, val_loaders
    
    # ===== EXPERIMENT EXECUTION =====
    
    def run_experiment(self, exp_name: str, exp_config: Dict,
                      device: str) -> Dict:
        """Execute single experiment."""
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        
        print(f"\n{'='*70}")
        print(f"{exp_name} | {dataset} | {method} | seed={seed}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            set_reproducible(seed=seed, device=device, verbose=False)
            
            # Load config
            config = self._get_config(
                dataset, method,
                exp_config.get('hyperparameters')
            )
            
            print(f"Config: batch_size={config['training'].get('batch_size')}, "
                  f"lr={config['training'].get('learning_rate')}, "
                  f"epochs={config['training'].get('epochs')}\n")

            # Debug DP config
            if method == 'dp':
                dp_cfg = config.get('differential_privacy', {})
                print(f"DP Config: σ={dp_cfg.get('noise_multiplier', 'N/A')}, "
                      f"ε_target={dp_cfg.get('target_epsilon', 'N/A')}\n")
            
            # Load data
            print("Loading data...")
            data = self._load_data(dataset)
            X_train, X_val, X_test, y_train, y_val, y_test = data[:6]
            
            print(f"Shapes: train={X_train.shape}, val={X_val.shape}, "
                  f"test={X_test.shape}\n")
            
            # Create model
            print("Creating model...")
            config['dataset']['input_dim'] = X_train.shape[1]
            model = UnifiedLSTMModel(config, device=device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params:,} parameters\n")
            
            # Run experiment
            result = self._run_method(
                method, model, config, X_train, y_train,
                X_val, y_val, X_test, y_test, device
            )

            # Calculate training time (excludes saving)
            training_elapsed = time.time() - start_time
            test_metrics = result['test_metrics']

            output_dir = (self.results_dir / method / dataset /
                         f'seed_{seed}')
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / 'results.json'

            detailed_results = {
                'seed': seed,
                'method': method,
                'dataset': dataset,
                'total_epochs': result['training_results'].get('total_epochs', 0),
                'best_val_acc': result['training_results'].get('best_val_acc', 0),
                'training_time_seconds': result['training_results'].get('training_time_seconds', 0),
                'accuracy': test_metrics.get('accuracy', 0),
                'precision': test_metrics.get('precision', 0),
                'recall': test_metrics.get('recall', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'precision_per_class': test_metrics.get('precision_per_class', []),
                'recall_per_class': test_metrics.get('recall_per_class', []),
                'f1_per_class': test_metrics.get('f1_per_class', []),
                'confusion_matrix': test_metrics.get('confusion_matrix', []),
                'class_names': test_metrics.get('class_names', []),
                'total_time_seconds': training_elapsed
            }

            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)

            # Total time includes saving
            total_elapsed = time.time() - start_time

            print(f"✅ Completed in {total_elapsed:.1f}s (training: {training_elapsed:.1f}s)")
            print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"   F1-Score: {test_metrics.get('f1_score', 0):.4f}")

            # Debug final DP metrics
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

            # Track in runner summary
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

            # Track failure in runner summary
            self.results.append(final_results)

            return final_results
    
    def _run_method(self, method, model, config, X_train, y_train,
                   X_val, y_val, X_test, y_test, device) -> Dict:
        """Run training method (consolidated)."""
        batch_size = config['training']['batch_size']
        num_workers = config['training'].get('num_workers', 0)
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        if method == 'baseline':
            train_loader, val_loader, test_loader = \
                self._create_dataloaders(X_train, y_train, X_val, y_val,
                                        X_test, y_test, batch_size,
                                        num_workers=num_workers)

            output_dir = tempfile.mkdtemp(prefix='checkpoint_')
            trainer = BaselineTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=output_dir
            )
            test_metrics = trainer.evaluate_full(test_loader)
            shutil.rmtree(output_dir, ignore_errors=True)

        elif method == 'dp':
            train_loader, val_loader, test_loader = \
                self._create_dataloaders(X_train, y_train, X_val, y_val,
                                        X_test, y_test, batch_size, is_dp=True,
                                        num_workers=num_workers)

            output_dir = tempfile.mkdtemp(prefix='checkpoint_')
            trainer = DPTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loader, val_loader, epochs=epochs,
                patience=patience, output_dir=output_dir
            )
            test_metrics = trainer.evaluate_full(test_loader)
            shutil.rmtree(output_dir, ignore_errors=True)
        
        elif method == 'fl':
            n_clients = config['federated_learning'].get('n_clients', 5)
            train_loaders, val_loaders = \
                self._create_fl_dataloaders(X_train, y_train, X_val, y_val,
                                           n_clients, batch_size,
                                           num_workers=num_workers)

            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long)),
                batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            
            client_ids = [f"client_{i:02d}" for i in range(n_clients)]
            trainer = FLTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loaders, val_loaders, client_ids=client_ids,
                epochs=config['federated_learning'].get('global_rounds', 100),
                patience=patience, output_dir=None
            )
            test_metrics = trainer.evaluate_full(test_loader)
        
        elif method == 'dp_fl':
            n_clients = config['federated_learning'].get('n_clients', 5)
            train_loaders, val_loaders = \
                self._create_fl_dataloaders(X_train, y_train, X_val, y_val,
                                           n_clients, batch_size, is_dp=True,
                                           num_workers=num_workers)

            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long)),
                batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            
            client_ids = [f"client_{i:02d}" for i in range(n_clients)]
            from src.training import FLDPTrainer
            trainer = FLDPTrainer(model, config, device=device)
            training_results = trainer.fit(
                train_loaders, val_loaders, client_ids=client_ids,
                epochs=config['federated_learning'].get('global_rounds', 100),
                patience=patience, output_dir=None
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
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)
        
        return self.results
    
    def save_results(self, output_file: str = 'experiments/results_log.json'):
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
        print(f"Results: {summary['successful']}/{summary['total_experiments']} successful")
        print(f"Saved to: {output_path}")
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
                       help='Device (cuda, cpu, auto)')
    parser.add_argument('--auto', action='store_true',
                       help='Skip confirmation')
    
    args = parser.parse_args()
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu'
             if args.device is None or args.device == 'auto'
             else args.device)
    
    print(f"Device: {device}\n")
    
    runner = ExperimentRunner(
        scenarios_dir=args.scenarios_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )
    
    # Load scenarios
    scenarios = ['baseline', 'dp', 'fl', 'dp_fl'] if args.scenario == 'all' else [args.scenario]
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
    
    # Filter
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