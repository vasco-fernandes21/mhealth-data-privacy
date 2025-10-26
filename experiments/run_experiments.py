#!/usr/bin/env python3
"""
Master script to run experiments from YAML configuration files.
Directly imports and executes training functions (no subprocess overhead).

Usage:
    python experiments/run_experiments.py --scenario baseline
    python experiments/run_experiments.py --scenario dp --tags tier1
    python experiments/run_experiments.py --scenario all
"""

import sys
import yaml
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_reproducible
from src.utils.logging_utils import setup_logging, get_logger
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.preprocessing.sleep_edf import load_windowed_sleep_edf
from src.preprocessing.wesad import load_processed_wesad_temporal
from torch.utils.data import TensorDataset, DataLoader


class ExperimentRunner:
    """Run experiments directly (no subprocess overhead)."""
    
    def __init__(self, scenarios_dir: str = 'experiments/scenarios', 
                 data_dir: str = './data/processed',
                 config_dir: str = './src/configs',
                 results_dir: str = './results'):
        self.scenarios_dir = Path(scenarios_dir)
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results = []
        
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
            # Skip disabled experiments
            if not exp_config.get('enabled', True):
                continue
            
            # Filter by tags
            if tags:
                exp_tags = exp_config.get('tags', [])
                if not any(tag in exp_tags for tag in tags):
                    continue
            
            # Filter by keywords
            if keywords and keywords.lower() not in exp_name.lower():
                continue
            
            # Filter by datasets
            if datasets and exp_config.get('dataset') not in datasets:
                continue
            
            # Filter by methods
            if methods and exp_config.get('method') not in methods:
                continue
            
            # Filter by epsilon (DP)
            if epsilon is not None:
                hp = exp_config.get('hyperparameters', {})
                dp_cfg = hp.get('differential_privacy', {})
                if dp_cfg.get('target_epsilon') != epsilon:
                    continue
            
            # Filter by clients (FL)
            if clients is not None:
                hp = exp_config.get('hyperparameters', {})
                fl_cfg = hp.get('federated_learning', {})
                if fl_cfg.get('n_clients') != clients:
                    continue
            
            filtered[exp_name] = exp_config
        
        return filtered
    
    def _load_data(self, dataset: str):
        """Load dataset."""
        data_path = self.data_dir / dataset
        
        if dataset == 'sleep-edf':
            (X_train, X_val, X_test, y_train, y_val, y_test,
             scaler, info, subjects_train) = load_windowed_sleep_edf(str(data_path))
        elif dataset == 'wesad':
            (X_train, X_val, X_test, y_train, y_val, y_test,
             label_encoder, info) = load_processed_wesad_temporal(str(data_path))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, info
    
    def _create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test,
                           batch_size: int):
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _get_model(self, dataset: str, config: dict, device: str):
        """Create model based on dataset."""
        if dataset == 'sleep-edf':
            return SleepEDFModel(config, device=device)
        elif dataset == 'wesad':
            return WESADModel(config, device=device)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, *configs) -> dict:
        """Deep merge configs."""
        from copy import deepcopy
        merged = {}
        for cfg in configs:
            if cfg is None:
                continue
            for key, value in cfg.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    self._merge_configs(merged[key], value)
                else:
                    merged[key] = deepcopy(value)
        return merged
    
    def run_experiment(self, exp_name: str, exp_config: Dict, device: str) -> Dict:
        """Execute single experiment directly (no subprocess)."""
        
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        
        print(f"\n{'='*70}")
        print(f"🚀 {exp_name}")
        print(f"   Dataset: {dataset}, Method: {method}, Seed: {seed}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success = False
        
        try:
            # Set reproducibility
            set_reproducible(seed=seed, device=device, verbose=False)
            
            # Load configs
            default_cfg = self._load_config(self.config_dir / 'training_defaults.yaml')
            
            config_mapping = {
                'sleep-edf': 'sleep_edf.yaml',
                'wesad': 'wesad.yaml'
            }
            config_filename = config_mapping.get(dataset, f'{dataset}.yaml')
            dataset_cfg = self._load_config(self.config_dir / config_filename)
            config = self._merge_configs(default_cfg, dataset_cfg)
            
            # Load data
            print("📊 Loading data...")
            X_train, X_val, X_test, y_train, y_val, y_test, info = \
                self._load_data(dataset)
            print(f"✅ Data loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            
            # Create dataloaders
            batch_size = config['training']['batch_size']
            train_loader, val_loader, test_loader = self._create_dataloaders(
                X_train, y_train, X_val, y_val, X_test, y_test, batch_size
            )
            
            # Create model
            print("🏗️  Creating model...")
            model = self._get_model(dataset, config, device)
            
            # Create trainer
            trainer = BaselineTrainer(model, config, device=device)
            
            # Train
            print("🎯 Training...")
            results = trainer.fit(
                train_loader,
                val_loader,
                epochs=config['training']['epochs'],
                patience=config['training'].get('early_stopping_patience', 10),
                output_dir=str(self.results_dir / 'baseline' / dataset / f'seed_{seed}')
            )
            
            # Evaluate on test set
            print("📈 Evaluating...")
            test_metrics = trainer.evaluate_full(test_loader)
            
            elapsed = time.time() - start_time
            success = True
            
            final_results = {
                'name': exp_name,
                'dataset': dataset,
                'method': method,
                'seed': seed,
                'success': success,
                'time_seconds': elapsed,
                'accuracy': test_metrics.get('accuracy', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Completed in {elapsed:.1f}s")
            print(f"   Accuracy: {final_results['accuracy']:.4f}")
            print(f"   F1-Score: {final_results['f1_score']:.4f}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: {e}")
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
    
    def run_all(self, experiments: Dict, device: str) -> List[Dict]:
        """Run all filtered experiments."""
        total = len(experiments)
        print(f"\n📊 Running {total} experiments...\n")
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"[{idx}/{total}]")
            self.run_experiment(exp_name, exp_config, device)
        
        return self.results
    
    def save_results(self, output_file: str = 'experiments/results_log.json'):
        """Save results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_time_hours': sum(r['time_seconds'] for r in self.results) / 3600,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Results saved to {output_path}")
        print(f"   Total: {summary['total_experiments']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Time: {summary['total_time_hours']:.1f} hours")
        print(f"{'='*70}\n")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments from YAML (direct execution)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiments.py --scenario baseline
  python experiments/run_experiments.py --scenario baseline --tags tier1 --datasets wesad
  python experiments/run_experiments.py --scenario all
  python experiments/run_experiments.py --scenario baseline --n_experiments 3
        """
    )
    
    parser.add_argument('--scenario', choices=['baseline', 'dp', 'fl', 'fl_dp', 'all'],
                       default='baseline', help='Which scenario to run')
    parser.add_argument('--tags', type=str, help='Filter by tags (comma-separated)')
    parser.add_argument('--keywords', type=str, help='Filter by keywords')
    parser.add_argument('--datasets', type=str, help='Filter by datasets (comma-separated)')
    parser.add_argument('--methods', type=str, help='Filter by methods (comma-separated)')
    parser.add_argument('--epsilon', type=float, help='Filter by epsilon (DP)')
    parser.add_argument('--clients', type=int, help='Filter by n_clients (FL)')
    parser.add_argument('--n_experiments', type=int, default=None,
                       help='Limit number of experiments')
    parser.add_argument('--output_file', default='experiments/results_log.json')
    parser.add_argument('--scenarios_dir', default='experiments/scenarios')
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--device', default=None, help='Device (cuda, cpu, auto)')
    parser.add_argument('--auto', action='store_true',
                   help='Skip confirmation prompt (ideal for Colab/CI)')
    args = parser.parse_args()
    
    # Determine device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"✅ Device: {device}")
    print(f"✅ Data dir: {args.data_dir}")
    print(f"✅ Config dir: {args.config_dir}")
    
    # Initialize runner
    runner = ExperimentRunner(
        scenarios_dir=args.scenarios_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )
    
    # Load scenarios
    if args.scenario == 'all':
        scenarios = ['baseline', 'dp', 'fl']
    else:
        scenarios = [args.scenario]
    
    all_experiments = {}
    for scenario in scenarios:
        try:
            scenario_data = runner.load_scenario(scenario)
            all_experiments.update(scenario_data['experiments'])
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            continue
    
    print(f"✅ Loaded {len(all_experiments)} experiments\n")
    
    # Filter experiments
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
        print("❌ No experiments matched")
        return 1
    
    print(f"\n{'='*70}")
    print(f"📋 Experiments to run: {len(filtered_experiments)}")
    print(f"{'='*70}")
    for exp_name in list(filtered_experiments.keys())[:10]:
        print(f"  - {exp_name}")
    if len(filtered_experiments) > 10:
        print(f"  ... and {len(filtered_experiments) - 10} more")
    
    # Confirm
    if not args.auto:
        confirm = input("\n👉 Proceed? (y/n): ").lower()
        if confirm != 'y':
            print("Cancelled")
            return 0
    else:
        print("\n✅ Auto mode: Starting execution...\n")
    
    # Run
    start_time = time.time()
    runner.run_all(filtered_experiments, device)
    elapsed = time.time() - start_time
    
    # Save results
    runner.save_results(args.output_file)
    
    # Summary
    print(f"\n{'='*70}")
    print("🎉 SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in runner.results if r['success'])
    print(f"✅ Successful: {successful}/{len(runner.results)}")
    print(f"⏱️  Total time: {elapsed/3600:.2f} hours")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())