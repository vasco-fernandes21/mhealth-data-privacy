#!/usr/bin/env python3
"""
Master script to run experiments from YAML configuration files.

The script reads YAML files, filters by tags/criteria, and executes.

Usage:
    python experiments/run_experiments.py --scenario baseline
    python experiments/run_experiments.py --scenario dp --tags tier1
    python experiments/run_experiments.py --scenario all --tags "sleep-edf,tier1"
    python experiments/run_experiments.py --scenario fl --filter "clients5"
    python experiments/run_experiments.py --scenario dp --epsilon 1.0
"""

import subprocess
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time


class ExperimentRunner:
    """Run experiments from YAML configuration."""
    
    def __init__(self, scenarios_dir: str = 'experiments/scenarios'):
        self.scenarios_dir = Path(scenarios_dir)
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
                          clients: int = None) -> Dict:
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
            
            # Filter by keywords in name
            if keywords:
                if keywords.lower() not in exp_name.lower():
                    continue
            
            # Filter by epsilon (for DP experiments)
            if epsilon is not None:
                hp = exp_config.get('hyperparameters', {})
                dp_cfg = hp.get('differential_privacy', {})
                if dp_cfg.get('target_epsilon') != epsilon:
                    continue
            
            # Filter by n_clients (for FL experiments)
            if clients is not None:
                hp = exp_config.get('hyperparameters', {})
                fl_cfg = hp.get('federated_learning', {})
                if fl_cfg.get('n_clients') != clients:
                    continue
            
            filtered[exp_name] = exp_config
        
        return filtered
    
    def run_experiment(self, exp_name: str, exp_config: Dict) -> Dict:
        """Execute single experiment."""
        dataset = exp_config['dataset']
        method = exp_config['method']
        seed = exp_config['seed']
        device = exp_config['device']
        
        print(f"\n{'='*70}")
        print(f"🚀 Running: {exp_name}")
        print(f"   Dataset: {dataset}, Method: {method}, Seed: {seed}")
        print(f"{'='*70}")
        
        # Build command based on method
        if method == 'baseline':
            cmd = f"python scripts/train_baseline.py --dataset {dataset} --seed {seed} --device {device}"
        
        elif method == 'dp':
            hp = exp_config['hyperparameters']
            dp_cfg = hp['differential_privacy']
            epsilon = dp_cfg['target_epsilon']
            cmd = f"python scripts/train_dp.py --dataset {dataset} --epsilon {epsilon} --seed {seed} --device {device}"
        
        elif method == 'fl':
            hp = exp_config['hyperparameters']
            fl_cfg = hp['federated_learning']
            n_clients = fl_cfg['n_clients']
            cmd = f"python scripts/train_fl.py --dataset {dataset} --n_clients {n_clients} --seed {seed} --device {device}"
        
        elif method == 'fl_dp':
            hp = exp_config['hyperparameters']
            fl_cfg = hp.get('federated_learning', {})
            dp_cfg = hp.get('differential_privacy', {})
            n_clients = fl_cfg['n_clients']
            epsilon = dp_cfg['target_epsilon']
            cmd = f"python scripts/train_fl_dp.py --dataset {dataset} --n_clients {n_clients} --epsilon {epsilon} --seed {seed} --device {device}"
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Execute command
        start_time = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                print(f"✅ Completed in {elapsed:.1f}s")
            else:
                print(f"❌ Failed after {elapsed:.1f}s")
                print(f"Error: {result.stderr}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            success = False
            print(f"❌ Exception: {e}")
        
        result_record = {
            'experiment_name': exp_name,
            'dataset': dataset,
            'method': method,
            'seed': seed,
            'success': success,
            'elapsed_time_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result_record)
        return result_record
    
    def run_all(self, experiments: Dict) -> List[Dict]:
        """Run all filtered experiments."""
        total = len(experiments)
        print(f"\n📊 Running {total} experiments...")
        
        for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            print(f"\n[{idx}/{total}]", end=" ")
            self.run_experiment(exp_name, exp_config)
        
        return self.results
    
    def save_results(self, output_file: str = 'experiments/results_log.json'):
        """Save experiment results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_time_hours': sum(r['elapsed_time_seconds'] for r in self.results) / 3600,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"   Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"   Total time: {summary['total_time_hours']:.1f} hours")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments from YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baseline experiments
  python experiments/run_experiments.py --scenario baseline
  
  # Run DP experiments with epsilon=1.0
  python experiments/run_experiments.py --scenario dp --epsilon 1.0
  
  # Run FL experiments with 5 clients on WESAD
  python experiments/run_experiments.py --scenario fl --clients 5 --keywords wesad
  
  # Run all tier1 experiments
  python experiments/run_experiments.py --scenario all --tags tier1
  
  # Run everything
  python experiments/run_experiments.py --scenario all
        """
    )
    
    parser.add_argument('--scenario', choices=['baseline', 'dp', 'fl', 'fl_dp', 'all'],
                       default='baseline', help='Which scenario to run')
    parser.add_argument('--tags', type=str, help='Filter by tags (comma-separated: tier1,sleep-edf)')
    parser.add_argument('--keywords', type=str, help='Filter by keywords in experiment name')
    parser.add_argument('--epsilon', type=float, help='Filter by epsilon value (DP only)')
    parser.add_argument('--clients', type=int, help='Filter by number of clients (FL only)')
    parser.add_argument('--output_file', default='experiments/results_log.json',
                       help='Output file for results')
    parser.add_argument('--scenarios_dir', default='experiments/scenarios',
                       help='Directory with scenario YAML files')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(scenarios_dir=args.scenarios_dir)
    
    # Load scenario(s)
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
    
    # Filter experiments
    tags = args.tags.split(',') if args.tags else None
    filtered_experiments = runner.filter_experiments(
        all_experiments,
        tags=tags,
        keywords=args.keywords,
        epsilon=args.epsilon,
        clients=args.clients
    )
    
    if not filtered_experiments:
        print("❌ No experiments matched the filter criteria")
        return 1
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS TO RUN: {len(filtered_experiments)}")
    print(f"{'='*70}")
    for exp_name in filtered_experiments.keys():
        print(f"  - {exp_name}")
    
    # Ask for confirmation
    confirm = input("\nProceed? (y/n): ").lower()
    if confirm != 'y':
        print("Cancelled")
        return 0
    
    # Run experiments
    start_time = time.time()
    results = runner.run_all(filtered_experiments)
    elapsed = time.time() - start_time
    
    # Save results
    runner.save_results(args.output_file)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"⏱️  Total time: {elapsed/3600:.2f} hours")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())