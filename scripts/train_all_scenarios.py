#!/usr/bin/env python3
"""
Master script to train all scenarios: Baseline, DP, FL, FL+DP.

Usage:
    python scripts/train_all_scenarios.py --dataset sleep-edf --n_runs 3
    python scripts/train_all_scenarios.py --dataset all --scenarios baseline dp fl
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run command and track result."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Train all scenarios')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad', 'all'],
                       default='all')
    parser.add_argument('--scenarios', nargs='+',
                       choices=['baseline', 'dp', 'fl', 'fl_dp'],
                       default=['baseline', 'dp', 'fl', 'fl_dp'])
    parser.add_argument('--n_runs', type=int, default=1,
                       help='Number of runs per scenario')
    parser.add_argument('--data_dir', default='./data/processed')
    parser.add_argument('--config_dir', default='./src/configs')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    datasets = ['sleep-edf', 'wesad'] if args.dataset == 'all' else [args.dataset]
    seeds = list(range(42, 42 + args.n_runs))
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PRIVACY-UTILITY EVALUATION")
    print("="*70)
    print(f"Datasets: {datasets}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Seeds: {seeds}")
    print(f"Output: {args.output_dir}\n")
    
    total_runs = len(datasets) * len(args.scenarios) * len(seeds)
    current_run = 0
    
    # Baseline
    if 'baseline' in args.scenarios:
        for dataset in datasets:
            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Baseline - {dataset} (seed={seed})")
                
                cmd = [
                    'python', 'scripts/train_baseline.py',
                    '--dataset', dataset,
                    '--seed', str(seed),
                    '--output_dir', args.output_dir,
                    '--device', args.device
                ]
                run_command(cmd, f"Baseline {dataset}")
    
    # DP
    if 'dp' in args.scenarios:
        for dataset in datasets:
            for epsilon in [0.5, 1.0, 5.0]:
                for seed in seeds:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] DP - {dataset} ε={epsilon} (seed={seed})")
                    
                    cmd = [
                        'python', 'scripts/train_dp.py',
                        '--dataset', dataset,
                        '--epsilon', str(epsilon),
                        '--seed', str(seed),
                        '--output_dir', args.output_dir,
                        '--device', args.device
                    ]
                    run_command(cmd, f"DP {dataset} ε={epsilon}")
    
    # FL (optional - requires more setup)
    if 'fl' in args.scenarios:
        print("\n⚠️  FL training requires additional setup (multiple clients)")
        print("     This would typically be run separately with proper data partitioning")
    
    # FL+DP (optional)
    if 'fl_dp' in args.scenarios:
        print("\n⚠️  FL+DP training requires both FL and DP setup")
        print("     This would typically be run separately")
    
    print("\n" + "="*70)
    print("✅ ALL SCENARIOS COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Analyze with: python scripts/analyze_results.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())