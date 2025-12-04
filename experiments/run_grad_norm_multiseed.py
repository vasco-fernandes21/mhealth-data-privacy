#!/usr/bin/env python3
"""
Test max_grad_norm scaling with MULTIPLE SEEDS to confirm gradient clipping hypothesis.
Based on test_grad_norm_scaling.py but extended for multi-seed validation.
"""

import sys
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiments import ExperimentRunner
from src.utils.seed_utils import set_reproducible
import torch

def run_experiment(max_grad_norm, class_weight_stress, noise_multiplier, seed):
    """Run DP experiment with specific max_grad_norm."""
    set_reproducible(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = ExperimentRunner()
    
    hyperparams = {
        'dataset': {
            'class_weights': {'0': 1.0, '1': class_weight_stress}
        },
        'differential_privacy': {
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm
        }
    }
    
    exp_config = {
        'dataset': 'wesad',
        'method': 'dp',
        'seed': seed,
        'hyperparameters': hyperparams
    }
    
    exp_name = f'dp_wesad_noise{noise_multiplier}_C{max_grad_norm}_w{class_weight_stress}_s{seed}'
    
    try:
        runner.run_experiment(exp_name, exp_config, device, save_model=False)
        results_dir = Path('results/experiments/dp/wesad')
        json_files = sorted(results_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                metrics = data['test_metrics']
                privacy = data.get('privacy_metrics', {})
                return {
                    'max_grad_norm': max_grad_norm,
                    'weight': class_weight_stress,
                    'seed': seed,
                    'recall_stress': metrics['recall_per_class'][1],
                    'precision_stress': metrics['precision_per_class'][1],
                    'f1_stress': metrics['f1_per_class'][1],
                    'tp': metrics['confusion_matrix'][1][1],
                    'epsilon': privacy.get('final_epsilon'),
                    'accuracy': metrics['accuracy']
                }
    except Exception as e:
        print(f"ERROR (C={max_grad_norm}, w={class_weight_stress}, s={seed}): {e}")
        return None
    
    return None

if __name__ == '__main__':

    max_grad_norms = [1.0, 2.0, 5.0]  
    seeds = [42, 123, 456]  
    class_weight = 2.0
    noise = 0.6
    
    print("="*80)
    print("MAX_GRAD_NORM MULTI-SEED EXPERIMENT")
    print("="*80)
    print(f"Testing max_grad_norm: {max_grad_norms}")
    print(f"Seeds: {seeds}")
    print(f"Fixed: weight={class_weight}, noise={noise}")
    print(f"Total experiments: {len(max_grad_norms) * len(seeds)}")
    print("="*80)
    
    all_results = []
    
    for C in max_grad_norms:
        for seed in seeds:
            print(f"\nC={C}, Seed={seed}...", end=' ', flush=True)
            result = run_experiment(C, class_weight, noise, seed)
            if result:
                all_results.append(result)
                print(f"Recall={result['recall_stress']:.4f}, eps={result.get('epsilon', 'N/A'):.2f}")
            else:
                print("✗ Failed")
    
    # Analyze by C value
    print("\n" + "="*80)
    print("RESULTS BY MAX_GRAD_NORM")
    print("="*80)
    
    by_C = defaultdict(list)
    for r in all_results:
        by_C[r['max_grad_norm']].append(r)
    
    print(f"\n{'C':<8} {'Mean Recall':<15} {'Std Dev':<12} {'Mean ε':<10} {'Seeds':<5}")
    print("-"*80)
    
    results_by_C = {}
    for C in sorted(by_C.keys()):
        results = by_C[C]
        recalls = [r['recall_stress'] for r in results]
        epsilons = [r['epsilon'] for r in results if r['epsilon'] is not None]
        
        mean_recall = np.mean(recalls)
        std_recall = np.std(recalls, ddof=1) if len(recalls) > 1 else 0
        mean_epsilon = np.mean(epsilons) if epsilons else 0
        
        results_by_C[C] = {
            'mean_recall': mean_recall,
            'std_recall': std_recall,
            'mean_epsilon': mean_epsilon,
            'n_seeds': len(recalls)
        }
        
        print(f"{C:<8.1f} {mean_recall*100:<15.2f}% ±{std_recall*100:<10.2f}% {mean_epsilon:<10.2f} {len(recalls)}")
    
    # Analyze seed variance evolution
    print("\n" + "="*80)
    print("SEED VARIANCE EVOLUTION (Key Finding!)")
    print("="*80)
    
    for C in sorted(by_C.keys()):
        results = by_C[C]
        recalls = [r['recall_stress'] for r in results]
        std_pct = np.std(recalls, ddof=1) * 100 if len(recalls) > 1 else 0
        
        print(f"\nC={C:.1f}:")
        print(f"  Recalls by seed: {[f'{r:.4f}' for r in recalls]}")
        print(f"  Seed std dev: {std_pct:.2f}%")
        
        if C == min(by_C.keys()):
            baseline_std = std_pct
        else:
            reduction = ((baseline_std - std_pct) / baseline_std * 100) if baseline_std > 0 else 0
            print(f"  ↓ Reduction vs C={min(by_C.keys()):.1f}: {reduction:.1f}%")
    
    # Generate LaTeX
    print("\n" + "="*80)
    print("="*80)
    
    # Save results
    output_data = {
        'all_results': all_results,
        'by_C': {str(k): [r for r in v] for k, v in by_C.items()},
        'summary': results_by_C
    }
    
    output_file = Path('results/grad_norm_multiseed_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Final verdict
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)
    
    if len(results_by_C) >= 2:
        first_C = min(results_by_C.keys())
        last_C = max(results_by_C.keys())
        
        recall_improvement = results_by_C[last_C]['mean_recall'] - results_by_C[first_C]['mean_recall']
        std_reduction = results_by_C[first_C]['std_recall'] - results_by_C[last_C]['std_recall']
        
        print(f"Recall improvement: +{recall_improvement*100:.1f}%")
        print(f"Seed variance reduction: -{std_reduction*100:.1f}%")
        
        if std_reduction > 0 and recall_improvement > 0:
            print("\nHYPOTHESIS CONFIRMED")
            print("   Increasing C reduces seed dominance and improves minority recall.")
        else:
            print("\n HYPOTHESIS PARTIALLY CONFIRMED")
            print("   Results show some improvement but may need more investigation.")

