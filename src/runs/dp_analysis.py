"""
Differential Privacy analysis functions for multiple training runs.

This module provides functions to analyze results from DP training runs:
- Load results from runs with different noise multipliers
- Plot privacy-utility tradeoff (epsilon vs accuracy/F1)
- Analyze DP parameter effects
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def load_dp_runs(base_results_dir: str, dataset: str = "sleep-edf") -> List[Dict]:
    """
    Load results from multiple DP runs with different noise multipliers.

    Args:
        base_results_dir: Base directory containing run subdirectories
        dataset: Dataset name ("sleep-edf" or "wesad")

    Returns:
        List of DP run results dictionaries
    """
    results = []

    # Look for run directories (dp_run1_noise0.5, dp_run2_noise0.7, etc.)
    for item in os.listdir(base_results_dir):
        run_dir = os.path.join(base_results_dir, item)
        if not os.path.isdir(run_dir) or not item.startswith("dp_run"):
            continue

        # Extract run number and noise multiplier
        try:
            parts = item.replace("dp_run", "").split("_noise")
            run_num = int(parts[0])
            noise_mult = float(parts[1])
        except (ValueError, IndexError):
            continue

        # Find results file
        results_file = os.path.join(run_dir, f"results_{dataset}_dp.json")
        if not os.path.exists(results_file):
            continue

        # Load results
        try:
            with open(results_file, 'r') as f:
                run_results = json.load(f)

            # Extract epsilon from dp_params or directly
            epsilon = run_results.get('dp_params', {}).get('final_epsilon', 0.0)
            if epsilon == 0.0 and 'epsilon' in run_results:
                epsilon = run_results['epsilon']

            results.append({
                'run': run_num,
                'noise_multiplier': noise_mult,
                'epsilon': epsilon,
                'accuracy': run_results['accuracy'],
                'precision': run_results['precision'],
                'recall': run_results['recall'],
                'f1_score': run_results['f1_score'],
                'timestamp': run_results.get('timestamp', datetime.now().isoformat())
            })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load results from {results_file}: {e}")
            continue

    # Sort by noise multiplier
    results.sort(key=lambda x: x['noise_multiplier'])
    return results

def plot_privacy_utility_tradeoff(results: List[Dict], dataset: str = "sleep-edf",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot privacy-utility tradeoff: epsilon vs accuracy and F1-score.

    Args:
        results: List of DP run results
        dataset: Dataset name for plot title
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    if not results:
        print("No results to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epsilons = [r['epsilon'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    noise_mults = [r['noise_multiplier'] for r in results]

    # Epsilon vs Accuracy
    axes[0].plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8,
                label='DP Accuracy')
    for i, (eps, acc, nm) in enumerate(zip(epsilons, accuracies, noise_mults)):
        axes[0].annotate(f'σ={nm}', (eps, acc), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=8)
    axes[0].set_xlabel('Privacy Budget (ε)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Privacy-Utility Trade-off: Accuracy - {dataset.upper()}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Epsilon vs F1-Score
    axes[1].plot(epsilons, f1_scores, marker='s', linewidth=2, markersize=8,
                label='DP F1-Score', color='green')
    for i, (eps, f1, nm) in enumerate(zip(epsilons, f1_scores, noise_mults)):
        axes[1].annotate(f'σ={nm}', (eps, f1), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=8)
    axes[1].set_xlabel('Privacy Budget (ε)')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title(f'Privacy-Utility Trade-off: F1-Score - {dataset.upper()}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def analyze_dp_experiment(base_results_dir: str, dataset: str = "sleep-edf",
                         save_plot: bool = True) -> Dict:
    """
    Complete analysis of DP experiment with different noise multipliers.

    Args:
        base_results_dir: Base directory containing DP run results
        dataset: Dataset name
        save_plot: Whether to save the plot

    Returns:
        Dictionary with results and statistics
    """
    print(f"🔒 Analyzing DP experiment for {dataset.upper()}")
    print("=" * 60)

    # Load results
    results = load_dp_runs(base_results_dir, dataset)

    if not results:
        print(f"❌ No DP results found in {base_results_dir}")
        return {}

    print(f"✅ Found {len(results)} DP runs")

    # Calculate ranges
    epsilons = [r['epsilon'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]

    stats = {
        'epsilon_range': {'min': min(epsilons), 'max': max(epsilons)},
        'accuracy_range': {'min': min(accuracies), 'max': max(accuracies)},
        'f1_range': {'min': min(f1_scores), 'max': max(f1_scores)},
        'num_runs': len(results)
    }

    # Print summary table
    print(f"\n🔐 DIFFERENTIAL PRIVACY MULTIPLE RUNS SUMMARY - {dataset.upper()}")
    print("=" * 70)
    print(f"Number of runs: {stats['num_runs']}\n")
    print("Run  Noise σ   ε      Accuracy  F1-Score  Precision  Recall")
    print("-" * 70)
    for r in results:
        print(f"{r['run']:3d}  {r['noise_multiplier']:5.1f}  {r['epsilon']:6.2f}  "
              f"{r['accuracy']:.4f}    {r['f1_score']:.4f}    {r['precision']:.4f}     {r['recall']:.4f}")
    print(f"\n📈 Metric Ranges:")
    print("-" * 35)
    print(f"Privacy Budget (ε):  [{stats['epsilon_range']['min']:.2f}, {stats['epsilon_range']['max']:.2f}]")
    print(f"Accuracy:            [{stats['accuracy_range']['min']:.4f}, {stats['accuracy_range']['max']:.4f}]")
    print(f"F1-Score:            [{stats['f1_range']['min']:.4f}, {stats['f1_range']['max']:.4f}]")
    print("=" * 70)

    # Generate privacy-utility tradeoff plot
    if save_plot:
        plot_path = os.path.join(base_results_dir, f'dp_{dataset}_tradeoff.png')
        plot_privacy_utility_tradeoff(results, dataset, plot_path)

    # Save summary
    summary_path = os.path.join(base_results_dir, f'dp_{dataset}_5runs.json')
    summary = {
        'dataset': dataset,
        'experiment': 'differential_privacy',
        'timestamp': datetime.now().isoformat(),
        'runs': results,
        'statistics': stats
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {summary_path}")

    return summary
