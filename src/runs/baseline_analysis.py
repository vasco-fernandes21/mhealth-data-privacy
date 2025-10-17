"""
Baseline analysis functions for multiple training runs.

This module provides functions to analyze results from multiple baseline training runs:
- Load results from multiple runs with different seeds
- Generate plots showing performance across runs
- Calculate statistics (mean, std) across runs
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def load_baseline_runs(base_results_dir: str, dataset: str = "sleep-edf") -> List[Dict]:
    """
    Load results from multiple baseline runs.

    Args:
        base_results_dir: Base directory containing run subdirectories
        dataset: Dataset name ("sleep-edf" or "wesad")

    Returns:
        List of run results dictionaries
    """
    results = []

    # Look for run directories (baseline_run1, baseline_run2, etc.)
    for item in os.listdir(base_results_dir):
        run_dir = os.path.join(base_results_dir, item)
        if not os.path.isdir(run_dir) or not item.startswith("baseline_run"):
            continue

        # Extract run number
        try:
            run_num = int(item.replace("baseline_run", ""))
        except ValueError:
            continue

        # Find results file
        results_file = os.path.join(run_dir, f"results_{dataset}_optimized.json")
        if not os.path.exists(results_file):
            results_file = os.path.join(run_dir, f"results_{dataset}_baseline.json")
            if not os.path.exists(results_file):
                continue

        # Load results
        try:
            with open(results_file, 'r') as f:
                run_results = json.load(f)

            results.append({
                'run': run_num,
                'seed': run_results.get('seed', 'unknown'),
                'accuracy': run_results['accuracy'],
                'precision': run_results['precision'],
                'recall': run_results['recall'],
                'f1_score': run_results['f1_score'],
                'timestamp': run_results.get('timestamp', datetime.now().isoformat())
            })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load results from {results_file}: {e}")
            continue

    # Sort by run number
    results.sort(key=lambda x: x['run'])
    return results

def calculate_baseline_stats(results: List[Dict]) -> Dict:
    """
    Calculate statistics across multiple baseline runs.

    Args:
        results: List of run results

    Returns:
        Dictionary with mean and std for each metric
    """
    if not results:
        return {}

    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    return {
        'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
        'f1_score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
        'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
        'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
        'num_runs': len(results)
    }

def plot_baseline_runs(results: List[Dict], dataset: str = "sleep-edf",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot accuracy and F1-score across baseline runs.

    Args:
        results: List of run results
        dataset: Dataset name for plot title
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    if not results:
        print("No results to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    runs = list(range(1, len(results) + 1))
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]

    stats = calculate_baseline_stats(results)

    # Accuracy plot
    axes[0].plot(runs, accuracies, marker='o', linewidth=2, markersize=8, label='Run Accuracy')
    axes[0].axhline(stats['accuracy']['mean'], color='red', linestyle='--',
                   label=f"Mean: {stats['accuracy']['mean']:.4f}")
    axes[0].fill_between(runs,
                        stats['accuracy']['mean'] - stats['accuracy']['std'],
                        stats['accuracy']['mean'] + stats['accuracy']['std'],
                        alpha=0.2, color='red',
                        label=f"±1 std: {stats['accuracy']['std']:.4f}")
    axes[0].set_xlabel('Run')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Accuracy per Run - {dataset.upper()} Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(runs)

    # F1-Score plot
    axes[1].plot(runs, f1_scores, marker='s', linewidth=2, markersize=8,
                label='Run F1-Score', color='green')
    axes[1].axhline(stats['f1_score']['mean'], color='darkgreen', linestyle='--',
                   label=f"Mean: {stats['f1_score']['mean']:.4f}")
    axes[1].fill_between(runs,
                        stats['f1_score']['mean'] - stats['f1_score']['std'],
                        stats['f1_score']['mean'] + stats['f1_score']['std'],
                        alpha=0.2, color='green',
                        label=f"±1 std: {stats['f1_score']['std']:.4f}")
    axes[1].set_xlabel('Run')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title(f'F1-Score per Run - {dataset.upper()} Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(runs)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def analyze_baseline_experiment(base_results_dir: str, dataset: str = "sleep-edf",
                               save_plot: bool = True) -> Dict:
    """
    Complete analysis of baseline experiment with multiple runs.

    Args:
        base_results_dir: Base directory containing run results
        dataset: Dataset name
        save_plot: Whether to save the plot

    Returns:
        Dictionary with results and statistics
    """
    print(f"🔍 Analyzing baseline experiment for {dataset.upper()}")
    print("=" * 60)

    # Load results
    results = load_baseline_runs(base_results_dir, dataset)

    if not results:
        print(f"❌ No baseline results found in {base_results_dir}")
        return {}

    print(f"✅ Found {len(results)} baseline runs")

    # Calculate statistics
    stats = calculate_baseline_stats(results)

    # Print summary table
    print(f"\n📊 BASELINE MULTIPLE RUNS SUMMARY - {dataset.upper()}")
    print("=" * 60)
    print(f"Number of runs: {stats['num_runs']}\n")
    print("Metric        Mean ± Std")
    print("-" * 30)
    for metric, values in stats.items():
        if metric != 'num_runs':
            print(f"{metric:12s}  {values['mean']:.4f} ± {values['std']:.4f}")
    print("=" * 60)

    # Generate plot
    if save_plot:
        plot_path = os.path.join(base_results_dir, f'baseline_{dataset}_5runs.png')
        plot_baseline_runs(results, dataset, plot_path)

    # Save summary
    summary_path = os.path.join(base_results_dir, f'baseline_{dataset}_5runs.json')
    summary = {
        'dataset': dataset,
        'experiment': 'baseline',
        'timestamp': datetime.now().isoformat(),
        'runs': results,
        'statistics': stats
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {summary_path}")

    return summary
