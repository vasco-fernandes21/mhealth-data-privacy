"""
Federated Learning analysis functions for multiple client configurations.

This module provides functions to analyze results from FL training runs:
- Load results from runs with different numbers of clients
- Compare performance across client configurations
- Analyze stability and variability
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def load_fl_runs(base_results_dir: str, dataset: str = "sleep-edf") -> List[Dict]:
    """
    Load results from multiple FL runs with different client numbers.

    Args:
        base_results_dir: Base directory containing run subdirectories
        dataset: Dataset name ("sleep-edf" or "wesad")

    Returns:
        List of FL run results dictionaries
    """
    results = []

    # Look for run directories (fl_clients2, fl_clients5, fl_clients10, etc.)
    for item in os.listdir(base_results_dir):
        run_dir = os.path.join(base_results_dir, item)
        if not os.path.isdir(run_dir) or not item.startswith("fl_clients"):
            continue

        # Extract number of clients
        try:
            num_clients = int(item.replace("fl_clients", ""))
        except ValueError:
            continue

        # Find results file
        results_file = os.path.join(run_dir, f"results_{dataset}_fl.json")
        if not os.path.exists(results_file):
            results_file = os.path.join(run_dir, "results_fl.json")
            if not os.path.exists(results_file):
                continue

        # Load results
        try:
            with open(results_file, 'r') as f:
                run_results = json.load(f)

            results.append({
                'num_clients': num_clients,
                'accuracy': run_results['accuracy'],
                'precision': run_results['precision'],
                'recall': run_results['recall'],
                'f1_score': run_results['f1_score'],
                'training_time': run_results.get('training_time', 0),
                'rounds': run_results.get('rounds', 0),
                'timestamp': run_results.get('timestamp', datetime.now().isoformat())
            })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load results from {results_file}: {e}")
            continue

    # Sort by number of clients
    results.sort(key=lambda x: x['num_clients'])
    return results

def plot_fl_performance(results: List[Dict], dataset: str = "sleep-edf",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot FL performance across different client numbers.

    Args:
        results: List of FL run results
        dataset: Dataset name for plot title
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    if not results:
        print("No results to plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    clients = [r['num_clients'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    # Accuracy vs Clients
    axes[0,0].plot(clients, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0,0].set_xlabel('Number of Clients')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title(f'Accuracy vs Clients - {dataset.upper()} FL')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(clients)

    # F1-Score vs Clients
    axes[0,1].plot(clients, f1_scores, marker='s', linewidth=2, markersize=8, color='green')
    axes[0,1].set_xlabel('Number of Clients')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].set_title(f'F1-Score vs Clients - {dataset.upper()} FL')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xticks(clients)

    # Precision vs Clients
    axes[1,0].plot(clients, precisions, marker='^', linewidth=2, markersize=8, color='orange')
    axes[1,0].set_xlabel('Number of Clients')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title(f'Precision vs Clients - {dataset.upper()} FL')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(clients)

    # Recall vs Clients
    axes[1,1].plot(clients, recalls, marker='v', linewidth=2, markersize=8, color='red')
    axes[1,1].set_xlabel('Number of Clients')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].set_title(f'Recall vs Clients - {dataset.upper()} FL')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xticks(clients)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def analyze_fl_experiment(base_results_dir: str, dataset: str = "sleep-edf",
                         save_plot: bool = True) -> Dict:
    """
    Complete analysis of FL experiment with different client numbers.

    Args:
        base_results_dir: Base directory containing FL run results
        dataset: Dataset name
        save_plot: Whether to save the plot

    Returns:
        Dictionary with results and statistics
    """
    print(f"🌐 Analyzing FL experiment for {dataset.upper()}")
    print("=" * 60)

    # Load results
    results = load_fl_runs(base_results_dir, dataset)

    if not results:
        print(f"❌ No FL results found in {base_results_dir}")
        return {}

    print(f"✅ Found {len(results)} FL runs")

    # Calculate statistics
    clients = [r['num_clients'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]

    stats = {
        'client_range': {'min': min(clients), 'max': max(clients)},
        'best_accuracy': max(accuracies),
        'best_f1': max(f1_scores),
        'num_runs': len(results)
    }

    # Print summary table
    print(f"\n🌍 FEDERATED LEARNING MULTIPLE RUNS SUMMARY - {dataset.upper()}")
    print("=" * 70)
    print(f"Number of runs: {stats['num_runs']}\n")
    print("Clients  Accuracy  F1-Score  Precision  Recall")
    print("-" * 50)
    for r in results:
        print(f"{r['num_clients']:4d}     {r['accuracy']:.4f}    {r['f1_score']:.4f}    {r['precision']:.4f}   {r['recall']:.4f}")
    print(f"\n📊 Best Performance:")
    print(f"  Highest Accuracy: {stats['best_accuracy']:.4f}")
    print(f"  Highest F1-Score: {stats['best_f1']:.4f}")
    print("=" * 70)

    # Generate performance plot
    if save_plot:
        plot_path = os.path.join(base_results_dir, f'fl_{dataset}_performance.png')
        plot_fl_performance(results, dataset, plot_path)

    # Save summary
    summary_path = os.path.join(base_results_dir, f'fl_{dataset}_summary.json')
    summary = {
        'dataset': dataset,
        'experiment': 'federated_learning',
        'timestamp': datetime.now().isoformat(),
        'runs': results,
        'statistics': stats
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {summary_path}")

    return summary
