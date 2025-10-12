"""
Visualization Module

This module provides functions for creating visualizations of results,
including trade-off curves, comparison plots, and confusion matrices.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_tradeoff_curve(results_dict: Dict[str, Dict[str, Any]], 
                       metric: str = 'accuracy', 
                       privacy_param: str = 'epsilon',
                       save_path: str = None,
                       title: str = "Privacy vs. Performance Trade-off") -> plt.Figure:
    """
    Plot trade-off curve between privacy and performance.
    
    Args:
        results_dict: Dictionary with model results
        metric: Performance metric to plot
        privacy_param: Privacy parameter name
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    privacy_values = []
    performance_values = []
    model_names = []
    
    for model_name, results in results_dict.items():
        if privacy_param in results and metric in results.get('metrics', {}):
            privacy_values.append(results[privacy_param])
            performance_values.append(results['metrics'][metric])
            model_names.append(model_name)
    
    if not privacy_values:
        print("No data found for trade-off curve")
        return fig
    
    # Sort by privacy parameter
    sorted_data = sorted(zip(privacy_values, performance_values, model_names))
    privacy_values, performance_values, model_names = zip(*sorted_data)
    
    # Plot
    ax.plot(privacy_values, performance_values, 'o-', linewidth=2, markersize=8)
    
    # Add labels
    for i, name in enumerate(model_names):
        ax.annotate(name, (privacy_values[i], performance_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel(f'{privacy_param.title()} (Privacy Level)', fontsize=12)
    ax.set_ylabel(f'{metric.title()}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Invert x-axis if epsilon (lower = more private)
    if privacy_param == 'epsilon':
        ax.invert_xaxis()
        ax.set_xlabel(f'{privacy_param.title()} (Lower = More Private)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trade-off curve saved: {save_path}")
    
    return fig


def plot_comparison_bars(results_dict: Dict[str, Dict[str, Any]], 
                        metrics: List[str] = ['accuracy', 'f1_weighted'],
                        save_path: str = None,
                        title: str = "Model Comparison") -> plt.Figure:
    """
    Plot bar chart comparing models across multiple metrics.
    
    Args:
        results_dict: Dictionary with model results
        metrics: List of metrics to compare
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data
        model_names = []
        values = []
        
        for model_name, results in results_dict.items():
            if 'metrics' in results and metric in results['metrics']:
                model_names.append(model_name)
                values.append(results['metrics'][metric])
        
        if not values:
            continue
        
        # Create bar plot
        bars = ax.bar(model_names, values, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison bars saved: {save_path}")
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None,
                         title: str = "Confusion Matrix", 
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix with heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: Names of classes
        title: Plot title
        save_path: Path to save plot (optional)
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    return fig


def plot_training_history(history: Dict[str, List], 
                         save_path: str = None,
                         title: str = "Training History") -> plt.Figure:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved: {save_path}")
    
    return fig


def plot_fl_convergence(fl_history: Dict[str, List], 
                       baseline_accuracy: float = None,
                       save_path: str = None,
                       title: str = "Federated Learning Convergence") -> plt.Figure:
    """
    Plot FL convergence over rounds.
    
    Args:
        fl_history: FL training history
        baseline_accuracy: Baseline model accuracy (optional)
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'accuracy' in fl_history:
        rounds = range(1, len(fl_history['accuracy']) + 1)
        ax.plot(rounds, fl_history['accuracy'], 'o-', linewidth=2, markersize=4, label='FL Accuracy')
        
        # Add baseline if provided
        if baseline_accuracy is not None:
            ax.axhline(y=baseline_accuracy, color='red', linestyle='--', 
                      linewidth=2, label=f'Baseline ({baseline_accuracy:.3f})')
        
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"FL convergence plot saved: {save_path}")
    
    return fig


def plot_privacy_analysis(results_dict: Dict[str, Dict[str, Any]], 
                         save_path: str = None,
                         title: str = "Privacy Analysis") -> plt.Figure:
    """
    Plot comprehensive privacy analysis.
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract baseline results
    baseline_results = None
    for model_name, results in results_dict.items():
        if 'privacy_technique' in results and results['privacy_technique'] == 'None':
            baseline_results = results
            break
    
    if baseline_results is None:
        print("No baseline results found")
        return fig
    
    baseline_accuracy = baseline_results['metrics']['accuracy']
    baseline_f1 = baseline_results['metrics']['f1_weighted']
    
    # Plot 1: Accuracy degradation
    ax1 = axes[0, 0]
    model_names = []
    accuracy_degradations = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results and 'privacy_technique' in results:
            if results['privacy_technique'] != 'None':
                model_names.append(model_name)
                degradation = baseline_accuracy - results['metrics']['accuracy']
                accuracy_degradations.append(degradation)
    
    if accuracy_degradations:
        bars = ax1.bar(model_names, accuracy_degradations, alpha=0.7, color='red')
        ax1.set_ylabel('Accuracy Degradation', fontsize=12)
        ax1.set_title('Accuracy Degradation vs. Baseline', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, accuracy_degradations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: F1 degradation
    ax2 = axes[0, 1]
    f1_degradations = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results and 'privacy_technique' in results:
            if results['privacy_technique'] != 'None':
                degradation = baseline_f1 - results['metrics']['f1_weighted']
                f1_degradations.append(degradation)
    
    if f1_degradations:
        bars = ax2.bar(model_names, f1_degradations, alpha=0.7, color='orange')
        ax2.set_ylabel('F1-Score Degradation', fontsize=12)
        ax2.set_title('F1-Score Degradation vs. Baseline', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, f1_degradations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Privacy vs. Performance scatter
    ax3 = axes[1, 0]
    privacy_params = []
    accuracies = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results and 'privacy_technique' in results:
            if results['privacy_technique'] != 'None':
                if 'epsilon' in results:
                    privacy_params.append(results['epsilon'])
                    accuracies.append(results['metrics']['accuracy'])
    
    if privacy_params:
        ax3.scatter(privacy_params, accuracies, s=100, alpha=0.7)
        ax3.set_xlabel('Epsilon (Privacy Level)', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Privacy vs. Performance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()  # Lower epsilon = more private
    
    # Plot 4: Model comparison radar chart (simplified)
    ax4 = axes[1, 1]
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    model_data = {}
    
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            model_data[model_name] = [results['metrics'].get(metric, 0) for metric in metrics]
    
    if model_data:
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, values) in enumerate(model_data.items()):
            ax4.bar(x + i*width, values, width, label=model_name, alpha=0.7)
        
        ax4.set_xlabel('Metrics', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Privacy analysis saved: {save_path}")
    
    return fig


def create_summary_dashboard(results_dict: Dict[str, Dict[str, Any]], 
                           save_path: str = None) -> plt.Figure:
    """
    Create a comprehensive summary dashboard.
    
    Args:
        results_dict: Dictionary with all model results
        save_path: Path to save dashboard (optional)
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model comparison bars
    ax1 = fig.add_subplot(gs[0, :2])
    plot_comparison_bars(results_dict, ['accuracy', 'f1_weighted'], ax=ax1)
    
    # Plot 2: Privacy trade-off
    ax2 = fig.add_subplot(gs[0, 2:])
    plot_tradeoff_curve(results_dict, ax=ax2)
    
    # Plot 3: Training history (if available)
    ax3 = fig.add_subplot(gs[1, :2])
    # This would need training history data
    
    # Plot 4: Privacy analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    plot_privacy_analysis(results_dict, ax=ax4)
    
    # Plot 5: Summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    summary_data = []
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            summary_data.append([
                model_name,
                f"{results['metrics'].get('accuracy', 0):.3f}",
                f"{results['metrics'].get('f1_weighted', 0):.3f}",
                results.get('privacy_technique', 'None'),
                str(results.get('privacy_parameter', 'N/A'))
            ])
    
    table = ax5.table(cellText=summary_data,
                     colLabels=['Model', 'Accuracy', 'F1-Score', 'Privacy Technique', 'Parameter'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.suptitle('Privacy-Preserving Health Data Analysis - Summary Dashboard', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary dashboard saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Visualization Module")
    print("This module provides functions for creating result visualizations.")
