"""
Evaluation Metrics Module

This module provides functions for computing and comparing evaluation metrics
across different models and privacy techniques.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple, Any
import json
import os
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multiclass
                metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
    
    return metrics


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_pred_proba: np.ndarray = None, class_names: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names
    }
    
    return results


def compare_models(results_dict: Dict[str, Dict[str, Any]], 
                  metric: str = 'f1_weighted') -> pd.DataFrame:
    """
    Compare multiple models based on specified metric.
    
    Args:
        results_dict: Dictionary with model results
        metric: Metric to use for comparison
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results and metric in results['metrics']:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['metrics'].get('accuracy', 0),
                'Precision': results['metrics'].get('precision_weighted', 0),
                'Recall': results['metrics'].get('recall_weighted', 0),
                'F1-Score': results['metrics'].get('f1_weighted', 0),
                'AUC': results['metrics'].get('auc_macro', results['metrics'].get('auc', 0))
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(metric.replace('_weighted', '').replace('_macro', '').title(), ascending=False)
    
    return df


def compute_privacy_metrics(baseline_results: Dict[str, Any], 
                           privacy_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute privacy-related metrics (degradation, etc.).
    
    Args:
        baseline_results: Results from baseline model
        privacy_results: Results from privacy-preserving model
    
    Returns:
        Dictionary with privacy metrics
    """
    baseline_accuracy = baseline_results['metrics']['accuracy']
    privacy_accuracy = privacy_results['metrics']['accuracy']
    
    baseline_f1 = baseline_results['metrics']['f1_weighted']
    privacy_f1 = privacy_results['metrics']['f1_weighted']
    
    privacy_metrics = {
        'accuracy_degradation': baseline_accuracy - privacy_accuracy,
        'accuracy_degradation_percent': ((baseline_accuracy - privacy_accuracy) / baseline_accuracy) * 100,
        'f1_degradation': baseline_f1 - privacy_f1,
        'f1_degradation_percent': ((baseline_f1 - privacy_f1) / baseline_f1) * 100,
        'accuracy_retention': privacy_accuracy / baseline_accuracy,
        'f1_retention': privacy_f1 / baseline_f1
    }
    
    return privacy_metrics


def compute_fl_metrics(fl_history: Dict[str, List], baseline_accuracy: float) -> Dict[str, Any]:
    """
    Compute Federated Learning specific metrics.
    
    Args:
        fl_history: FL training history
        baseline_accuracy: Baseline model accuracy
    
    Returns:
        Dictionary with FL metrics
    """
    if 'accuracy' not in fl_history:
        return {'error': 'No accuracy history found'}
    
    accuracies = fl_history['accuracy']
    
    # Convergence metrics
    final_accuracy = accuracies[-1]
    max_accuracy = max(accuracies)
    
    # Find convergence point (accuracy within 1% of final)
    convergence_threshold = final_accuracy * 0.99
    convergence_round = None
    for i, acc in enumerate(accuracies):
        if acc >= convergence_threshold:
            convergence_round = i
            break
    
    # Communication efficiency
    total_rounds = len(accuracies)
    
    fl_metrics = {
        'final_accuracy': final_accuracy,
        'max_accuracy': max_accuracy,
        'convergence_round': convergence_round,
        'total_rounds': total_rounds,
        'accuracy_vs_baseline': final_accuracy - baseline_accuracy,
        'accuracy_retention': final_accuracy / baseline_accuracy,
        'convergence_efficiency': convergence_round / total_rounds if convergence_round else 1.0
    }
    
    return fl_metrics


def create_results_summary(all_results: Dict[str, Dict[str, Any]], 
                          save_path: str = None) -> pd.DataFrame:
    """
    Create a comprehensive results summary.
    
    Args:
        all_results: Dictionary with all model results
        save_path: Path to save summary (optional)
    
    Returns:
        DataFrame with results summary
    """
    summary_data = []
    
    for model_name, results in all_results.items():
        if 'metrics' in results:
            metrics = results['metrics']
            summary_data.append({
                'Model': model_name,
                'Dataset': results.get('dataset', 'Unknown'),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision_weighted', 0),
                'Recall': metrics.get('recall_weighted', 0),
                'F1-Score': metrics.get('f1_weighted', 0),
                'AUC': metrics.get('auc_macro', metrics.get('auc', 0)),
                'Privacy_Technique': results.get('privacy_technique', 'None'),
                'Privacy_Parameter': results.get('privacy_parameter', 'N/A')
            })
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Results summary saved: {save_path}")
    
    return df


def save_evaluation_results(results: Dict[str, Any], save_path: str, 
                           filename: str = 'evaluation_results.json'):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Directory to save results
        filename: Name for the results file
    """
    os.makedirs(save_path, exist_ok=True)
    
    results_file = os.path.join(save_path, filename)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved: {results_file}")


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results file
    
    Returns:
        Loaded results dictionary
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_statistical_significance(results1: Dict[str, Any], results2: Dict[str, Any],
                                   metric: str = 'accuracy', n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    Compute statistical significance between two models using bootstrap.
    
    Args:
        results1: Results from first model
        results2: Results from second model
        metric: Metric to compare
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    # This is a simplified version - in practice, you'd need the actual predictions
    # for proper statistical testing
    metric1 = results1['metrics'].get(metric, 0)
    metric2 = results2['metrics'].get(metric, 0)
    
    # For now, just return the difference
    # In a real implementation, you'd bootstrap the predictions
    difference = metric1 - metric2
    
    return {
        'metric': metric,
        'model1_score': metric1,
        'model2_score': metric2,
        'difference': difference,
        'relative_difference': difference / metric1 if metric1 != 0 else 0
    }


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("This module provides functions for computing and comparing evaluation metrics.")
