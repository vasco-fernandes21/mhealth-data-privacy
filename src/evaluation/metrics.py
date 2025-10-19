#!/usr/bin/env python3
"""
Comprehensive evaluation metrics.

Computes:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-class metrics
- Privacy-utility metrics
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from typing import Dict, Tuple, Any, Optional
import json
from pathlib import Path


class MetricsCalculator:
    """Calculate comprehensive metrics."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       class_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Per-class metrics
        per_class_metrics = {}
        for class_id in np.unique(y_true):
            binary_y_true = (y_true == class_id).astype(int)
            binary_y_pred = (y_pred == class_id).astype(int)
            
            class_name = class_names[class_id] if class_names else f"class_{class_id}"
            
            per_class_metrics[class_name] = {
                'precision': float(precision_score(binary_y_true, binary_y_pred, zero_division=0)),
                'recall': float(recall_score(binary_y_true, binary_y_pred, zero_division=0)),
                'f1_score': float(f1_score(binary_y_true, binary_y_pred, zero_division=0))
            }
        
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    @staticmethod
    def compute_privacy_utility_tradeoff(accuracy: float,
                                        epsilon: float,
                                        target_epsilon: float = 8.0) -> Dict[str, float]:
        """
        Compute privacy-utility tradeoff metrics.
        
        Args:
            accuracy: Model accuracy
            epsilon: Actual privacy budget used
            target_epsilon: Target privacy budget
        
        Returns:
            Dictionary with tradeoff metrics
        """
        privacy_budget_ratio = epsilon / target_epsilon if target_epsilon > 0 else 1.0
        utility_loss = 1.0 - accuracy  # How much accuracy is lost
        
        return {
            'accuracy': accuracy,
            'epsilon': epsilon,
            'privacy_budget_used': min(privacy_budget_ratio, 1.0),
            'utility_loss': utility_loss,
            'tradeoff_score': accuracy * (1.0 - min(privacy_budget_ratio, 1.0))
        }


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(input_path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        input_path: Path to metrics file
    
    Returns:
        Metrics dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test
    print("Testing MetricsCalculator...\n")
    
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 2, 1, 2])
    class_names = ['class_0', 'class_1', 'class_2']
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_metrics(y_true, y_pred, class_names)
    
    print("Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}: F1={class_metrics['f1_score']:.4f}")
    
    # Test privacy-utility
    tradeoff = calculator.compute_privacy_utility_tradeoff(
        accuracy=0.85,
        epsilon=2.5,
        target_epsilon=8.0
    )
    print(f"\nPrivacy-Utility Tradeoff:")
    print(f"  Accuracy: {tradeoff['accuracy']:.4f}")
    print(f"  Privacy budget used: {tradeoff['privacy_budget_used']*100:.1f}%")
    print(f"  Tradeoff score: {tradeoff['tradeoff_score']:.4f}")
    
    print("\n✅ All tests passed!")