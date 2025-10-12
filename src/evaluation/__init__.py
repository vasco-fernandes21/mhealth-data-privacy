"""
Evaluation module for metrics computation and visualization.

This module provides:
- Model evaluation metrics (accuracy, precision, recall, F1)
- Visualization functions (trade-off curves, confusion matrices)
- Statistical analysis tools
"""

from .metrics import evaluate_model, compute_metrics, compare_models
from .visualization import (
    plot_tradeoff_curve,
    plot_comparison_bars,
    plot_confusion_matrix,
    plot_training_history,
)

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "compare_models",
    "plot_tradeoff_curve",
    "plot_comparison_bars",
    "plot_confusion_matrix",
    "plot_training_history",
]

