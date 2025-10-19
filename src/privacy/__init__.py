#!/usr/bin/env python3
"""
Evaluation and analysis modules.

Modules (Tier 2+):
- metrics: Classification metrics (accuracy, F1, etc)
- privacy_utility_analysis: Privacy-utility tradeoff analysis
- report_generator: Automated report generation

Features:
- Standard ML metrics
- Privacy metrics (epsilon bounds, etc)
- Plotting utilities
- Comparative analysis
- Paper figure generation

Usage (Tier 2+):
    from src.evaluation import metrics, privacy_utility_analysis
    
    # Evaluate
    results = metrics.evaluate_classification(y_true, y_pred)
    
    # Analyze privacy-utility tradeoff
    analysis = privacy_utility_analysis.analyze(
        scenarios=['baseline', 'dp', 'fl', 'fl_dp'],
        results_dir='./results'
    )
    
    # Generate plots
    privacy_utility_analysis.plot_tradeoff(analysis, output_dir='./paper/figures')
"""

__all__ = [
    'metrics',
    'privacy_utility_analysis',
    'report_generator',
]