"""
Runs analysis module for mHealth data privacy project.

This module provides functions to analyze results from multiple training runs:
- Baseline analysis: Statistics and plots for multiple baseline runs
- DP analysis: Privacy-utility tradeoff analysis for different noise multipliers
- FL analysis: Performance analysis across different client numbers
"""

from .baseline_analysis import (
    load_baseline_runs,
    plot_baseline_runs,
    calculate_baseline_stats,
    analyze_baseline_experiment
)

from .dp_analysis import (
    load_dp_runs,
    plot_privacy_utility_tradeoff,
    analyze_dp_experiment
)

from .fl_analysis import (
    load_fl_runs,
    plot_fl_performance,
    analyze_fl_experiment
)

__all__ = [
    # Baseline analysis
    'load_baseline_runs',
    'plot_baseline_runs',
    'calculate_baseline_stats',
    'analyze_baseline_experiment',

    # DP analysis
    'load_dp_runs',
    'plot_privacy_utility_tradeoff',
    'analyze_dp_experiment',

    # FL analysis
    'load_fl_runs',
    'plot_fl_performance',
    'analyze_fl_experiment'
]
