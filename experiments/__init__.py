#!/usr/bin/env python3
"""
Experiment management and execution.

Modules:
- config.py: Experiment configuration
- run_single_experiment.py: Run single experiment
- run_all_experiments.py: Run all experiments
- logging_setup.py: Experiment logging
- results_analyzer.py: Analyze experiment results

Features:
- Reproducible experiment execution
- Result tracking and logging
- Automatic analysis
- Report generation

Usage:
    from experiments import run_all_experiments
    
    results = run_all_experiments(
        datasets=['sleep-edf', 'wesad'],
        scenarios=['baseline', 'dp', 'fl', 'fl_dp'],
        n_runs=3,
        save_results=True
    )
"""

__all__ = []