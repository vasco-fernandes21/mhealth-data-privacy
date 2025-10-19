#!/usr/bin/env python3
"""
Command-line scripts for the mHealth Privacy package.

Scripts:
- preprocess_all.py: Preprocess all datasets
- train_baseline.py: Train baseline models
- train_dp.py: Train with Differential Privacy
- train_fl.py: Train with Federated Learning
- train_fl_dp.py: Train with FL+DP
- train_all_scenarios.py: Run all scenarios
- sweep_hyperparams.py: Hyperparameter tuning
- generate_report.py: Generate analysis reports

Usage:
    python scripts/preprocess_all.py --data_dir ./data
    python scripts/train_baseline.py --dataset sleep-edf
    python scripts/generate_report.py --results_dir ./results
"""

__all__ = []