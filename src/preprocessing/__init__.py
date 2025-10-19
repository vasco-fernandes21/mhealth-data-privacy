#!/usr/bin/env python3
"""
Data preprocessing modules for mHealth datasets.

Datasets:
- Sleep-EDF: Sleep stage classification (5 classes)
- WESAD: Stress detection (binary classification)

Each module provides:
- load_*: Load raw data
- preprocess_*: Preprocess and create train/val/test splits
- load_processed_*: Load preprocessed data

Features:
- Subject-wise splitting (for FL)
- Windowing for temporal models
- Augmentation for WESAD
- Data quality reports
- Per-channel normalization

Usage:
    from src.preprocessing.sleep_edf import preprocess_sleep_edf, load_windowed_sleep_edf
    from src.preprocessing.wesad import preprocess_wesad_temporal, load_augmented_wesad_temporal
    
    # Preprocess
    info = preprocess_sleep_edf(
        data_dir='data/raw/sleep-edf',
        output_dir='data/processed/sleep-edf'
    )
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = \\
        load_windowed_sleep_edf('data/processed/sleep-edf')
"""

# Don't import here to avoid issues when modules aren't needed
# Users should import directly:
#   from src.preprocessing.sleep_edf import ...
#   from src.preprocessing.wesad import ...

__all__ = [
    'sleep_edf',
    'wesad',
    'quality_report',
    'common',
]