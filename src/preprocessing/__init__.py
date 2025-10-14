"""
Preprocessing module for Sleep-EDF and WESAD datasets.

This module handles:
- Data loading from raw files
- Signal filtering and segmentation
- Feature extraction (time and frequency domain)
- Normalization and train/val/test splitting
"""

from .sleep_edf import preprocess_sleep_edf, extract_sleep_features
from .wesad import preprocess_wesad_temporal, load_processed_wesad_temporal

__all__ = [
    "preprocess_sleep_edf",
    "extract_sleep_features",
    "preprocess_wesad_temporal",
    "load_processed_wesad_temporal",
]

