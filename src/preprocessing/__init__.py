"""
Preprocessing module for Sleep-EDF and WESAD datasets.

This module handles:
- Data loading from raw files
- Signal filtering and segmentation
- Feature extraction (time and frequency domain)
- Normalization and train/val/test splitting

Includes both standard and highly optimized versions:
- Standard versions: Original implementations
- Optimized versions: 3-10x faster with parallel processing and vectorization
"""

# Standard versions (now with subject-wise split)
from .sleep_edf import preprocess_sleep_edf, load_processed_sleep_edf
from .wesad import preprocess_wesad_temporal, load_processed_wesad_temporal

__all__ = [
    "preprocess_sleep_edf",
    "load_processed_sleep_edf",
    "preprocess_wesad_temporal",
    "load_processed_wesad_temporal",
]

