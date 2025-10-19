#!/usr/bin/env python3
"""
Common preprocessing utilities shared across datasets.

Provides:
- Data validation
- Normalization helpers
- Subject management
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import joblib

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate data arrays.
    
    Args:
        X: Feature array
        y: Label array
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check shapes
    if len(X) != len(y):
        errors.append(f"Sample mismatch: X has {len(X)} samples, y has {len(y)}")
    
    # Check NaN/Inf
    if np.isnan(X).any():
        n_nan = np.sum(np.isnan(X))
        errors.append(f"Found {n_nan} NaN values in X")
    
    if np.isinf(X).any():
        n_inf = np.sum(np.isinf(X))
        errors.append(f"Found {n_inf} Inf values in X")
    
    # Check labels
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        errors.append(f"Labels not sequential: {unique_labels}")
    
    # Check label balance
    label_counts = np.bincount(y.astype(int))
    if max(label_counts) / min(label_counts) > 10:
        logger.warning("High class imbalance detected")
    
    return len(errors) == 0, errors


def get_subject_splits(data_dir: str) -> Dict[str, List[str]]:
    """
    Get subject splits for federated learning.
    
    Args:
        data_dir: Preprocessed data directory
    
    Returns:
        Dictionary with train/val/test subjects
    """
    data_path = Path(data_dir)
    
    try:
        info = joblib.load(data_path / 'preprocessing_info.pkl')
        
        return {
            'train': info['subject_splits']['train_subjects'],
            'val': info['subject_splits']['val_subjects'],
            'test': info['subject_splits']['test_subjects']
        }
    except Exception as e:
        logger.warning(f"Could not load subject splits: {e}")
        return None


def get_subject_data(data_dir: str, subject_id: str, split: str = 'train'):
    """
    Get all samples for a specific subject.
    
    Args:
        data_dir: Preprocessed data directory
        subject_id: Subject identifier
        split: 'train', 'val', or 'test'
    
    Returns:
        (X, y) for that subject
    """
    data_path = Path(data_dir)
    
    # Load full split
    X = np.load(data_path / f'X_{split}.npy')
    y = np.load(data_path / f'y_{split}.npy')
    
    # Load subject IDs if available
    try:
        subjects = np.load(data_path / f'subjects_{split}.npy', allow_pickle=True)
        
        # Filter by subject
        mask = subjects == subject_id
        return X[mask], y[mask]
    except FileNotFoundError:
        logger.warning("Subject IDs not available")
        return X, y


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Label array
    
    Returns:
        Class weights
    """
    unique, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    weights = total / (len(unique) * counts)
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return weights


def check_data_leakage(train_subjects: List[str],
                      val_subjects: List[str],
                      test_subjects: List[str]) -> bool:
    """
    Check for subject overlap between splits (data leakage).
    
    Args:
        train_subjects: Training subjects
        val_subjects: Validation subjects
        test_subjects: Test subjects
    
    Returns:
        True if no leakage, False otherwise
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    
    leakage = False
    
    if train_set & val_set:
        logger.warning(f"Train-Val overlap: {train_set & val_set}")
        leakage = True
    
    if train_set & test_set:
        logger.warning(f"Train-Test overlap: {train_set & test_set}")
        leakage = True
    
    if val_set & test_set:
        logger.warning(f"Val-Test overlap: {val_set & test_set}")
        leakage = True
    
    if not leakage:
        logger.info("✅ No data leakage detected")
    
    return not leakage


if __name__ == "__main__":
    print("Common preprocessing utilities loaded")