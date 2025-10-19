#!/usr/bin/env python3
"""
Data Quality Report Generation.

Analyzes preprocessed data and generates quality metrics.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any
import json

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class DataQualityAnalyzer:
    """Analyze data quality."""
    
    @staticmethod
    def analyze_dataset(X: np.ndarray, y: np.ndarray, 
                       dataset_name: str) -> Dict[str, Any]:
        """
        Analyze dataset quality.
        
        Args:
            X: Feature array
            y: Label array
            dataset_name: Name of dataset
        
        Returns:
            Quality metrics dictionary
        """
        report = {
            'dataset_name': dataset_name,
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'n_classes': len(np.unique(y)),
            'class_names': list(np.unique(y).astype(str)),
            'data_shape': list(X.shape),
            'label_shape': list(y.shape),
            
            # Statistical properties
            'feature_statistics': {
                'mean': float(np.mean(X)),
                'std': float(np.std(X)),
                'min': float(np.min(X)),
                'max': float(np.max(X)),
                'median': float(np.median(X)),
                'q1': float(np.quantile(X, 0.25)),
                'q3': float(np.quantile(X, 0.75))
            },
            
            # Class distribution
            'class_distribution': {}
        }
        
        # Per-class statistics
        for class_id in np.unique(y):
            mask = y == class_id
            n_samples = np.sum(mask)
            percentage = n_samples / len(y) * 100
            
            report['class_distribution'][f'class_{class_id}'] = {
                'n_samples': int(n_samples),
                'percentage': float(percentage)
            }
        
        # Data quality checks
        report['data_quality'] = {
            'has_nan': bool(np.isnan(X).any()),
            'has_inf': bool(np.isinf(X).any()),
            'n_nan': int(np.sum(np.isnan(X))),
            'n_inf': int(np.sum(np.isinf(X))),
            'min_class_samples': int(min(
                np.sum(y == c) for c in np.unique(y)
            )),
            'max_class_samples': int(max(
                np.sum(y == c) for c in np.unique(y)
            )),
            'class_imbalance_ratio': float(
                max(np.sum(y == c) for c in np.unique(y)) /
                min(np.sum(y == c) for c in np.unique(y))
            )
        }
        
        return report
    
    @staticmethod
    def generate_full_report(data_dir: str, dataset_name: str) -> Dict[str, Any]:
        """
        Generate full data quality report for train/val/test splits.
        
        Args:
            data_dir: Directory with preprocessed data
            dataset_name: 'sleep-edf' or 'wesad'
        
        Returns:
            Full report
        """
        data_path = Path(data_dir)
        
        report = {
            'dataset': dataset_name,
            'timestamp': str(Path.cwd()),
            'splits': {}
        }
        
        # Load data
        X_train = np.load(data_path / 'X_train.npy')
        y_train = np.load(data_path / 'y_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        y_val = np.load(data_path / 'y_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        # Analyze each split
        report['splits']['train'] = DataQualityAnalyzer.analyze_dataset(
            X_train, y_train, f"{dataset_name}_train"
        )
        report['splits']['validation'] = DataQualityAnalyzer.analyze_dataset(
            X_val, y_val, f"{dataset_name}_val"
        )
        report['splits']['test'] = DataQualityAnalyzer.analyze_dataset(
            X_test, y_test, f"{dataset_name}_test"
        )
        
        # Overall statistics
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        
        report['overall'] = DataQualityAnalyzer.analyze_dataset(
            X_all, y_all, f"{dataset_name}_combined"
        )
        
        # Check consistency
        report['consistency_checks'] = {
            'all_splits_same_n_features': (
                X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
            ),
            'all_splits_same_n_classes': (
                len(np.unique(y_train)) == len(np.unique(y_val)) == len(np.unique(y_test))
            ),
            'class_distribution_balanced': (
                report['overall']['data_quality']['class_imbalance_ratio'] < 2.0
            )
        }
        
        return report
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            output_path: Path to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {output_path}")


if __name__ == "__main__":
    # Test
    print("Data Quality Analyzer loaded successfully")