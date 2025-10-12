"""
Model Utilities

This module provides utility functions for model management,
including save/load operations and common configurations.
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def save_model(model: tf.keras.Model, save_path: str, model_name: str, 
               additional_info: Optional[Dict] = None):
    """
    Save a Keras model with metadata.
    
    Args:
        model: Keras model to save
        save_path: Directory to save the model
        model_name: Name for the model file
        additional_info: Additional metadata to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_file = os.path.join(save_path, f'{model_name}.h5')
    model.save(model_file)
    print(f"Model saved: {model_file}")
    
    # Save metadata
    if additional_info:
        metadata_file = os.path.join(save_path, f'{model_name}_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(additional_info, f, indent=2)
        print(f"Metadata saved: {metadata_file}")


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a Keras model from file.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def save_training_history(history: tf.keras.callbacks.History, save_path: str, 
                         history_name: str):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras training history
        save_path: Directory to save the history
        history_name: Name for the history file
    """
    os.makedirs(save_path, exist_ok=True)
    
    history_file = os.path.join(save_path, f'{history_name}_history.json')
    with open(history_file, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f, indent=2)
    print(f"Training history saved: {history_file}")


def save_results(results: Dict[str, Any], save_path: str, results_name: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Directory to save the results
        results_name: Name for the results file
    """
    os.makedirs(save_path, exist_ok=True)
    
    results_file = os.path.join(save_path, f'{results_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")


def create_callbacks(patience: int = 10, min_lr: float = 1e-6, 
                    monitor: str = 'val_loss') -> list:
    """
    Create common training callbacks.
    
    Args:
        patience: Early stopping patience
        min_lr: Minimum learning rate
        monitor: Metric to monitor
    
    Returns:
        List of Keras callbacks
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        )
    ]


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
    
    Returns:
        Model summary string
    """
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    return buffer.getvalue()


def count_parameters(model: tf.keras.Model) -> int:
    """
    Count total number of trainable parameters in model.
    
    Args:
        model: Keras model
    
    Returns:
        Number of trainable parameters
    """
    return model.count_params()


def get_model_size_mb(model: tf.keras.Model) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: Keras model
    
    Returns:
        Model size in MB
    """
    # Save model temporarily to get file size
    temp_path = '/tmp/temp_model.h5'
    model.save(temp_path)
    size_bytes = os.path.getsize(temp_path)
    os.remove(temp_path)
    
    return size_bytes / (1024 * 1024)  # Convert to MB


def compare_models(model_paths: list, model_names: list = None) -> Dict[str, Any]:
    """
    Compare multiple models by loading and analyzing them.
    
    Args:
        model_paths: List of paths to model files
        model_names: List of names for models (optional)
    
    Returns:
        Dictionary with comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(model_paths))]
    
    comparison = {}
    
    for path, name in zip(model_paths, model_names):
        if os.path.exists(path):
            try:
                model = load_model(path)
                comparison[name] = {
                    'parameters': count_parameters(model),
                    'size_mb': get_model_size_mb(model),
                    'layers': len(model.layers),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape
                }
            except Exception as e:
                comparison[name] = {'error': str(e)}
        else:
            comparison[name] = {'error': 'File not found'}
    
    return comparison


def save_comparison(comparison: Dict[str, Any], save_path: str, filename: str = 'model_comparison.json'):
    """
    Save model comparison results.
    
    Args:
        comparison: Comparison results dictionary
        save_path: Directory to save the comparison
        filename: Name for the comparison file
    """
    os.makedirs(save_path, exist_ok=True)
    
    comparison_file = os.path.join(save_path, filename)
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Model comparison saved: {comparison_file}")


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata file
    
    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_experiment_log(experiment_name: str, config: Dict[str, Any], 
                         results: Dict[str, Any], save_path: str):
    """
    Create a comprehensive experiment log.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration used
        results: Results obtained
        save_path: Directory to save the log
    """
    os.makedirs(save_path, exist_ok=True)
    
    log = {
        'experiment_name': experiment_name,
        'timestamp': str(pd.Timestamp.now()),
        'configuration': config,
        'results': results
    }
    
    log_file = os.path.join(save_path, f'{experiment_name}_experiment_log.json')
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Experiment log saved: {log_file}")


if __name__ == "__main__":
    print("Model Utilities Module")
    print("This module provides utility functions for model management.")
