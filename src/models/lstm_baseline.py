"""
LSTM Baseline Model Implementation

This module provides the baseline LSTM model architecture and training functions
for health data classification without privacy-preserving techniques.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def build_lstm_model(input_shape: Tuple[int, int], n_classes: int, 
                    lstm_units: int = 64, dense_units: int = 32,
                    dropout_rate: float = 0.5, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build LSTM baseline model architecture.
    
    Args:
        input_shape: Shape of input data (n_timesteps, n_features)
        n_classes: Number of output classes
        lstm_units: Number of LSTM units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def reshape_for_lstm(X: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Reshape data for LSTM input format.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        window_size: Number of timesteps for LSTM window
    
    Returns:
        Reshaped data of shape (n_samples - window_size + 1, window_size, n_features)
    """
    n_samples, n_features = X.shape
    X_reshaped = []
    
    for i in range(n_samples - window_size + 1):
        X_reshaped.append(X[i:i+window_size, :])
    
    return np.array(X_reshaped)


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray,
                  config: Dict[str, Any]) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train baseline LSTM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration dictionary
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("="*70)
    print("TRAINING LSTM BASELINE MODEL")
    print("="*70)
    
    # Reshape data for LSTM
    window_size = config.get('window_size', 10)
    X_train_lstm = reshape_for_lstm(X_train, window_size)
    X_val_lstm = reshape_for_lstm(X_val, window_size)
    
    # Adjust labels
    y_train_lstm = y_train[window_size-1:]
    y_val_lstm = y_val[window_size-1:]
    
    # Convert to categorical
    n_classes = len(np.unique(y_train_lstm))
    y_train_cat = to_categorical(y_train_lstm, n_classes)
    y_val_cat = to_categorical(y_val_lstm, n_classes)
    
    print(f"LSTM data shapes:")
    print(f"  Train: {X_train_lstm.shape}, Val: {X_val_lstm.shape}")
    print(f"  Classes: {n_classes}")
    
    # Build model
    input_shape = (window_size, X_train_lstm.shape[2])
    model = build_lstm_model(
        input_shape=input_shape,
        n_classes=n_classes,
        lstm_units=config.get('lstm_units', 64),
        dense_units=config.get('dense_units', 32),
        dropout_rate=config.get('dropout_rate', 0.5),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nTraining model...")
    history = model.fit(
        X_train_lstm, y_train_cat,
        validation_data=(X_val_lstm, y_val_cat),
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                  window_size: int = 10) -> Dict[str, float]:
    """
    Evaluate trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        window_size: Window size used for LSTM
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Reshape test data
    X_test_lstm = reshape_for_lstm(X_test, window_size)
    y_test_lstm = y_test[window_size-1:]
    
    # Convert to categorical
    n_classes = len(np.unique(y_test_lstm))
    y_test_cat = to_categorical(y_test_lstm, n_classes)
    
    # Predictions
    y_pred_probs = model.predict(X_test_lstm, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_lstm, y_pred)
    precision = precision_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_lstm, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return results


def save_model(model: tf.keras.Model, history: tf.keras.callbacks.History,
               results: Dict[str, float], save_path: str, dataset_name: str):
    """
    Save trained model, history, and results.
    
    Args:
        model: Trained Keras model
        history: Training history
        results: Evaluation results
        save_path: Directory to save files
        dataset_name: Name of dataset (for file naming)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_path, f'lstm_baseline_{dataset_name}.h5')
    model.save(model_path)
    print(f"Model saved: {model_path}")
    
    # Save training history
    history_path = os.path.join(save_path, f'history_baseline_{dataset_name}.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f, indent=2)
    print(f"History saved: {history_path}")
    
    # Save results
    results_path = os.path.join(save_path, f'results_baseline_{dataset_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load trained model from file.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def create_callbacks(patience: int = 10, min_lr: float = 1e-6) -> list:
    """
    Create training callbacks.
    
    Args:
        patience: Early stopping patience
        min_lr: Minimum learning rate
    
    Returns:
        List of callbacks
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        )
    ]


def get_default_config() -> Dict[str, Any]:
    """
    Get default training configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'window_size': 10,
        'lstm_units': 64,
        'dense_units': 32,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10
    }


if __name__ == "__main__":
    # Example usage
    print("LSTM Baseline Model Module")
    print("Use this module to train baseline models for comparison with DP and FL approaches.")
