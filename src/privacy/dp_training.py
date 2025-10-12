"""
Differential Privacy Training Module

This module provides functions for training models with Differential Privacy
using TensorFlow Privacy.
"""

import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def build_dp_model(input_shape: Tuple[int, int], n_classes: int,
                  l2_norm_clip: float = 1.0, noise_multiplier: float = 1.1,
                  learning_rate: float = 0.001, lstm_units: int = 64,
                  dense_units: int = 32, dropout_rate: float = 0.5) -> tf.keras.Model:
    """
    Build LSTM model with Differential Privacy optimizer.
    
    Args:
        input_shape: Shape of input data (n_timesteps, n_features)
        n_classes: Number of output classes
        l2_norm_clip: Clipping norm for gradients
        noise_multiplier: Noise multiplier for DP
        learning_rate: Learning rate
        lstm_units: Number of LSTM units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model with DP optimizer
    """
    # Build model architecture
    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_classes, activation='softmax')
    ])
    
    # Create DP optimizer
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=learning_rate
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def compute_epsilon(steps: int, num_train_examples: int, batch_size: int,
                   noise_multiplier: float, delta: float = 1e-5) -> float:
    """
    Compute epsilon (privacy budget) for given parameters.
    
    Args:
        steps: Number of training steps
        num_train_examples: Number of training examples
        batch_size: Batch size
        noise_multiplier: Noise multiplier used
        delta: Delta parameter (typically 1e-5)
    
    Returns:
        Computed epsilon value
    """
    if noise_multiplier == 0:
        return float('inf')
    
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=num_train_examples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=steps / (num_train_examples / batch_size),
        delta=delta
    )[0]


def train_with_dp(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 config: Dict[str, Any]) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    """
    Train model with Differential Privacy.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
    
    Returns:
        Tuple of (trained_model, training_history, privacy_info)
    """
    print("="*70)
    print("TRAINING WITH DIFFERENTIAL PRIVACY")
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
    
    # DP parameters
    l2_norm_clip = config.get('l2_norm_clip', 1.0)
    noise_multiplier = config.get('noise_multiplier', 1.1)
    epsilon_target = config.get('epsilon_target', 1.0)
    
    print(f"\nDP Parameters:")
    print(f"  L2 norm clip: {l2_norm_clip}")
    print(f"  Noise multiplier: {noise_multiplier}")
    print(f"  Target epsilon: {epsilon_target}")
    
    # Build DP model
    input_shape = (window_size, X_train_lstm.shape[2])
    model = build_dp_model(
        input_shape=input_shape,
        n_classes=n_classes,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        learning_rate=config.get('learning_rate', 0.001),
        lstm_units=config.get('lstm_units', 64),
        dense_units=config.get('dense_units', 32),
        dropout_rate=config.get('dropout_rate', 0.5)
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
    print(f"\nTraining model with DP...")
    history = model.fit(
        X_train_lstm, y_train_cat,
        validation_data=(X_val_lstm, y_val_cat),
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )
    
    # Compute actual epsilon
    num_train_examples = len(X_train_lstm)
    batch_size = config.get('batch_size', 32)
    steps = len(history.history['loss'])
    
    actual_epsilon = compute_epsilon(
        steps=steps,
        num_train_examples=num_train_examples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier
    )
    
    privacy_info = {
        'epsilon_target': epsilon_target,
        'epsilon_actual': actual_epsilon,
        'noise_multiplier': noise_multiplier,
        'l2_norm_clip': l2_norm_clip,
        'delta': 1e-5,
        'steps': steps,
        'num_train_examples': num_train_examples
    }
    
    print(f"\nPrivacy Analysis:")
    print(f"  Target epsilon: {epsilon_target}")
    print(f"  Actual epsilon: {actual_epsilon:.4f}")
    print(f"  Privacy budget used: {actual_epsilon/epsilon_target*100:.1f}%")
    
    return model, history, privacy_info


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


def evaluate_dp_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                     window_size: int = 10) -> Dict[str, float]:
    """
    Evaluate DP model on test data.
    
    Args:
        model: Trained DP model
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
    
    print(f"\nDP Model Test Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return results


def save_dp_model(model: tf.keras.Model, history: tf.keras.callbacks.History,
                 results: Dict[str, float], privacy_info: Dict[str, Any],
                 save_path: str, model_name: str):
    """
    Save DP model, history, results, and privacy info.
    
    Args:
        model: Trained DP model
        history: Training history
        results: Evaluation results
        privacy_info: Privacy analysis info
        save_path: Directory to save files
        model_name: Name for the model
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_path, f'{model_name}.h5')
    model.save(model_path)
    print(f"DP model saved: {model_path}")
    
    # Save training history
    history_path = os.path.join(save_path, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f, indent=2)
    print(f"History saved: {history_path}")
    
    # Save results
    results_path = os.path.join(save_path, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    # Save privacy info
    privacy_path = os.path.join(save_path, f'{model_name}_privacy.json')
    with open(privacy_path, 'w') as f:
        json.dump(privacy_info, f, indent=2)
    print(f"Privacy info saved: {privacy_path}")


def get_dp_configs(epsilon_values: list = [0.1, 1.0, 5.0, 10.0]) -> list:
    """
    Get DP configurations for different epsilon values.
    
    Args:
        epsilon_values: List of epsilon values to test
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for epsilon in epsilon_values:
        # Estimate noise multiplier for target epsilon
        # This is a rough approximation - in practice, you'd use more sophisticated methods
        if epsilon <= 0.5:
            noise_multiplier = 2.0
        elif epsilon <= 1.0:
            noise_multiplier = 1.5
        elif epsilon <= 5.0:
            noise_multiplier = 1.1
        else:
            noise_multiplier = 0.8
        
        config = {
            'window_size': 10,
            'lstm_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'patience': 10,
            'l2_norm_clip': 1.0,
            'noise_multiplier': noise_multiplier,
            'epsilon_target': epsilon,
            'privacy_technique': 'DP'
        }
        configs.append(config)
    
    return configs


if __name__ == "__main__":
    print("Differential Privacy Training Module")
    print("This module provides functions for training models with DP.")
