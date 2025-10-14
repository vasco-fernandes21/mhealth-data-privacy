#!/usr/bin/env python3
"""
Train baseline CNN-LSTM model for WESAD binary stress classification.

Optimized for 32 Hz sampling frequency for optimal signal quality.
Simple, efficient architecture suitable for privacy analysis (DP-SGD, Federated Learning).
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.wesad import load_processed_wesad_temporal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


def build_cnn_lstm_binary(input_shape, n_classes=2, learning_rate=0.001):
    """
    Build CNN-LSTM model for binary stress classification.
    
    Architecture optimized for 32 Hz, 14 channels, 1920 timesteps (60s windows):
    - 2 Conv1D blocks: Local pattern extraction
    - 1 LSTM layer: Temporal dependencies
    - Dense layers: Classification
    
    Total params: ~523K (lightweight for privacy analysis)
    
    Args:
        input_shape: (n_channels, window_size), e.g., (14, 1920) for 32 Hz
        n_classes: Number of output classes (default: 2 for binary)
        learning_rate: Adam optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    # L2 regularization to reduce overfitting
    l2_reg = regularizers.l2(1e-4)

    model = Sequential([
        # First Conv block: Fewer filters + L2 + spatial dropout
        Conv1D(32, kernel_size=7, activation='relu', padding='same', kernel_regularizer=l2_reg, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),  # 1920 → 480
        SpatialDropout1D(0.2),

        # Second Conv block: Fewer filters + L2 + spatial dropout
        Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),  # 480 → 240
        SpatialDropout1D(0.2),

        # LSTM: Fewer units + dropout
        LSTM(32, return_sequences=False, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg),
        Dropout(0.5),

        # Dense layers: L2 + dropout
        Dense(32, activation='relu', kernel_regularizer=l2_reg),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name='CNN_LSTM_Binary_32Hz_Regularized')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    
    return model


def _simple_oversample(X: np.ndarray, y: np.ndarray):
    """Naive majority-class down to minority? No, replicate minority to balance (simple oversampling)."""
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())
    max_count = max(counts.values())
    X_balanced = []
    y_balanced = []
    for c in classes:
        idx = np.where(y == c)[0]
        reps = int(np.ceil(max_count / len(idx)))
        idx_rep = np.tile(idx, reps)[:max_count]
        X_balanced.append(X[idx_rep])
        y_balanced.append(y[idx_rep])
    Xb = np.concatenate(X_balanced, axis=0)
    yb = np.concatenate(y_balanced, axis=0)
    # Shuffle
    perm = np.random.permutation(len(yb))
    return Xb[perm], yb[perm]


def train_model(X_train, y_train, X_val, y_val, output_dir, 
                epochs=100, batch_size=32, learning_rate=0.001,
                balance: str = 'class_weight',
                augment: bool = True,
                noise_std: float = 0.01,
                max_time_shift: int = 8):
    """
    Train CNN-LSTM model with early stopping and learning rate reduction.
    
    Args:
        X_train, y_train: Training data (X shape: (n, channels, timesteps))
        X_val, y_val: Validation data
        output_dir: Directory to save best model
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
    
    Returns:
        Tuple of (model, history, training_time_seconds)
    """
    print("\n" + "="*70)
    print("TRAINING CNN-LSTM MODEL")
    print("="*70)
    
    # Optional balancing
    class_weights = None
    if balance == 'class_weight':
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        print(f"\nUsing class weights: {class_weights}")
    elif balance == 'oversample':
        print("\nApplying simple oversampling of minority class to balance train set...")
        X_train, y_train = _simple_oversample(X_train, y_train)
        print(f"  New train shape: {X_train.shape}, class distribution: {Counter(y_train.tolist())}")
    elif balance == 'smote':
        if SMOTE is None:
            print("\nSMOTE requested but imblearn not available. Falling back to class_weight.")
            class_weights_array = compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        else:
            # Reshape to (n_samples, features) for SMOTE, then back to (C, T)
            n, c, t = X_train.shape
            X2d = X_train.reshape(n, c * t)
            print("\nApplying SMOTE to training set...")
            sm = SMOTE()
            X2d_bal, y_train = sm.fit_resample(X2d, y_train)
            X_train = X2d_bal.reshape(len(y_train), c, t)
            print(f"  New train shape: {X_train.shape}, class distribution: {Counter(y_train.tolist())}")

    # Optional simple temporal augmentation (noise + small time shift)
    if augment:
        rng = np.random.default_rng(42)
        X_aug = X_train.copy()
        # Gaussian noise
        X_aug += rng.normal(0, noise_std, size=X_aug.shape)
        # Small time shift per sample
        if max_time_shift > 0:
            for i in range(X_aug.shape[0]):
                shift = int(rng.integers(-max_time_shift, max_time_shift + 1))
                if shift != 0:
                    if shift > 0:
                        X_aug[i, :, shift:] = X_aug[i, :, :-shift]
                    else:
                        X_aug[i, :, :shift] = X_aug[i, :, -shift:]
        X_train = X_aug

    # Prepare labels (after any balancing)
    n_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape} → {y_train_cat.shape}")
    print(f"  Val:   {X_val.shape} → {y_val_cat.shape}")
    print(f"  Classes: {n_classes}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\nBuilding model for input shape: {input_shape}")
    
    model = build_cnn_lstm_binary(input_shape, n_classes, learning_rate)
    
    print("\nModel architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_model_wesad.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"  Epochs: {len(history.history['loss'])}")
    
    return model, history, training_time


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    # Predictions
    n_classes = len(np.unique(y_test))
    y_test_cat = to_categorical(y_test, n_classes)
    
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  Predicted →")
    print(f"  Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:12s}: {row}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Build results dictionary
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': {class_names[i]: float(precision_per_class[i]) for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: float(recall_per_class[i]) for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    return results


def save_results(model, history, results, training_time, models_dir, results_dir):
    """
    Save model, training history, and evaluation results.
    
    Args:
        model: Trained model
        history: Training history object
        results: Evaluation results dictionary
        training_time: Training time in seconds
        models_dir: Directory to save model files
        results_dir: Directory to save result JSONs
    """
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'model_wesad_binary.h5')
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'epochs': len(history.history['loss']),
        'training_time_seconds': float(training_time)
    }
    
    history_path = os.path.join(models_dir, 'history_wesad_binary.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ History saved: {history_path}")
    
    # Save results to models dir
    results_model_path = os.path.join(models_dir, 'results_wesad_binary.json')
    with open(results_model_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_model_path}")
    
    # Save results to results dir
    results_path = os.path.join(results_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    print(f"\n{'='*70}")


def main():
    """Main training pipeline."""
    
    print("="*70)
    print("WESAD BINARY STRESS CLASSIFICATION - BASELINE MODEL")
    print("Dataset: WESAD (32 Hz, binary: stress vs non-stress)")
    print("Model: CNN-LSTM (optimized for privacy analysis)")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_dir = str(base_dir / "models/wesad/baseline")
    results_dir = str(base_dir / "results/wesad/baseline")
    
    # Check data
    if not os.path.exists(data_dir):
        print(f"\n❌ Error: Data not found at {data_dir}")
        print("Run preprocessing first:")
        print("  python src/preprocessing/wesad.py --binary")
        return 1
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = load_processed_wesad_temporal(data_dir)
    
    print(f"\nDataset info:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {info['class_names']}")
    print(f"  Channels: {info['n_channels']}")
    print(f"  Window: {info['window_size']} samples ({info['window_duration_s']:.1f}s)")
    print(f"  Frequency: {info['target_freq']} Hz")
    
    # Train model
    model, history, training_time = train_model(
        X_train, y_train,
        X_val, y_val,
        models_dir,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        balance='oversample'
    )
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, info['class_names'])
    
    # Save everything
    save_results(model, history, results, training_time, models_dir, results_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  Stress Recall: {results['recall_per_class']['stress']:.4f}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"\n  Models:  {models_dir}")
    print(f"  Results: {results_dir}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
