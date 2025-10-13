#!/usr/bin/env python3
"""
Train baseline LSTM model for WESAD dataset
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.wesad import load_processed_wesad
from models.lstm_baseline import train_baseline, evaluate_model, save_model, get_default_config

def main():
    print("="*70)
    print("TRAINING WESAD BASELINE MODEL")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_output_dir = str(base_dir / "models/wesad/baseline")
    results_output_dir = str(base_dir / "results/wesad/baseline")
    
    # Create directories
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)
    
    # Load processed data
    print("Loading processed WESAD data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, info = load_processed_wesad(data_dir)
    
    print(f"\nDataset info:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    
    # Get training configuration
    config = get_default_config()
    config.update({
        'window_size': 10,
        'lstm_units': 128,
        'dense_units': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 64,
        'patience': 15
    })
    
    print(f"\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train model
    print(f"\nStarting training...")
    model, history = train_baseline(X_train, y_train, X_val, y_val, config)
    
    # Evaluate model
    print(f"\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test, config['window_size'])
    
    # Save model and results
    print(f"\nSaving model and results...")
    save_model(model, history, results, models_output_dir, "wesad")
    
    # Save results to results directory
    results_path = os.path.join(results_output_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results also saved to: {results_path}")
    
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1-Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
