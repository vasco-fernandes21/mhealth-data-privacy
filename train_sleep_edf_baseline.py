#!/usr/bin/env python3
"""
Script para treinar o modelo baseline do Sleep-EDF
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.sleep_edf import load_processed_sleep_edf
from models.lstm_baseline import train_baseline, evaluate_model, save_model, get_default_config

def main():
    print("="*70)
    print("TRAINING SLEEP-EDF BASELINE MODEL")
    print("="*70)
    
    # Paths
    data_path = "data/processed/sleep-edf"
    output_dir = "models/sleep-edf"
    results_dir = "results/sleep-edf"
    
    # Criar diretórios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Carregar dados processados
    print("Loading processed Sleep-EDF data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = load_processed_sleep_edf(data_path)
    
    print(f"\nDataset info:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    
    # Configuração do modelo
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
    
    # Treinar modelo
    print(f"\nStarting training...")
    model, history = train_baseline(X_train, y_train, X_val, y_val, config)
    
    # Avaliar modelo
    print(f"\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test, config['window_size'])
    
    # Guardar modelo e resultados
    print(f"\nSaving model and results...")
    save_model(model, history, results, output_dir, "sleep-edf")
    
    # Guardar resultados também no diretório de resultados
    import json
    results_path = os.path.join(results_dir, 'baseline_results.json')
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
