#!/usr/bin/env python3
"""
Script para executar todo o treino localmente.
Uso: python scripts/train_all.py
"""

import os
import sys
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.lstm_baseline import train_baseline
from privacy.dp_training import train_dp_model
from privacy.fl_training import train_fl_model

def main():
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--data_dir', default='./data', help='Base data directory')
    parser.add_argument('--models_dir', default='./models', help='Models output directory')
    parser.add_argument('--results_dir', default='./results', help='Results output directory')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad', 'all'], default='all', help='Dataset to train on')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip_dp', action='store_true', help='Skip DP training')
    parser.add_argument('--skip_fl', action='store_true', help='Skip FL training')
    
    args = parser.parse_args()
    
    # Definir paths
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    
    # Criar diretórios
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determinar datasets
    datasets = []
    if args.dataset == 'all':
        datasets = ['sleep-edf', 'wesad']
    else:
        datasets = [args.dataset]
    
    print("="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    
    for dataset in datasets:
        print(f"\n📊 Training on {dataset} dataset...")
        
        processed_path = data_dir / 'processed' / dataset
        if not processed_path.exists():
            print(f"❌ Processed data not found: {processed_path}")
            continue
        
        # Baseline
        if not args.skip_baseline:
            print(f"\n🔵 Training baseline model for {dataset}...")
            try:
                train_baseline(
                    dataset_name=dataset,
                    data_path=str(processed_path),
                    output_dir=str(models_dir / dataset),
                    results_dir=str(results_dir / dataset)
                )
                print(f"✅ Baseline training completed for {dataset}!")
            except Exception as e:
                print(f"❌ Baseline training failed for {dataset}: {e}")
        
        # Differential Privacy
        if not args.skip_dp:
            print(f"\n🔒 Training DP models for {dataset}...")
            try:
                train_dp_model(
                    dataset_name=dataset,
                    data_path=str(processed_path),
                    output_dir=str(models_dir / dataset),
                    results_dir=str(results_dir / dataset)
                )
                print(f"✅ DP training completed for {dataset}!")
            except Exception as e:
                print(f"❌ DP training failed for {dataset}: {e}")
        
        # Federated Learning
        if not args.skip_fl:
            print(f"\n🌐 Training FL models for {dataset}...")
            try:
                train_fl_model(
                    dataset_name=dataset,
                    data_path=str(processed_path),
                    output_dir=str(models_dir / dataset),
                    results_dir=str(results_dir / dataset)
                )
                print(f"✅ FL training completed for {dataset}!")
            except Exception as e:
                print(f"❌ FL training failed for {dataset}: {e}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
