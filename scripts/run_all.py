#!/usr/bin/env python3
"""
Script master para executar todo o pipeline localmente.
Uso: python scripts/run_all.py
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run complete pipeline')
    parser.add_argument('--data_dir', default='./data', help='Base data directory')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing')
    parser.add_argument('--skip_training', action='store_true', help='Skip training')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip analysis')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad', 'all'], default='all', help='Dataset to process')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # 1. Pré-processamento
    if not args.skip_preprocessing:
        print("\n🔄 Step 1: Preprocessing...")
        os.system(f"python scripts/preprocess_all.py --data_dir {args.data_dir} --dataset {args.dataset}")
    
    # 2. Treino
    if not args.skip_training:
        print("\n🤖 Step 2: Training...")
        os.system(f"python scripts/train_all.py --data_dir {args.data_dir} --dataset {args.dataset}")
    
    # 3. Análise
    if not args.skip_analysis:
        print("\n📊 Step 3: Analysis...")
        os.system(f"python scripts/run_analysis.py --dataset {args.dataset}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

