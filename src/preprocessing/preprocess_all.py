#!/usr/bin/env python3
"""
Script para executar todo o pré-processamento localmente.
Uso: python scripts/preprocess_all.py
"""

import os
import sys
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.sleep_edf import preprocess_sleep_edf
from preprocessing.wesad import preprocess_wesad

def main():
    parser = argparse.ArgumentParser(description='Preprocess all datasets')
    parser.add_argument('--data_dir', default='./data', help='Base data directory')
    parser.add_argument('--skip_sleep_edf', action='store_true', help='Skip Sleep-EDF preprocessing')
    parser.add_argument('--skip_wesad', action='store_true', help='Skip WESAD preprocessing')
    
    args = parser.parse_args()
    
    # Definir paths
    raw_dir = Path(args.data_dir) / 'raw'
    processed_dir = Path(args.data_dir) / 'processed'
    
    # Criar diretórios
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREPROCESSING ALL DATASETS")
    print("="*70)
    
    # Sleep-EDF
    if not args.skip_sleep_edf:
        print("\n🛌 Processing Sleep-EDF dataset...")
        sleep_edf_raw = raw_dir / 'sleep-edf'
        sleep_edf_processed = processed_dir / 'sleep-edf'
        
        if sleep_edf_raw.exists():
            preprocess_sleep_edf(
                data_dir=str(sleep_edf_raw),
                output_dir=str(sleep_edf_processed),
                test_size=0.15,
                val_size=0.15,
                random_state=42
            )
            print("✅ Sleep-EDF preprocessing completed!")
        else:
            print(f"❌ Sleep-EDF raw data not found: {sleep_edf_raw}")
    
    # WESAD
    if not args.skip_wesad:
        print("\n😰 Processing WESAD dataset...")
        wesad_raw = raw_dir / 'wesad'
        wesad_processed = processed_dir / 'wesad'
        
        if wesad_raw.exists():
            preprocess_wesad(
                data_dir=str(wesad_raw),
                output_dir=str(wesad_processed),
                test_size=0.15,
                val_size=0.15,
                random_state=42
            )
            print("✅ WESAD preprocessing completed!")
        else:
            print(f"❌ WESAD raw data not found: {wesad_raw}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
