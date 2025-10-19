#!/usr/bin/env python3
"""
Script para executar todo o pré-processamento localmente.
Configura windowing para Sleep-EDF e augmentation para WESAD automaticamente.

Uso:
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --skip_sleep_edf
    python scripts/preprocess_all.py --skip_wesad
    python scripts/preprocess_all.py --force_reprocess
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad_temporal
from src.utils.logging_utils import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description='Preprocess all datasets')
    
    # Dataset selection
    parser.add_argument('--data_dir', default='./data', help='Base data directory')
    parser.add_argument('--skip_sleep_edf', action='store_true', help='Skip Sleep-EDF preprocessing')
    parser.add_argument('--skip_wesad', action='store_true', help='Skip WESAD preprocessing')
    
    # Sleep-EDF parameters
    parser.add_argument('--sleep_edf_window_size', type=int, default=10,
                       help='Sleep-EDF window size for LSTM (default: 10 epochs)')
    
    # WESAD parameters
    parser.add_argument('--wesad_target_freq', type=int, default=32,
                       help='WESAD target frequency Hz (default: 32)')
    parser.add_argument('--wesad_window_size', type=int, default=1920,
                       help='WESAD window size in samples (default: 1920=60s@32Hz)')
    parser.add_argument('--wesad_overlap', type=float, default=0.5,
                       help='WESAD window overlap (default: 0.5)')
    parser.add_argument('--wesad_n_augmentations', type=int, default=2,
                       help='WESAD augmentations per sample (default: 2 → 3× training data)')
    
    # Common parameters
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Test set size (default: 0.15)')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Validation set size (default: 0.15)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Force reprocessing even if data exists')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(output_dir='./results', level='INFO', verbose=True)
    logger = get_logger(__name__)
    
    print("\n" + "="*80)
    print("🔄 PREPROCESSING ALL DATASETS")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Random state: {args.random_state}")
    print(f"Parallel workers: {args.n_workers}")
    print(f"Test/Val split: {args.test_size}/{args.val_size}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # ============================================================================
    # Sleep-EDF Preprocessing
    # ============================================================================
    if not args.skip_sleep_edf:
        print("\n🛌 SLEEP-EDF PREPROCESSING")
        print("-" * 80)
        
        sleep_edf_raw = Path(args.data_dir) / 'raw' / 'sleep-edf'
        sleep_edf_processed = Path(args.data_dir) / 'processed' / 'sleep-edf'
        
        if sleep_edf_raw.exists():
            try:
                info = preprocess_sleep_edf(
                    data_dir=str(sleep_edf_raw),
                    output_dir=str(sleep_edf_processed),
                    test_size=args.test_size,
                    val_size=args.val_size,
                    random_state=args.random_state,
                    n_workers=args.n_workers,
                    force_reprocess=args.force_reprocess,
                    create_windows=True,
                    window_size=args.sleep_edf_window_size
                )
                
                print(f"\n✅ Sleep-EDF preprocessing completed!")
                print(f"   • Train: {info['train_size']} epochs")
                print(f"   • Val: {info['val_size']} epochs")
                print(f"   • Test: {info['test_size']} epochs")
                if info.get('has_windowed_data'):
                    print(f"   • Windowed data: {args.sleep_edf_window_size}-epoch windows created")
                
            except Exception as e:
                print(f"❌ Sleep-EDF preprocessing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Sleep-EDF raw data not found: {sleep_edf_raw}")
    
    # ============================================================================
    # WESAD Preprocessing
    # ============================================================================
    if not args.skip_wesad:
        print("\n💊 WESAD PREPROCESSING")
        print("-" * 80)
        
        wesad_raw = Path(args.data_dir) / 'raw' / 'wesad'
        wesad_processed = Path(args.data_dir) / 'processed' / 'wesad'
        
        if wesad_raw.exists():
            try:
                info = preprocess_wesad_temporal(
                    data_dir=str(wesad_raw),
                    output_dir=str(wesad_processed),
                    # Signal processing
                    target_freq=args.wesad_target_freq,
                    window_size=args.wesad_window_size,
                    overlap=args.wesad_overlap,
                    # Classification
                    binary=True,
                    # Splitting
                    test_size=args.test_size,
                    val_size=args.val_size,
                    random_state=args.random_state,
                    # Parallel processing
                    n_workers=args.n_workers,
                    force_reprocess=args.force_reprocess,
                    # Augmentation
                    create_augmentations=True,
                    n_augmentations=args.wesad_n_augmentations
                )
                
                print(f"\n✅ WESAD preprocessing completed!")
                print(f"   • Classes: {info['class_names']}")
                print(f"   • Train: {info['train_size']} windows")
                print(f"   • Val: {info['val_size']} windows")
                print(f"   • Test: {info['test_size']} windows")
                if info.get('has_augmented_data'):
                    aug_factor = info.get('n_augmentations', 2) + 1
                    print(f"   • Augmented training: {info['train_size'] * aug_factor} samples ({aug_factor}× factor)")
                
            except Exception as e:
                print(f"❌ WESAD preprocessing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  WESAD raw data not found: {wesad_raw}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\n📊 Datasets ready for training:")
    print(f"   • Sleep-EDF: {Path(args.data_dir) / 'processed' / 'sleep-edf'}")
    print(f"   • WESAD: {Path(args.data_dir) / 'processed' / 'wesad'}")
    print(f"\n💡 Next steps:")
    print(f"   1. python scripts/train_baseline.py --dataset sleep-edf")
    print(f"   2. python scripts/train_baseline.py --dataset wesad")
    print(f"   3. python scripts/train_dp.py --dataset sleep-edf --epsilon 1.0")
    print("="*80 + "\n")


if __name__ == "__main__":
    sys.exit(main())