#!/usr/bin/env python3
"""
Script para executar todo o pré-processamento localmente.
Configura windowing para Sleep-EDF automaticamente.

Uso:
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --dataset wesad
    python scripts/preprocess_all.py --dataset sleep-edf
    python scripts/preprocess_all.py --dataset all --force_reprocess
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad_temporal
from src.utils.logging_utils import setup_logging, get_logger


def verify_raw_data(data_dir: Path) -> dict:
    """Verify raw data exists."""
    status = {
        'sleep-edf': (Path(data_dir) / 'raw' / 'sleep-edf').exists(),
        'wesad': (Path(data_dir) / 'raw' / 'wesad').exists()
    }
    
    print("\n📂 Raw Data Status:")
    for dataset, exists in status.items():
        symbol = "✓" if exists else "✗"
        print(f"   {symbol} {dataset}: {'Found' if exists else 'NOT FOUND'}")
    
    return status


def verify_processed_data(data_dir: Path) -> dict:
    """Verify preprocessed data exists."""
    required_files = {
        'sleep-edf': ['X_train.npy', 'X_val.npy', 'X_test.npy', 
                     'y_train.npy', 'y_val.npy', 'y_test.npy'],
        'wesad': ['X_train.npy', 'X_val.npy', 'X_test.npy',
                 'y_train.npy', 'y_val.npy', 'y_test.npy']
    }
    
    status = {}
    print("\n✨ Preprocessed Data Status:")
    
    for dataset, files in required_files.items():
        dataset_dir = Path(data_dir) / 'processed' / dataset
        all_exist = all((dataset_dir / f).exists() for f in files)
        status[dataset] = all_exist
        
        symbol = "✓" if all_exist else "✗"
        count = sum(1 for f in files if (dataset_dir / f).exists())
        print(f"   {symbol} {dataset}: {count}/{len(files)} files")
    
    return status


def print_summary(dataset_info: dict, dataset_name: str):
    """Print preprocessing summary."""
    print(f"\n   📊 {dataset_name.upper()} Summary:")
    print(f"      • Classes: {dataset_info.get('class_names', 'N/A')}")
    print(f"      • Train: {dataset_info.get('train_size', 'N/A')} samples")
    print(f"      • Val: {dataset_info.get('val_size', 'N/A')} samples")
    print(f"      • Test: {dataset_info.get('test_size', 'N/A')} samples")
    
    if dataset_name == 'sleep-edf':
        print(f"      • Window: {dataset_info.get('window_size', 'N/A')}-epoch ✓")
    
    if dataset_name == 'wesad':
        print(f"      • Channels: {dataset_info.get('n_channels', 'N/A')}")
        print(f"      • Strategy: {dataset_info.get('strategy', 'N/A')}")


def preprocess_sleep_edf_wrapper(data_dir: Path, args, logger) -> bool:
    """Preprocess Sleep-EDF dataset."""
    print("\n" + "="*80)
    print("🛌 SLEEP-EDF PREPROCESSING")
    print("="*80)
    
    sleep_edf_raw = Path(data_dir) / 'raw' / 'sleep-edf'
    sleep_edf_processed = Path(data_dir) / 'processed' / 'sleep-edf'
    
    if not sleep_edf_raw.exists():
        print(f"❌ Sleep-EDF raw data not found: {sleep_edf_raw}")
        return False
    
    try:
        print(f"\nConfiguration:")
        print(f"   • Window size: {args.sleep_edf_window_size} epochs")
        print(f"   • Test/Val split: {args.test_size}/{args.val_size}")
        print(f"   • Parallel workers: {args.n_workers}")
        print(f"   • Force reprocess: {args.force_reprocess}")
        
        print(f"\nProcessing...")
        start = time.time()
        
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
        
        elapsed = time.time() - start
        print(f"\n✅ Sleep-EDF preprocessing completed in {elapsed:.1f}s!")
        print_summary(info, 'sleep-edf')
        
        logger.info(f"Sleep-EDF preprocessing completed: {info}")
        return True
        
    except Exception as e:
        print(f"\n❌ Sleep-EDF preprocessing failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Sleep-EDF preprocessing failed: {e}", exc_info=True)
        return False


def preprocess_wesad_wrapper(data_dir: Path, args, logger) -> bool:
    """Preprocess WESAD dataset."""
    print("\n" + "="*80)
    print("💊 WESAD PREPROCESSING")
    print("="*80)
    
    wesad_raw = Path(data_dir) / 'raw' / 'wesad'
    wesad_processed = Path(data_dir) / 'processed' / 'wesad'
    
    if not wesad_raw.exists():
        print(f"❌ WESAD raw data not found: {wesad_raw}")
        return False
    
    try:
        print(f"\nConfiguration:")
        print(f"   • Target frequency: {args.wesad_target_freq} Hz")
        print(f"   • Window size: {args.wesad_window_size} samples "
              f"({args.wesad_window_size/args.wesad_target_freq:.0f}s)")
        print(f"   • Overlap: {args.wesad_overlap*100:.0f}%")
        print(f"   • Test/Val split: {args.test_size}/{args.val_size}")
        print(f"   • Parallel workers: {args.n_workers}")
        print(f"   • Force reprocess: {args.force_reprocess}")
        
        print(f"\nProcessing...")
        start = time.time()
        
        info = preprocess_wesad_temporal(
            data_dir=str(wesad_raw),
            output_dir=str(wesad_processed),
            target_freq=args.wesad_target_freq,
            window_size=args.wesad_window_size,
            overlap=args.wesad_overlap,
            binary=True,
            n_workers=args.n_workers,
            force_reprocess=args.force_reprocess
        )
        
        elapsed = time.time() - start
        print(f"\n✅ WESAD preprocessing completed in {elapsed:.1f}s!")
        print_summary(info, 'wesad')
        
        logger.info(f"WESAD preprocessing completed: {info}")
        return True
        
    except Exception as e:
        print(f"\n❌ WESAD preprocessing failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"WESAD preprocessing failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess datasets (Sleep-EDF and/or WESAD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess both datasets
  python scripts/preprocess_all.py

  # Preprocess only WESAD
  python scripts/preprocess_all.py --dataset wesad

  # Preprocess only Sleep-EDF with custom window size
  python scripts/preprocess_all.py --dataset sleep-edf --sleep_edf_window_size 15

  # Force reprocessing
  python scripts/preprocess_all.py --force_reprocess

  # Check data status without processing
  python scripts/preprocess_all.py --check_only
        """)
    
    parser.add_argument(
        '--dataset',
        choices=['all', 'sleep-edf', 'wesad'],
        default='all',
        help='Which dataset(s) to preprocess (default: all)'
    )
    
    parser.add_argument(
        '--data_dir',
        default='./data',
        help='Base data directory (default: ./data)'
    )
    
    parser.add_argument(
        '--sleep_edf_window_size',
        type=int,
        default=10,
        help='Sleep-EDF window size for LSTM in epochs (default: 10)'
    )
    
    parser.add_argument(
        '--wesad_target_freq',
        type=int,
        default=32,
        help='WESAD target frequency in Hz (default: 32)'
    )
    
    parser.add_argument(
        '--wesad_window_size',
        type=int,
        default=1920,
        help='WESAD window size in samples (default: 1920 = 60s @ 32Hz)'
    )
    
    parser.add_argument(
        '--wesad_overlap',
        type=float,
        default=0.5,
        help='WESAD window overlap ratio (default: 0.5)'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Test set size ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help='Validation set size ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--force_reprocess',
        action='store_true',
        help='Force reprocessing even if data exists'
    )
    
    parser.add_argument(
        '--check_only',
        action='store_true',
        help='Check data status without processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    setup_logging(output_dir='./results', level='INFO', verbose=args.verbose)
    logger = get_logger(__name__)
    
    print("\n" + "="*80)
    print("🔄 DATASET PREPROCESSING PIPELINE")
    print("="*80)
    
    data_dir = Path(args.data_dir)
    
    raw_status = verify_raw_data(data_dir)
    
    if not args.force_reprocess:
        processed_status = verify_processed_data(data_dir)
    else:
        processed_status = {'sleep-edf': False, 'wesad': False}
        print("\n⚠️  Force reprocess enabled - existing data will be overwritten")
    
    if args.check_only:
        print("\n✅ Data status check complete")
        return 0
    
    print("\n" + "="*80)
    print("Starting preprocessing...")
    print("="*80)
    
    start_time = time.time()
    results = {'sleep-edf': None, 'wesad': None}
    
    datasets_to_process = []
    if args.dataset in ['all', 'sleep-edf']:
        if args.force_reprocess or not processed_status.get('sleep-edf', False):
            if raw_status['sleep-edf']:
                datasets_to_process.append('sleep-edf')
            else:
                print(f"\n⚠️  Skipping Sleep-EDF: raw data not found")
        else:
            print(f"\n⏭️  Skipping Sleep-EDF: already preprocessed")
    
    if args.dataset in ['all', 'wesad']:
        if args.force_reprocess or not processed_status.get('wesad', False):
            if raw_status['wesad']:
                datasets_to_process.append('wesad')
            else:
                print(f"\n⚠️  Skipping WESAD: raw data not found")
        else:
            print(f"\n⏭️  Skipping WESAD: already preprocessed")
    
    if 'sleep-edf' in datasets_to_process:
        results['sleep-edf'] = preprocess_sleep_edf_wrapper(data_dir, args, logger)
    
    if 'wesad' in datasets_to_process:
        results['wesad'] = preprocess_wesad_wrapper(data_dir, args, logger)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING PIPELINE COMPLETE!")
    print("="*80)
    
    print("\n📊 Results:")
    if results['sleep-edf'] is not None:
        status = "✓ Success" if results['sleep-edf'] else "✗ Failed"
        print(f"   Sleep-EDF: {status}")
    if results['wesad'] is not None:
        status = "✓ Success" if results['wesad'] else "✗ Failed"
        print(f"   WESAD: {status}")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    print(f"\n💡 Next steps:")
    processed_path = Path(args.data_dir) / 'processed'
    
    if (results['sleep-edf'] or (processed_status.get('sleep-edf') and not args.force_reprocess)):
        sleep_edf_path = processed_path / 'sleep-edf'
        print(f"   1. Sleep-EDF ready: {sleep_edf_path}")
        print(f"      python scripts/train_baseline.py --dataset sleep-edf")
    
    if (results['wesad'] or (processed_status.get('wesad') and not args.force_reprocess)):
        wesad_path = processed_path / 'wesad'
        print(f"   2. WESAD ready: {wesad_path}")
        print(f"      python scripts/train_baseline.py --dataset wesad")
    
    print("\n" + "="*80 + "\n")
    
    if any(r is False for r in results.values() if r is not None):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())