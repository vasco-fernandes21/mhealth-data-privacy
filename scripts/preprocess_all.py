#!/usr/bin/env python3
"""
Main preprocessing script for Sleep-EDF and WESAD datasets.
Ultra-optimized for M4 (10 cores).
FEATURES-ONLY VERSION for Sleep-EDF
"""

import sys
import time
import argparse
import psutil
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad_temporal
from src.utils.logging_utils import setup_logging, get_logger


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_memory(label=""):
    """Print memory usage checkpoint."""
    mem = get_memory_usage()
    print(f"   Memory [{label}] {mem:.0f} MB")


def verify_raw_data(data_dir: Path) -> dict:
    """Verify raw data exists."""
    status = {
        'sleep-edf': (Path(data_dir) / 'raw' / 'sleep-edf').exists(),
        'wesad': (Path(data_dir) / 'raw' / 'wesad').exists()
    }

    print("\nRaw Data Status:")
    for dataset, exists in status.items():
        symbol = "✓" if exists else "✗"
        print(f"   {symbol} {dataset}")

    return status


def verify_processed_data(data_dir: Path) -> dict:
    """Verify preprocessed data exists."""
    required_files = {
        'sleep-edf': [
            'X_train.npy', 'X_val.npy', 'X_test.npy',
            'y_train.npy', 'y_val.npy', 'y_test.npy',
            'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy',
            'scaler.pkl', 'preprocessing_info.pkl'
        ],
        'wesad': [
            'X_train.npy', 'X_val.npy', 'X_test.npy',
            'y_train.npy', 'y_val.npy', 'y_test.npy',
            'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy',
            'metadata.pkl', 'normalization_stats.pkl'
        ]
    }

    status = {}
    print("\nPreprocessed Data Status:")

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
    if not dataset_info:
        return

    print(f"\n   Summary {dataset_name.upper()}:")

    if dataset_name == 'sleep-edf':
        print(f"      Features: {dataset_info.get('n_features', 'N/A')}")
        print(f"      Classes: {dataset_info.get('class_names', 'N/A')}")
        print(f"      Samples Train: {dataset_info.get('train_size', 'N/A')}")
        print(f"      Samples Val: {dataset_info.get('val_size', 'N/A')}")
        print(f"      Samples Test: {dataset_info.get('test_size', 'N/A')}")
        print(f"      Total subjects: {len(dataset_info.get('subject_splits', {}).get('train_subjects', []))}")
        print(f"      Processing time: {dataset_info.get('processing_time_s', 'N/A'):.1f}s" if 'processing_time_s' in dataset_info else "")
        print(f"      Type: Features-only (24D per epoch)")

    elif dataset_name == 'wesad':
        print(f"      Channels: {dataset_info.get('n_channels', 'N/A')}")
        print(f"      Classes: {dataset_info.get('class_names', 'N/A')}")
        print(f"      Samples Train: {dataset_info.get('train_size', 'N/A')}")
        print(f"      Samples Val: {dataset_info.get('val_size', 'N/A')}")
        print(f"      Samples Test: {dataset_info.get('test_size', 'N/A')}")
        print(f"      Window: {dataset_info.get('window_duration_s', 'N/A'):.1f}s")
        print(f"      Overlap: {dataset_info.get('overlap', 0)*100:.0f}%")
        print(f"      Subjects: {dataset_info.get('total_subjects', 'N/A')}")
        print(f"      Processing time: {dataset_info.get('processing_time_s', 'N/A'):.1f}s")


def preprocess_sleep_edf_wrapper(
    data_dir: Path, args, logger
) -> bool:
    """Preprocess Sleep-EDF dataset (FEATURES-ONLY)."""
    print("\n" + "="*80)
    print("SLEEP-EDF PREPROCESSING (FEATURES-ONLY)")
    print("="*80)

    sleep_edf_raw = Path(data_dir) / 'raw' / 'sleep-edf'
    sleep_edf_processed = Path(data_dir) / 'processed' / 'sleep-edf'

    if not sleep_edf_raw.exists():
        print(f"ERROR: Sleep-EDF raw data not found: {sleep_edf_raw}")
        return False

    try:
        print(f"\nConfiguration:")
        print(f"   Epoch duration: 30 seconds (100 Hz)")
        print(f"   Features per epoch: 24D")
        print(f"      - 8 features × 3 channels (EEG, EEG, EOG)")
        print(f"      - Per-channel: mean, std, max, min, δ, θ, α, β")
        print(f"   Split: {(1-args.test_size-args.val_size)*100:.0f}% train / "
              f"{args.val_size*100:.0f}% val / {args.test_size*100:.0f}% test")
        print(f"   Subject-wise split: YES (critical for FL)")
        print(f"   Workers: {args.n_workers}")
        print(f"   Force reprocess: {args.force_reprocess}")
        print_memory("Start")

        print(f"\nProcessing Sleep-EDF...")
        start = time.time()

        info = preprocess_sleep_edf(
            data_dir=str(sleep_edf_raw / 'sleep-cassette'),
            output_dir=str(sleep_edf_processed),
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            n_workers=args.n_workers,
            force_reprocess=args.force_reprocess,
            create_windows=False  
        )

        elapsed = time.time() - start
        print(f"\nSleep-EDF preprocessing completed in {elapsed:.1f}s")
        print_memory("End")
        print_summary(info, 'sleep-edf')

        logger.info(f"Sleep-EDF preprocessing (features-only) completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"\nERROR: Sleep-EDF preprocessing failed!")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Sleep-EDF preprocessing failed: {e}", exc_info=True)
        return False


def preprocess_wesad_wrapper(
    data_dir: Path, args, logger
) -> bool:
    """Preprocess WESAD dataset."""
    print("\n" + "="*80)
    print("WESAD PREPROCESSING (TEMPORAL WINDOWS)")
    print("="*80)

    wesad_raw = Path(data_dir) / 'raw' / 'wesad'
    wesad_processed = Path(data_dir) / 'processed' / 'wesad'

    if not wesad_raw.exists():
        print(f"ERROR: WESAD raw data not found: {wesad_raw}")
        return False

    try:
        print(f"\nConfiguration:")
        print(f"   Target frequency: {args.wesad_target_freq} Hz")
        print(f"   Window size: {args.wesad_window_size} samples "
              f"({args.wesad_window_size/args.wesad_target_freq:.1f}s)")
        print(f"   Overlap: {args.wesad_overlap*100:.0f}%")
        print(f"   Stride: {int(args.wesad_window_size*(1-args.wesad_overlap))} samples")
        print(f"   Split: {(1-args.test_size-args.val_size)*100:.0f}% train / "
              f"{args.val_size*100:.0f}% val / {args.test_size*100:.0f}% test")
        print(f"   Subject-wise split: YES (critical for FL)")
        print(f"   Workers: {args.n_workers}")
        print(f"   Force reprocess: {args.force_reprocess}")
        print_memory("Start")

        print(f"\nProcessing WESAD...")
        start = time.time()

        info = preprocess_wesad_temporal(
            data_dir=str(wesad_raw),
            output_dir=str(wesad_processed),
            target_freq=args.wesad_target_freq,
            window_size=args.wesad_window_size,
            overlap=args.wesad_overlap,
            label_threshold=args.wesad_label_threshold,
            binary=True,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            n_workers=args.n_workers,
            force_reprocess=args.force_reprocess
        )

        elapsed = time.time() - start
        print(f"\nWESAD preprocessing completed in {elapsed:.1f}s")
        print_memory("End")
        print_summary(info, 'wesad')

        logger.info(f"WESAD preprocessing completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"\nERROR: WESAD preprocessing failed!")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"WESAD preprocessing failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Sleep-EDF (features) and WESAD (temporal) datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DATASET CONFIGURATIONS:
  Sleep-EDF:    Features-only (24D per 30s epoch)
  WESAD:        Temporal windows (14 channels × 1024 samples)

EXAMPLES:
  # Preprocess all datasets
  python scripts/preprocess_all.py
  
  # Preprocess only Sleep-EDF
  python scripts/preprocess_all.py --dataset sleep-edf
  
  # Force reprocess (overwrite existing)
  python scripts/preprocess_all.py --force_reprocess
  
  # Check data status without processing
  python scripts/preprocess_all.py --check_only
  
  # Custom WESAD window size
  python scripts/preprocess_all.py --wesad_window_size 2048
        """)

    parser.add_argument(
        '--dataset',
        choices=['all', 'sleep-edf', 'wesad'],
        default='all',
        help='Which dataset to preprocess (default: all)'
    )

    parser.add_argument(
        '--data_dir',
        default='./data',
        help='Base data directory (default: ./data)'
    )

    # WESAD arguments (Sleep-EDF has no parameters - fixed 24D features)
    parser.add_argument(
        '--wesad_target_freq',
        type=int,
        default=32,
        help='WESAD target frequency in Hz (default: 32)'
    )

    parser.add_argument(
        '--wesad_window_size',
        type=int,
        default=1024,
        help='WESAD window size in samples (default: 1024 = 32s @ 32Hz)'
    )

    parser.add_argument(
        '--wesad_overlap',
        type=float,
        default=0.75,
        help='WESAD window overlap ratio (default: 0.75)'
    )

    parser.add_argument(
        '--wesad_label_threshold',
        type=float,
        default=0.7,
        help='WESAD label threshold for window validity (default: 0.7)'
    )

    # Common arguments
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Test set fraction (default: 0.15)'
    )

    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help='Validation set fraction (default: 0.15)'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--n_workers',
        type=int,
        default=0,
        help='Number of parallel workers (default: 0=auto-detect)'
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

    # Auto-detect n_workers (use 4 by default for M4, max 6)
    if args.n_workers == 0:
        cpu_count = psutil.cpu_count()
        args.n_workers = min(4, max(2, cpu_count // 2))

    setup_logging(
        output_dir='./results', level='INFO', verbose=args.verbose
    )
    logger = get_logger(__name__)

    print("\n" + "="*80)
    print("DATASET PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Detected {psutil.cpu_count()} CPUs, using {args.n_workers} workers")
    print(f"Random state: {args.random_state}")
    print("\nDataset configurations:")
    print("  • Sleep-EDF:  Features-only (24D per epoch, ~3.4 MB)")
    print("  • WESAD:      Temporal windows (14×1024, ~133 MB)")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_status = verify_raw_data(data_dir)

    if not args.force_reprocess:
        processed_status = verify_processed_data(data_dir)
    else:
        processed_status = {'sleep-edf': False, 'wesad': False}
        print("\nWARNING: Force reprocess enabled - "
              "existing data will be overwritten")

    if args.check_only:
        print("\nData status check complete")
        return 0

    print("\n" + "="*80)
    print("Starting preprocessing...")
    print("="*80)

    start_time = time.time()
    results = {'sleep-edf': None, 'wesad': None}

    datasets_to_process = []

    if args.dataset in ['all', 'sleep-edf']:
        if args.force_reprocess or \
           not processed_status.get('sleep-edf', False):
            if raw_status['sleep-edf']:
                datasets_to_process.append('sleep-edf')
            else:
                print("\nWARNING: Skipping Sleep-EDF: raw data not found")
        else:
            print("\nINFO: Skipping Sleep-EDF: already preprocessed (features)")

    if args.dataset in ['all', 'wesad']:
        if args.force_reprocess or \
           not processed_status.get('wesad', False):
            if raw_status['wesad']:
                datasets_to_process.append('wesad')
            else:
                print("\nWARNING: Skipping WESAD: raw data not found")
        else:
            print("\nINFO: Skipping WESAD: already preprocessed")

    if not datasets_to_process:
        print("\nNo datasets to process!")
        print("Tip: Use --force_reprocess to reprocess existing datasets")
        return 0

    if 'sleep-edf' in datasets_to_process:
        results['sleep-edf'] = preprocess_sleep_edf_wrapper(
            data_dir, args, logger
        )

    if 'wesad' in datasets_to_process:
        results['wesad'] = preprocess_wesad_wrapper(data_dir, args, logger)

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*80)

    print("\nResults:")
    if results['sleep-edf'] is not None:
        status = "✓ Success" if results['sleep-edf'] else "✗ Failed"
        print(f"   Sleep-EDF (features): {status}")
    if results['wesad'] is not None:
        status = "✓ Success" if results['wesad'] else "✗ Failed"
        print(f"   WESAD (temporal): {status}")

    print(f"\nPerformance:")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Final memory: {get_memory_usage():.0f} MB")

    print(f"\nPreprocessed data locations:")
    processed_path = Path(args.data_dir) / 'processed'

    sleep_edf_ok = results['sleep-edf'] or \
                   (processed_status.get('sleep-edf') and
                    not args.force_reprocess)
    if sleep_edf_ok:
        sleep_edf_path = processed_path / 'sleep-edf'
        print(f"   Sleep-EDF: {sleep_edf_path}")
        train_file = sleep_edf_path / 'X_train.npy'
        if train_file.exists():
            data = __import__('numpy').load(train_file)
            print(f"            Shape: {data.shape} (features)")
            print(f"            Size: {data.nbytes / (1024**2):.1f} MB")

    wesad_ok = results['wesad'] or \
               (processed_status.get('wesad') and not args.force_reprocess)
    if wesad_ok:
        wesad_path = processed_path / 'wesad'
        print(f"   WESAD: {wesad_path}")
        train_file = wesad_path / 'X_train.npy'
        if train_file.exists():
            data = __import__('numpy').load(train_file)
            print(f"           Shape: {data.shape} (channels × samples)")
            print(f"           Size: {data.nbytes / (1024**2):.1f} MB")

    print("\n" + "="*80)
    print("Next steps:")
    print("   python scripts/run_experiments.py --scenario baseline")
    print("   python scripts/run_experiments.py --scenario fl")
    print("   python scripts/run_experiments.py --scenario dp")
    print("="*80 + "\n")

    if any(r is False for r in results.values() if r is not None):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())