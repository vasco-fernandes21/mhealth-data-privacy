#!/usr/bin/env python3
"""
Preprocessing pipeline for Sleep-EDF and WESAD (features-only).
"""

import sys
import time
import argparse
import psutil
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad
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
        symbol = "OK" if exists else "MISSING"
        print(f"   [{symbol}] {dataset}")

    return status


def verify_processed_data(data_dir: Path) -> dict:
    """Verify preprocessed data exists."""
    status = {}

    print("\nPreprocessed Data Status:")

    # Sleep-EDF features
    sleep_dir = Path(data_dir) / 'processed' / 'sleep-edf'
    sleep_ok = all((sleep_dir / f).exists() for f in
                   ['X_train.npy', 'y_train.npy', 'scaler.pkl'])
    status['sleep-edf'] = sleep_ok
    print(f"   [{'OK' if sleep_ok else 'MISSING'}] sleep-edf (24D features)")

    # WESAD features
    wesad_dir = Path(data_dir) / 'processed' / 'wesad'
    wesad_ok = all((wesad_dir / f).exists() for f in
                   ['X_train.npy', 'y_train.npy', 'scaler.pkl'])
    status['wesad'] = wesad_ok
    print(f"   [{'OK' if wesad_ok else 'MISSING'}] wesad (140D features)")

    return status


def preprocess_sleep_edf_wrapper(
    data_dir: Path, args, logger
) -> bool:
    """Preprocess Sleep-EDF dataset (features-only)."""
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
        print(f"   Type: Features-only (24D per 30s epoch)")
        print(f"   Features: 8 per channel × 3 channels (EEG, EEG, EOG)")
        print(f"   Split: {(1-args.test_size-args.val_size)*100:.0f}% train / "
              f"{args.val_size*100:.0f}% val / {args.test_size*100:.0f}% test")
        print(f"   Workers: {args.n_workers}")
        print_memory("Start")

        print(f"\nProcessing...")
        start = time.time()

        info = preprocess_sleep_edf(
            data_dir=str(sleep_edf_raw / 'sleep-cassette'),
            output_dir=str(sleep_edf_processed),
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            n_workers=args.n_workers,
            force_reprocess=args.force_reprocess
        )

        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")
        print_memory("End")

        if info:
            print(f"\nSummary:")
            print(f"   Features: 24D per epoch")
            print(f"   Classes: {info.get('n_classes')} {info.get('class_names')}")
            print(f"   Train: {info.get('train_size')} | "
                  f"Val: {info.get('val_size')} | Test: {info.get('test_size')}")
            print(f"   Size: {info.get('total_size_mb', 0):.1f} MB")

        logger.info(f"Sleep-EDF preprocessing completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Sleep-EDF preprocessing failed: {e}", exc_info=True)
        return False


def preprocess_wesad_wrapper(
    data_dir: Path, args, logger
) -> bool:
    """Preprocess WESAD dataset (features-only)."""
    print("\n" + "="*80)
    print("WESAD PREPROCESSING (FEATURES-ONLY)")
    print("="*80)

    wesad_raw = Path(data_dir) / 'raw' / 'wesad'
    wesad_processed = Path(data_dir) / 'processed' / 'wesad'

    if not wesad_raw.exists():
        print(f"ERROR: WESAD raw data not found: {wesad_raw}")
        return False

    try:
        print(f"\nConfiguration:")
        print(f"   Type: Features-only (140D from sliding windows)")
        print(f"   Window: {args.wesad_window_size/args.wesad_target_freq:.1f}s "
              f"@ {args.wesad_target_freq}Hz")
        print(f"   Overlap: {args.wesad_overlap*100:.0f}%")
        print(f"   Features: 10 per channel × 14 channels = 140D")
        print(f"   Split: {(1-args.test_size-args.val_size)*100:.0f}% train / "
              f"{args.val_size*100:.0f}% val / {args.test_size*100:.0f}% test")
        print(f"   Workers: {args.n_workers}")
        print_memory("Start")

        print(f"\nProcessing...")
        start = time.time()

        info = preprocess_wesad(
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
        print(f"Completed in {elapsed:.1f}s")
        print_memory("End")

        if info:
            print(f"\nSummary:")
            print(f"   Features: 140D")
            print(f"   Classes: {info.get('n_classes')} {info.get('class_names')}")
            print(f"   Train: {info.get('train_size')} | "
                  f"Val: {info.get('val_size')} | Test: {info.get('test_size')}")
            print(f"   Size: {info.get('total_size_mb', 0):.1f} MB")

        logger.info(f"WESAD preprocessing completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"WESAD preprocessing failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing pipeline for Sleep-EDF and WESAD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DATASETS:
  sleep-edf:   Features-only (24D per epoch, ~3.4 MB)
  wesad:       Features-only (140D from windows, ~2.5 MB)

EXAMPLES:
  # Preprocess all
  python scripts/preprocess_all.py

  # Only Sleep-EDF
  python scripts/preprocess_all.py --dataset sleep-edf

  # Only WESAD
  python scripts/preprocess_all.py --dataset wesad

  # Check status
  python scripts/preprocess_all.py --check_only

  # Force reprocess
  python scripts/preprocess_all.py --force_reprocess
        """)

    parser.add_argument(
        '--dataset',
        choices=['all', 'sleep-edf', 'wesad'],
        default='all',
        help='Which dataset to preprocess'
    )

    parser.add_argument(
        '--data_dir',
        default='./data',
        help='Base data directory'
    )

    # WESAD arguments
    parser.add_argument(
        '--wesad_target_freq',
        type=int,
        default=32,
        help='WESAD target frequency (Hz)'
    )

    parser.add_argument(
        '--wesad_window_size',
        type=int,
        default=1024,
        help='WESAD window size (samples)'
    )

    parser.add_argument(
        '--wesad_overlap',
        type=float,
        default=0.75,
        help='WESAD window overlap'
    )

    parser.add_argument(
        '--wesad_label_threshold',
        type=float,
        default=0.7,
        help='WESAD label threshold'
    )

    # Common arguments
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Test set size'
    )

    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help='Validation set size'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--n_workers',
        type=int,
        default=0,
        help='Parallel workers (0=auto)'
    )

    parser.add_argument(
        '--force_reprocess',
        action='store_true',
        help='Force reprocessing'
    )

    parser.add_argument(
        '--check_only',
        action='store_true',
        help='Check status without processing'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Auto-detect workers
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

    print("\nAvailable datasets:")
    print("  • sleep-edf:  24D features (3.4 MB)")
    print("  • wesad:      140D features (2.5 MB)")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_status = verify_raw_data(data_dir)

    if not args.force_reprocess:
        processed_status = verify_processed_data(data_dir)
    else:
        processed_status = {
            'sleep-edf': False,
            'wesad': False
        }
        print("\nForce reprocess enabled")

    if args.check_only:
        print("\nStatus check complete")
        return 0

    print("\n" + "="*80)
    print("Starting preprocessing...")
    print("="*80)

    start_time = time.time()
    results = {}

    # Determine what to process
    datasets_to_process = []

    if args.dataset in ['all', 'sleep-edf']:
        if args.force_reprocess or not processed_status.get(
            'sleep-edf', False
        ):
            if raw_status['sleep-edf']:
                datasets_to_process.append('sleep-edf')
            else:
                print("SKIP: sleep-edf (raw data not found)")
        else:
            print("SKIP: sleep-edf (already processed)")

    if args.dataset in ['all', 'wesad']:
        if args.force_reprocess or not processed_status.get('wesad', False):
            if raw_status['wesad']:
                datasets_to_process.append('wesad')
            else:
                print("SKIP: wesad (raw data not found)")
        else:
            print("SKIP: wesad (already processed)")

    if not datasets_to_process:
        print("Nothing to process")
        return 0

    # Process
    if 'sleep-edf' in datasets_to_process:
        results['sleep-edf'] = preprocess_sleep_edf_wrapper(
            data_dir, args, logger
        )

    if 'wesad' in datasets_to_process:
        results['wesad'] = preprocess_wesad_wrapper(
            data_dir, args, logger
        )

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)

    print("\nResults:")
    for dataset, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"   [{status}] {dataset}")

    print(f"\nPerformance:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Final memory: {get_memory_usage():.0f} MB")

    print(f"\nData locations:")
    for dataset in ['sleep-edf', 'wesad']:
        dataset_path = Path(args.data_dir) / 'processed' / dataset
        if dataset_path.exists():
            train_file = dataset_path / 'X_train.npy'
            if train_file.exists():
                data = __import__('numpy').load(train_file)
                size = data.nbytes / (1024**2)
                print(f"   {dataset:15s}: {data.shape} ({size:.1f} MB)")

    print("\nNext: python scripts/run_experiments.py --scenario baseline")
    print("="*80 + "\n")

    if any(r is False for r in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())