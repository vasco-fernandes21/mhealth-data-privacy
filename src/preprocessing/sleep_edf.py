#!/usr/bin/env python3
"""
Sleep-EDF Dataset Preprocessing 
- Non-overlapping temporal windows based on epochs
- imap_unordered with streaming
- Pre-allocation of arrays
- Pre-computed windows for fast loading
"""

import numpy as np
import pyedflib
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import glob
import time
import gc
from typing import Tuple, Dict, List, Optional
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename."""
    basename = os.path.basename(filename)
    return basename[:6]


def load_sleep_edf_expanded_hypnogram(
    hypno_file: str,
    target_epoch_duration: int = 30
) -> Optional[np.ndarray]:
    """Load and convert hypnogram to 30-second epochs."""
    if not os.path.exists(hypno_file):
        return None
    
    try:
        f = pyedflib.EdfReader(hypno_file)
        annotations = f.read_annotation()
        f.close()
        
        sleep_stages = []
        
        for onset, duration, description in annotations:
            desc_str = description.decode('utf-8') \
                if isinstance(description, bytes) else str(description)
            
            if 'Sleep stage' not in desc_str:
                continue
            
            if 'Sleep stage W' in desc_str:
                stage = 0
            elif 'Sleep stage 1' in desc_str:
                stage = 1
            elif 'Sleep stage 2' in desc_str:
                stage = 2
            elif 'Sleep stage 3' in desc_str or 'Sleep stage 4' in desc_str:
                stage = 3
            elif 'Sleep stage R' in desc_str:
                stage = 4
            else:
                continue
            
            try:
                duration_sec = int(duration) \
                    if isinstance(duration, int) \
                    else int(duration.decode('utf-8'))
            except:
                duration_sec = 30
            
            n_epochs = int(duration_sec / target_epoch_duration)
            sleep_stages.extend([stage] * n_epochs)
            
            remainder = duration_sec % target_epoch_duration
            if remainder >= target_epoch_duration / 2:
                sleep_stages.append(stage)
        
        return np.array(sleep_stages, dtype=np.uint8)
        
    except Exception as e:
        print(f"    WARNING: Error loading hypnogram: {e}")
        return None


def load_sleep_edf_signals(
    psg_file: str
) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """Load raw signals from Sleep-EDF file."""
    if not os.path.exists(psg_file):
        return None
    
    try:
        f = pyedflib.EdfReader(psg_file)
        
        signals = []
        channel_labels = []
        
        for i in range(min(3, f.signals_in_file)):
            signals.append(f.readSignal(i))
            channel_labels.append(f.getLabel(i))
        
        sample_freq = f.getSampleFrequency(0)
        f.close()
        
        return np.array(signals, dtype=np.float32), sample_freq, \
            channel_labels
        
    except Exception as e:
        print(f"    WARNING: Error loading signals: {e}")
        return None


def filter_signals(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """Apply Butterworth bandpass filters to raw signals."""
    if signals.shape[1] < 100:
        return signals.astype(np.float32)
    
    filtered = np.zeros_like(signals, dtype=np.float32)
    
    try:
        for i in range(signals.shape[0]):
            try:
                sos = signal.butter(
                    3, [0.5, 32 if i < 2 else 10],
                    btype='band', fs=sfreq, output='sos'
                )
                filtered[i] = signal.sosfiltfilt(
                    sos, signals[i]
                ).astype(np.float32)
            except:
                filtered[i] = signals[i].astype(np.float32)
    except ValueError:
        return signals.astype(np.float32)
    
    return filtered


def segment_into_epochs(
    signals: np.ndarray, labels: np.ndarray,
    sfreq: float, epoch_duration: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment raw signals into 30-second epochs."""
    n_samples_epoch = int(sfreq * epoch_duration)
    n_epochs = signals.shape[1] // n_samples_epoch
    n_available_labels = len(labels)
    n_epochs_to_use = min(n_epochs, n_available_labels)
    
    if n_epochs_to_use == 0:
        return np.empty(
            (0, signals.shape[0], n_samples_epoch), dtype=np.float32
        ), np.array([], dtype=np.uint8)
    
    signal_data = signals[:, :n_epochs_to_use * n_samples_epoch]
    epochs = signal_data.reshape(
        signals.shape[0], n_epochs_to_use, n_samples_epoch
    )
    epochs = epochs.transpose(1, 0, 2)
    
    epoch_labels = labels[:n_epochs_to_use]
    
    return epochs.astype(np.float32), epoch_labels


def process_single_sleep_edf_file(
    args: Tuple
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Process a single Sleep-EDF file in parallel."""
    psg_file, hypno_file, file_idx, total_files = args
    
    psg_name = os.path.basename(psg_file)
    print(f"  [{file_idx}/{total_files}] {psg_name}...", end=' ', flush=True)
    
    try:
        result = load_sleep_edf_signals(psg_file)
        if result is None:
            print("FAILED")
            return None, None, None
        signals, sfreq, channels = result
        
        labels = load_sleep_edf_expanded_hypnogram(hypno_file)
        if labels is None:
            print("FAILED")
            return None, None, None
        
        filtered_signals = filter_signals(signals, sfreq)
        epochs, epoch_labels = segment_into_epochs(
            filtered_signals, labels, sfreq
        )
        
        if len(epochs) == 0:
            print("FAILED (no epochs)")
            return None, None, None
        
        subject_id = psg_name.replace('-PSG.edf', '')
        
        print(f"OK ({len(epochs)} epochs)")
        return epochs, epoch_labels, subject_id
        
    except Exception as e:
        print(f"FAILED ({e})")
        return None, None, None


def normalize_per_channel_chunked(
    X_train: np.ndarray, X_val: np.ndarray,
    X_test: np.ndarray, chunk_size: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize per-channel in chunks to save memory."""
    print("  Normalizing per-channel (chunked)...")
    n_epochs_train, n_channels, samples_per_epoch = X_train.shape
    
    scaler = StandardScaler()
    
    print(f"    Fitting scaler on {n_epochs_train} training epochs...")
    for i in range(0, n_epochs_train, chunk_size):
        end_idx = min(i + chunk_size, n_epochs_train)
        X_chunk_flat = X_train[i:end_idx].reshape(
            -1, n_channels * samples_per_epoch
        )
        scaler.partial_fit(X_chunk_flat)
        del X_chunk_flat
        gc.collect()
    
    print(f"    Transforming training data...")
    X_train_norm = np.zeros_like(X_train, dtype=np.float32)
    for i in range(0, n_epochs_train, chunk_size):
        end_idx = min(i + chunk_size, n_epochs_train)
        X_chunk_flat = X_train[i:end_idx].reshape(
            -1, n_channels * samples_per_epoch
        )
        X_train_norm[i:end_idx] = scaler.transform(X_chunk_flat).reshape(
            X_train[i:end_idx].shape
        ).astype(np.float32)
        del X_chunk_flat
        gc.collect()
    
    print(f"    Transforming validation data...")
    X_val_flat = X_val.reshape(-1, n_channels * samples_per_epoch)
    X_val_norm = scaler.transform(X_val_flat).reshape(
        X_val.shape
    ).astype(np.float32)
    del X_val_flat
    gc.collect()
    
    print(f"    Transforming test data...")
    X_test_flat = X_test.reshape(-1, n_channels * samples_per_epoch)
    X_test_norm = scaler.transform(X_test_flat).reshape(
        X_test.shape
    ).astype(np.float32)
    del X_test_flat
    gc.collect()
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def split_by_subjects(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    test_size: float = 0.15, val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Subject-wise split into train/val/test."""
    unique_subjects = sorted(list(set(subjects)))
    
    subj_trainval, subj_test = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )
    
    val_size_adj = val_size / (1 - test_size)
    subj_train, subj_val = train_test_split(
        subj_trainval, test_size=val_size_adj, random_state=random_state
    )
    
    train_mask = np.isin(subjects, subj_train)
    val_mask = np.isin(subjects, subj_val)
    test_mask = np.isin(subjects, subj_test)
    
    train_dict = {
        'X': X[train_mask],
        'y': y[train_mask],
        'subjects': subjects[train_mask]
    }
    
    val_dict = {
        'X': X[val_mask],
        'y': y[val_mask],
        'subjects': subjects[val_mask]
    }
    
    test_dict = {
        'X': X[test_mask],
        'y': y[test_mask],
        'subjects': subjects[test_mask]
    }
    
    split_info = {
        'train_subjects': subj_train,
        'val_subjects': subj_val,
        'test_subjects': subj_test,
        'n_train_epochs': len(train_dict['X']),
        'n_val_epochs': len(val_dict['X']),
        'n_test_epochs': len(test_dict['X'])
    }
    
    return train_dict, val_dict, test_dict, split_info


def create_windows_from_epochs(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    window_epochs: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping windows from epochs."""
    n_epochs = X.shape[0]
    n_channels, samples_per_epoch = X.shape[1:]
    
    n_windows = n_epochs // window_epochs
    n_epochs_used = n_windows * window_epochs
    
    if n_windows <= 0:
        return (
            np.empty(
                (0, window_epochs, n_channels * samples_per_epoch),
                dtype=np.float32
            ),
            np.array([], dtype=y.dtype),
            np.array([], dtype=object)
        )
    
    print(f"    Creating {n_windows} windows from {n_epochs} epochs")
    
    X_used = X[:n_epochs_used]
    y_used = y[:n_epochs_used]
    subjects_used = subjects[:n_epochs_used]
    
    X_reshaped = X_used.reshape(
        n_windows, window_epochs, n_channels, samples_per_epoch
    )
    
    X_windows = X_reshaped.reshape(
        n_windows, window_epochs, n_channels * samples_per_epoch
    ).astype(np.float32)
    
    y_reshaped = y_used.reshape(n_windows, window_epochs)
    y_windows = y_reshaped[:, -1]
    
    subjects_reshaped = subjects_used.reshape(n_windows, window_epochs)
    subj_windows = subjects_reshaped[:, -1]
    
    print(f"    Final: {n_windows} windows (dropped {n_epochs - n_epochs_used} epochs)")
    
    gc.collect()
    
    return X_windows, y_windows, subj_windows


def preprocess_sleep_edf(
    data_dir: str,
    output_dir: str,
    window_epochs: int = 10,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_workers: int = None,
    force_reprocess: bool = False
) -> Dict:
    """
    Complete Sleep-EDF preprocessing pipeline.
    Creates pre-computed windows for fast loading.
    """
    print("="*70)
    print("SLEEP-EDF PREPROCESSING (PRE-COMPUTED WINDOWS)")
    print("="*70 + "\n")
    
    start_time_total = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy',
        'preprocessing_info.pkl', 'scaler.pkl'
    ]
    
    all_exist = all(
        os.path.exists(os.path.join(output_dir, f)) for f in required_files
    )
    
    # Clean up old files if force_reprocess
    if force_reprocess:
        print("Force reprocess enabled - cleaning up old files...")
        old_files = [
            # Current format (windows)
            'X_train.npy', 'X_val.npy', 'X_test.npy',
            'y_train.npy', 'y_val.npy', 'y_test.npy',
            'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy',
            # Old format (epochs - if switching back)
            'X_train_epochs.npy', 'X_val_epochs.npy', 'X_test_epochs.npy',
            'y_train_epochs.npy', 'y_val_epochs.npy', 'y_test_epochs.npy',
            'subjects_train_epochs.npy', 'subjects_val_epochs.npy', 'subjects_test_epochs.npy',
            # Very old format (windows with _windows suffix)
            'X_train_windows.npy', 'X_val_windows.npy', 'X_test_windows.npy',
            'y_train_windows.npy', 'y_val_windows.npy', 'y_test_windows.npy',
            'subjects_train_windows.npy', 'subjects_val_windows.npy', 'subjects_test_windows.npy',
            # Metadata files
            'preprocessing_info.pkl', 'scaler.pkl', 'metadata.pkl'
        ]
        
        removed_count = 0
        for f in old_files:
            filepath = os.path.join(output_dir, f)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Could not remove {f}: {e}")
        
        if removed_count > 0:
            print(f"  Removed {removed_count} old file(s)\n")
        else:
            print("  No old files to remove\n")
    
    if all_exist and not force_reprocess:
        print(f"Already preprocessed! Loading metadata...\n")
        info = joblib.load(
            os.path.join(output_dir, 'preprocessing_info.pkl')
        )
        return info
    
    print("Matching PSG-Hypnogram files...")
    psg_files = sorted(glob.glob(
        os.path.join(data_dir, '**/*-PSG.edf'), recursive=True
    ))
    hypno_files = sorted(glob.glob(
        os.path.join(data_dir, '**/*-Hypnogram.edf'), recursive=True
    ))
    
    print(f"Found {len(psg_files)} PSG files, {len(hypno_files)} Hypnogram files")
    
    psg_by_subject = {extract_subject_id(f): f for f in psg_files}
    hypno_by_subject = {extract_subject_id(f): f for f in hypno_files}
    
    matched_subjects = set(psg_by_subject.keys()) & \
        set(hypno_by_subject.keys())
    file_pairs = [
        (psg_by_subject[s], hypno_by_subject[s])
        for s in sorted(matched_subjects)
    ]
    
    print(f"Matched {len(file_pairs)}/{max(len(psg_files), len(hypno_files))} pairs\n")
    
    if not file_pairs:
        raise RuntimeError(
            f"No matching PSG-Hypnogram pairs found in {data_dir}"
        )
    
    file_pairs_with_idx = [
        (p, h, i+1, len(file_pairs)) for i, (p, h) in enumerate(file_pairs)
    ]
    
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    print(f"Processing with {n_workers} workers (imap_unordered)...\n")
    start_time = time.time()
    
    all_epochs = []
    all_labels = []
    all_subjects = []
    
    with Pool(processes=n_workers) as pool:
        for idx, (epochs, labels, subject_id) in enumerate(
            pool.imap_unordered(
                process_single_sleep_edf_file, file_pairs_with_idx,
                chunksize=2
            )
        ):
            if epochs is not None and labels is not None:
                all_epochs.append(epochs)
                all_labels.append(labels)
                all_subjects.extend([subject_id] * len(epochs))
            
            if (idx + 1) % 2 == 0:
                gc.collect()
    
    elapsed_processing = time.time() - start_time
    print(f"\nProcessed in {elapsed_processing:.1f}s\n")
    
    print("Combining epochs (pre-allocation)...")
    total_epochs = sum(len(e) for e in all_epochs)
    n_channels, samples_per_epoch = all_epochs[0].shape[1:]
    
    X_epochs = np.zeros(
        (total_epochs, n_channels, samples_per_epoch), dtype=np.float32
    )
    y_epochs = np.zeros(total_epochs, dtype=np.uint8)
    
    idx = 0
    for epochs, labels in zip(all_epochs, all_labels):
        n = len(epochs)
        X_epochs[idx:idx+n] = epochs
        y_epochs[idx:idx+n] = labels
        idx += n
        del epochs, labels
    
    subjects_epochs = np.array(all_subjects, dtype=object)
    
    del all_epochs, all_labels, all_subjects
    gc.collect()
    
    print(f"Combined: {X_epochs.shape}\n")
    
    print(f"Label distribution:")
    unique, counts = np.unique(y_epochs, return_counts=True)
    stages = ['W', 'N1', 'N2', 'N3', 'R']
    for u, c in zip(unique, counts):
        print(f"  {stages[u]}: {c}")
    print()
    
    print(f"Subject-wise split...")
    unique_subjects = sorted(list(set(subjects_epochs)))
    print(f"Total subjects: {len(unique_subjects)}\n")
    
    train_dict, val_dict, test_dict, split_info = split_by_subjects(
        X_epochs, y_epochs, subjects_epochs, test_size, val_size,
        random_state
    )
    
    print(f"Split summary (epochs):")
    print(f"  Train: {split_info['n_train_epochs']} epochs")
    print(f"  Val:   {split_info['n_val_epochs']} epochs")
    print(f"  Test:  {split_info['n_test_epochs']} epochs\n")
    
    del X_epochs, y_epochs, subjects_epochs
    gc.collect()
    
    print("Normalizing (chunked)...")
    X_train_norm, X_val_norm, X_test_norm, scaler = \
        normalize_per_channel_chunked(
            train_dict['X'], val_dict['X'], test_dict['X'], chunk_size=500
        )
    print("Normalized\n")
    
    del train_dict['X'], val_dict['X'], test_dict['X']
    gc.collect()
    
    print("Creating windows...")
    X_train_win, y_train_win, subj_train_win = create_windows_from_epochs(
        X_train_norm, train_dict['y'], train_dict['subjects'], window_epochs
    )
    X_val_win, y_val_win, subj_val_win = create_windows_from_epochs(
        X_val_norm, val_dict['y'], val_dict['subjects'], window_epochs
    )
    X_test_win, y_test_win, subj_test_win = create_windows_from_epochs(
        X_test_norm, test_dict['y'], test_dict['subjects'], window_epochs
    )
    print()
    
    del X_train_norm, X_val_norm, X_test_norm
    del train_dict, val_dict, test_dict
    gc.collect()
    
    print(f"Saving windows to {output_dir}...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_win)
    print("  Saved X_train.npy")
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_win)
    print("  Saved y_train.npy")
    np.save(os.path.join(output_dir, 'subjects_train.npy'), subj_train_win)
    print("  Saved subjects_train.npy")

    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_win)
    print("  Saved X_val.npy")
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val_win)
    print("  Saved y_val.npy")
    np.save(os.path.join(output_dir, 'subjects_val.npy'), subj_val_win)
    print("  Saved subjects_val.npy")

    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_win)
    print("  Saved X_test.npy")
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test_win)
    print("  Saved y_test.npy")
    np.save(os.path.join(output_dir, 'subjects_test.npy'), subj_test_win)
    print("  Saved subjects_test.npy")

    # Save metadata before deleting variables
    n_windows_train = len(X_train_win)
    n_windows_val = len(X_val_win)
    n_windows_test = len(X_test_win)
    window_shape = X_train_win.shape[1:]
    
    del X_train_win, X_val_win, X_test_win
    del y_train_win, y_val_win, y_test_win
    gc.collect()
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("  Saved scaler.pkl")
    print()
    
    elapsed_total = time.time() - start_time_total
    metadata = {
        'n_windows_train': n_windows_train,
        'n_windows_val': n_windows_val,
        'n_windows_test': n_windows_test,
        'window_shape': window_shape,
        'window_epochs': window_epochs,
        'n_channels': n_channels,
        'samples_per_epoch': samples_per_epoch,
        'sfreq': 100,
        'epoch_duration_s': 30,
        'n_classes': 5,
        'class_names': ['W', 'N1', 'N2', 'N3', 'R'],
        'split_type': 'subject-wise',
        'window_type': 'pre_computed_non_overlapping',
        'train_subjects': [str(s) for s in split_info['train_subjects']],
        'val_subjects': [str(s) for s in split_info['val_subjects']],
        'test_subjects': [str(s) for s in split_info['test_subjects']],
        'total_subjects': len(unique_subjects),
        'n_workers': n_workers,
        'processing_time_s': elapsed_processing,
        'total_time_s': elapsed_total
    }
    
    joblib.dump(
        metadata, os.path.join(output_dir, 'preprocessing_info.pkl')
    )
    
    print("="*70)
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_total:.1f}s (processing: {elapsed_processing:.1f}s)\n")
    
    return metadata


def load_windowed_sleep_edf(data_dir: str) -> Tuple:
    """Load preprocessed Sleep-EDF windowed data."""
    print(f"Loading Sleep-EDF from {data_dir}...\n")

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    # Subjects are object arrays (strings), need allow_pickle=True
    subjects_train = np.load(
        os.path.join(data_dir, 'subjects_train.npy'), allow_pickle=True
    )

    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    subjects_val = np.load(
        os.path.join(data_dir, 'subjects_val.npy'), allow_pickle=True
    )

    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    subjects_test = np.load(
        os.path.join(data_dir, 'subjects_test.npy'), allow_pickle=True
    )

    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    
    # Check if preprocessing_info.pkl exists
    info_path = os.path.join(data_dir, 'preprocessing_info.pkl')
    if os.path.exists(info_path):
        info = joblib.load(info_path)
    else:
        print("  WARNING: preprocessing_info.pkl not found, using defaults")
        info = {
            'window_epochs': 10,
            'n_channels': 3,
            'samples_per_epoch': 3000,
            'n_classes': 5,
            'class_names': ['W', 'N1', 'N2', 'N3', 'R']
        }

    print(f"Loaded:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}\n")

    subjects = np.concatenate([subjects_train, subjects_val, subjects_test])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, \
        info, subjects


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess Sleep-EDF dataset'
    )
    parser.add_argument('--data_dir', default='data/raw/sleep-edf')
    parser.add_argument('--output_dir', default='data/processed/sleep-edf')
    parser.add_argument('--window_epochs', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--force_reprocess', action='store_true')
    
    args = parser.parse_args()
    
    info = preprocess_sleep_edf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_epochs=args.window_epochs,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_workers=args.n_workers,
        force_reprocess=args.force_reprocess
    )