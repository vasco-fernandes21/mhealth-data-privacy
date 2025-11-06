#!/usr/bin/env python3
"""
WESAD Dataset Preprocessing (OPTIMIZED VERSION)

Improvements:
- window_size=1024 (32s context)
- Per-subject normalization (removes inter-individual bias)
- Balanced class handling
- Subject-wise splitting
"""

import numpy as np
import pickle
from scipy import signal
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import glob
from typing import Tuple, Dict, List
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_wesad_file(pkl_path: str) -> Dict:
    """Load a single WESAD pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    
    return {
        'ecg': chest['ECG'],
        'eda_chest': chest['EDA'],
        'temp_chest': chest['Temp'],
        'acc_chest': chest['ACC'],
        'emg': chest['EMG'],
        'resp': chest['Resp'],
        'acc_wrist': wrist['ACC'],
        'bvp': wrist['BVP'],
        'eda_wrist': wrist['EDA'],
        'temp_wrist': wrist['TEMP'],
        'labels': data['label'],
        'subject': data.get('subject', 'unknown'),
        'sfreq': {
            'chest': 700,
            'acc_wrist': 32,
            'bvp': 64,
            'eda_wrist': 4,
            'temp_wrist': 4
        }
    }


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def resample_signal(arr: np.ndarray, orig_freq: int,
                   target_freq: int) -> np.ndarray:
    """Resample signal using polyphase filtering."""
    if arr.ndim == 1:
        return signal.resample_poly(arr, int(target_freq), int(orig_freq),
                                   axis=0).astype(np.float32)
    resampled_cols = []
    for c in range(arr.shape[1]):
        resampled_cols.append(
            signal.resample_poly(arr[:, c], int(target_freq),
                               int(orig_freq), axis=0).astype(np.float32)
        )
    return np.stack(resampled_cols, axis=1)


def downsample_labels_temporal(labels: np.ndarray, orig_freq: int,
                               target_freq: int) -> np.ndarray:
    """Downsample labels using temporal mapping."""
    orig_length = len(labels)
    target_length = int(np.round(orig_length * (target_freq / orig_freq)))
    indices = np.round(
        np.linspace(0, orig_length - 1, target_length)
    ).astype(int)
    return labels[indices]


def resample_all_signals(data: Dict, target_freq: int = 32) -> Dict:
    """Resample all signals to target frequency."""
    sfreq = data['sfreq']
    resampled = {}
    
    # Chest signals (700 Hz → target_freq)
    resampled['ecg'] = resample_signal(data['ecg'], sfreq['chest'], target_freq)
    resampled['eda_chest'] = resample_signal(data['eda_chest'], sfreq['chest'], target_freq)
    resampled['temp_chest'] = resample_signal(data['temp_chest'], sfreq['chest'], target_freq)
    resampled['acc_chest'] = resample_signal(data['acc_chest'], sfreq['chest'], target_freq)
    resampled['emg'] = resample_signal(data['emg'], sfreq['chest'], target_freq)
    resampled['resp'] = resample_signal(data['resp'], sfreq['chest'], target_freq)
    
    # Wrist signals
    resampled['acc_wrist'] = resample_signal(data['acc_wrist'], sfreq['acc_wrist'], target_freq)
    resampled['bvp'] = resample_signal(data['bvp'], sfreq['bvp'], target_freq)
    resampled['eda_wrist'] = resample_signal(data['eda_wrist'], sfreq['eda_wrist'], target_freq)
    resampled['temp_wrist'] = resample_signal(data['temp_wrist'], sfreq['temp_wrist'], target_freq)
    
    # Labels
    resampled['labels'] = downsample_labels_temporal(data['labels'],
                                                    sfreq['chest'], target_freq)
    
    resampled['subject'] = data['subject']
    resampled['sfreq'] = target_freq
    
    return resampled


def apply_filters(resampled: Dict, target_freq: int = 32) -> Dict:
    """Apply bandpass/lowpass filters."""
    filtered = resampled.copy()
    
    try:
        sos_ecg = signal.butter(4, [0.5, 15], btype='band', fs=target_freq, output='sos')
        sos_bvp = signal.butter(4, [0.5, 12], btype='band', fs=target_freq, output='sos')
        sos_acc = signal.butter(4, [0.1, 2], btype='band', fs=target_freq, output='sos')
        sos_eda = signal.butter(4, 1.5, btype='low', fs=target_freq, output='sos')
        sos_resp = signal.butter(4, [0.1, 0.5], btype='band', fs=target_freq, output='sos')
    except Exception as e:
        print(f"Warning: filter design failed ({e})")
        return resampled
    
    signal_filters = {
        'ecg': sos_ecg, 'bvp': sos_bvp, 'acc_chest': sos_acc,
        'acc_wrist': sos_acc, 'eda_chest': sos_eda, 'eda_wrist': sos_eda,
        'resp': sos_resp, 'emg': sos_ecg, 'temp_chest': sos_eda, 'temp_wrist': sos_eda
    }
    
    for key, sos_filter in signal_filters.items():
        if key in filtered and filtered[key] is not None:
            try:
                filtered[key] = signal.sosfiltfilt(sos_filter, filtered[key], axis=0)
            except Exception:
                pass
    
    return filtered


def reduce_outliers(signals: Dict, lower_p: float = 1.0,
                   upper_p: float = 99.0) -> Dict:
    """Clip outliers using percentile method."""
    clipped = signals.copy()
    keys = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp',
            'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']
    
    for k in keys:
        if k in clipped and clipped[k] is not None:
            arr = clipped[k]
            if arr.ndim == 1:
                lo, hi = np.percentile(arr, lower_p), np.percentile(arr, upper_p)
                clipped[k] = np.clip(arr, lo, hi)
            else:
                for c in range(arr.shape[1]):
                    lo, hi = np.percentile(arr[:, c], lower_p), np.percentile(arr[:, c], upper_p)
                    arr[:, c] = np.clip(arr[:, c], lo, hi)
    
    return clipped


# ============================================================================
# WINDOWING
# ============================================================================

def create_temporal_windows(data: Dict, window_size: int = 1024,
                           overlap: float = 0.5,
                           label_threshold: float = 0.7) -> Tuple[
                               np.ndarray, np.ndarray, List[str], int, np.ndarray]:
    """Create temporal sliding windows with strict label assignment."""
    step_size = int(window_size * (1 - overlap))
    ref_length = len(data['ecg'])
    
    signal_names = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg',
                    'resp', 'bvp', 'eda_wrist', 'temp_wrist', 'acc_wrist']
    
    # Build channel names
    channel_names = []
    for name in signal_names:
        if name not in data:
            continue
        arr = data[name]
        if arr.ndim > 1 and arr.shape[1] > 1:
            for ax in ['x', 'y', 'z'][:arr.shape[1]]:
                channel_names.append(f"{name}_{ax}")
        else:
            channel_names.append(name)
    
    n_channels = len(channel_names)
    windows_list = []
    labels_list = []
    subjects_list = []
    n_dropped = 0
    
    for start_idx in range(0, ref_length - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        if not all(len(data.get(name, [])) >= end_idx for name in signal_names if name in data):
            continue
        
        # Extract window
        window_data = np.zeros((n_channels, window_size), dtype=np.float32)
        ch_idx = 0
        
        for name in signal_names:
            if name not in data:
                continue
            arr = data[name]
            
            if arr.ndim > 1 and arr.shape[1] > 1:
                for c in range(arr.shape[1]):
                    window_data[ch_idx] = arr[start_idx:end_idx, c]
                    ch_idx += 1
            else:
                series = arr if arr.ndim == 1 else arr[:, 0]
                window_data[ch_idx] = series[start_idx:end_idx]
                ch_idx += 1
        
        # Strict label assignment
        window_labels = data['labels'][start_idx:end_idx]
        valid_labels = window_labels[window_labels != 0]
        
        if len(valid_labels) < window_size * label_threshold:
            n_dropped += 1
            continue
        
        if len(valid_labels) > 0:
            window_label = np.bincount(valid_labels.astype(int)).argmax()
        else:
            n_dropped += 1
            continue
        
        windows_list.append(window_data)
        labels_list.append(window_label)
        subjects_list.append(data['subject'])
    
    return (np.array(windows_list, dtype=np.float32), 
            np.array(labels_list, dtype=np.uint8),
            channel_names, n_dropped,
            np.array(subjects_list, dtype=object))


# ============================================================================
# NORMALIZATION (PER-SUBJECT)
# ============================================================================

def zscore_per_subject(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Normalize per-subject to remove inter-individual bias."""
    Xn = np.zeros_like(X, dtype=np.float32)
    
    for subj in np.unique(subjects):
        mask = subjects == subj
        X_subj = X[mask]
        
        # Per-channel z-score
        mean = X_subj.mean(axis=(0, 2), keepdims=True)
        std = X_subj.std(axis=(0, 2), keepdims=True) + 1e-8
        
        Xn[mask] = (X_subj - mean) / std
    
    return Xn


def compute_normalization_stats(X_train: np.ndarray) -> Dict:
    """Compute per-channel z-score normalization from training set."""
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)
    train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    return {
        'type': 'per_channel_zscore',
        'mean': train_mean,
        'std': train_std,
    }


def apply_normalization(X: np.ndarray, stats: Dict) -> np.ndarray:
    """Apply saved normalization statistics."""
    return (X - stats['mean']) / stats['std']


# ============================================================================
# CLASS IMBALANCE
# ============================================================================

def compute_class_weights(y: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """Compute balanced class weights."""
    class_counts = np.bincount(y.astype(int))
    n_classes = len(class_counts)
    n_samples = len(y)
    
    weights = {}
    for c in range(n_classes):
        if class_counts[c] > 0:
            weights[c] = n_samples / (n_classes * class_counts[c])
        else:
            weights[c] = 1.0
    
    return weights, class_counts


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_wesad_file(args: Tuple) -> Tuple[Dict, str]:
    """Process a single WESAD file in parallel."""
    pkl_file, target_freq, window_size, overlap, label_threshold = args
    
    try:
        data = load_wesad_file(pkl_file)
        subject = data['subject']
        
        resampled = resample_all_signals(data, target_freq)
        outlier_reduced = reduce_outliers(resampled)
        filtered = apply_filters(outlier_reduced, target_freq)
        
        windows, labels, ch_names, n_dropped, subjs = create_temporal_windows(
            filtered, window_size, overlap, label_threshold
        )
        
        return {
            'windows': windows,
            'labels': labels,
            'subjects': subjs,
            'channel_names': ch_names,
            'n_dropped': n_dropped
        }, subject
    
    except Exception as e:
        print(f"  Error: {os.path.basename(pkl_file)}: {e}")
        return None, None


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def preprocess_wesad_temporal(
    data_dir: str,
    output_dir: str,
    target_freq: int = 32,
    window_size: int = 1024,  
    overlap: float = 0.5,
    label_threshold: float = 0.7,
    binary: bool = True,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
    n_workers: int = None,
    force_reprocess: bool = False) -> Dict:
    """
    Preprocess WESAD dataset (OPTIMIZED).
    
    Pipeline:
    1. Load raw data from pickle files
    2. Resample to 32 Hz
    3. Apply bandpass filtering
    4. Create sliding windows (1024 samples = 32s)
    5. Subject-wise train/val/test split
    6. Per-subject z-score normalization
    7. Per-channel global normalization (on training set)
    """
    print("="*70)
    print("WESAD PREPROCESSING (OPTIMIZED)")
    print("="*70)
    print(f"Target freq: {target_freq} Hz")
    print(f"Window: {window_size} samples ({window_size/target_freq:.1f}s)")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Split: Train {(1-test_size-val_size)*100:.0f}% | "
          f"Val {val_size*100:.0f}% | Test {test_size*100:.0f}%\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already processed
    required_files = ['X_train.npy', 'X_val.npy', 'X_test.npy',
                     'y_train.npy', 'y_val.npy', 'y_test.npy']
    all_exist = all(os.path.exists(os.path.join(output_dir, f))
                   for f in required_files)
    
    if all_exist and not force_reprocess:
        print(f"⏭️  Data already preprocessed!")
        try:
            info = joblib.load(os.path.join(output_dir, "metadata.pkl"))
            return info
        except:
            return {}
    
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    pkl_files = glob.glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} pickle files\n")
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {data_dir}")
    
    # Parallel processing
    process_args = [
        (pkl_file, target_freq, window_size, overlap, label_threshold)
        for pkl_file in pkl_files
    ]
    
    print(f"Processing with {n_workers} workers...")
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_wesad_file, process_args)
    
    elapsed_processing = time.time() - start_time
    
    # Combine results
    all_windows = []
    all_labels = []
    all_subjects = []
    total_dropped = 0
    channel_names = None
    
    for result, subject in results:
        if result is not None:
            if channel_names is None:
                channel_names = result['channel_names']
            
            all_windows.append(result['windows'])
            all_labels.append(result['labels'])
            all_subjects.extend(result['subjects'])
            total_dropped += result['n_dropped']
    
    print(f"Processed in {elapsed_processing:.1f}s")
    print(f"Windows dropped: {total_dropped}\n")
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subjects_arr = np.array(all_subjects, dtype=object)
    
    print(f"Total windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Channels: {len(channel_names)}\n")
    
    # Binary classification
    if binary:
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        y_binary = (y == 2).astype(int)  # 0: non-stress, 1: stress
        class_names = ['non-stress', 'stress']
        
        print(f"Binary classification:")
        n_nonstress = np.sum(y_binary == 0)
        n_stress = np.sum(y_binary == 1)
        print(f"  Non-stress (0): {n_nonstress:5d} ({100*n_nonstress/len(y_binary):.1f}%)")
        print(f"  Stress (1):     {n_stress:5d} ({100*n_stress/len(y_binary):.1f}%)\n")
    else:
        y_binary = y
        class_names = ['baseline', 'stress', 'amusement']
    
    # Subject-wise split
    print(f"Subject-wise split...")
    unique_subjects = sorted(list(set(subjects_arr)))
    
    subj_trainval, subj_test = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )
    val_size_adj = val_size / (1 - test_size)
    subj_train, subj_val = train_test_split(
        subj_trainval, test_size=val_size_adj, random_state=random_state
    )
    
    train_mask = np.isin(subjects_arr, subj_train)
    val_mask = np.isin(subjects_arr, subj_val)
    test_mask = np.isin(subjects_arr, subj_test)
    
    X_train = X[train_mask]
    y_train = y_binary[train_mask]
    subj_train_arr = subjects_arr[train_mask]
    
    X_val = X[val_mask]
    y_val = y_binary[val_mask]
    subj_val_arr = subjects_arr[val_mask]
    
    X_test = X[test_mask]
    y_test = y_binary[test_mask]
    subj_test_arr = subjects_arr[test_mask]
    
    print(f"  Train: {len(subj_train)} subjects → {len(X_train)} windows")
    print(f"  Val:   {len(subj_val)} subjects → {len(X_val)} windows")
    print(f"  Test:  {len(subj_test)} subjects → {len(X_test)} windows\n")
    
    # ✅ Per-subject normalization
    print(f"Normalization: per-subject z-score...")
    X_train = zscore_per_subject(X_train, subj_train_arr)
    X_val = zscore_per_subject(X_val, subj_val_arr)
    X_test = zscore_per_subject(X_test, subj_test_arr)
    
    # Per-channel global normalization (for consistency)
    print(f"Normalization: per-channel z-score (global)...")
    norm_stats = compute_normalization_stats(X_train)
    
    X_train = apply_normalization(X_train, norm_stats)
    X_val = apply_normalization(X_val, norm_stats)
    X_test = apply_normalization(X_test, norm_stats)
    
    # Class weights
    weights, class_counts = compute_class_weights(y_train)
    print(f"\nClass weights:")
    for c, w in weights.items():
        print(f"  {class_names[c]}: {w:.3f}\n")
    
    # Save
    print(f"Saving to {output_dir}...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    joblib.dump(LabelEncoder().fit(class_names),
               os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(norm_stats,
               os.path.join(output_dir, "normalization_stats.pkl"))
    
    # Metadata
    metadata = {
        'n_channels': X_train.shape[1],
        'channel_names': channel_names,
        'window_size': window_size,
        'window_duration_s': window_size / target_freq,
        'target_freq': target_freq,
        'overlap': overlap,
        'class_names': class_names,
        'class_weights': weights,
        'normalization': 'per_subject_zscore + per_channel_zscore',
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_windows': len(X),
        'total_subjects': len(unique_subjects),
        'train_subjects': [str(s) for s in subj_train],
        'val_subjects': [str(s) for s in subj_val],
        'test_subjects': [str(s) for s in subj_test]
    }
    
    joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))
    
    print(f"{'='*70}")
    print(f"✅ Preprocessing complete!")
    print(f"{'='*70}\n")
    
    return metadata


def load_processed_wesad_temporal(data_dir: str) -> Tuple:
    """Load preprocessed WESAD data."""
    print(f"Loading WESAD from {data_dir}...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    info = joblib.load(os.path.join(data_dir, 'metadata.pkl'))
    scaler = joblib.load(os.path.join(data_dir, 'normalization_stats.pkl'))
    
    print(f"✅ Loaded:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Classes: {info['class_names']}\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD')
    parser.add_argument('--data_dir', default='data/raw/wesad')
    parser.add_argument('--output_dir', default='data/processed/wesad_optimized')
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    
    args = parser.parse_args()
    
    preprocess_wesad_temporal(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        force_reprocess=args.force
    )