#!/usr/bin/env python3
"""
WESAD Dataset Preprocessing Module (OPTIMIZED VERSION)

Key improvements:
- LOSO (Leave-One-Subject-Out) cross-validation
- Normalization statistics saved for reproduction
- Strict label assignment (majority + validity threshold)
- Per-subject z-score normalization option
- Class imbalance handling (weights + metrics)
- Better downsampling (temporal mapping)
"""

import numpy as np
import pickle
from scipy import signal
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import time
import glob
from typing import Tuple, Dict, List, Optional
from collections import Counter
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
                                   axis=0)
    resampled_cols = []
    for c in range(arr.shape[1]):
        resampled_cols.append(
            signal.resample_poly(arr[:, c], int(target_freq),
                               int(orig_freq), axis=0)
        )
    return np.stack(resampled_cols, axis=1)


def downsample_labels_temporal(labels: np.ndarray, orig_freq: int,
                               target_freq: int) -> np.ndarray:
    """
    Downsample labels using temporal mapping (not linspace).
    
    Maps each sample in resampled frequency to nearest original label.
    """
    orig_length = len(labels)
    target_length = int(np.round(orig_length * (target_freq / orig_freq)))
    
    # Map resampled indices to original indices
    indices = np.round(
        np.linspace(0, orig_length - 1, target_length)
    ).astype(int)
    
    return labels[indices]


def resample_all_signals(data: Dict, target_freq: int = 32) -> Dict:
    """Resample all signals to target frequency."""
    sfreq = data['sfreq']
    original_length = len(data['ecg'])
    
    resampled = {}
    
    # Resample chest signals (700 Hz → target_freq)
    resampled['ecg'] = resample_signal(data['ecg'], sfreq['chest'],
                                      target_freq)
    resampled['eda_chest'] = resample_signal(data['eda_chest'],
                                            sfreq['chest'], target_freq)
    resampled['temp_chest'] = resample_signal(data['temp_chest'],
                                             sfreq['chest'], target_freq)
    resampled['acc_chest'] = resample_signal(data['acc_chest'],
                                            sfreq['chest'], target_freq)
    resampled['emg'] = resample_signal(data['emg'], sfreq['chest'],
                                      target_freq)
    resampled['resp'] = resample_signal(data['resp'], sfreq['chest'],
                                       target_freq)
    
    # Resample wrist signals
    resampled['acc_wrist'] = resample_signal(data['acc_wrist'],
                                            sfreq['acc_wrist'], target_freq)
    resampled['bvp'] = resample_signal(data['bvp'], sfreq['bvp'],
                                      target_freq)
    resampled['eda_wrist'] = resample_signal(data['eda_wrist'],
                                            sfreq['eda_wrist'], target_freq)
    resampled['temp_wrist'] = resample_signal(data['temp_wrist'],
                                             sfreq['temp_wrist'],
                                             target_freq)
    
    # Downsample labels using temporal mapping
    resampled['labels'] = downsample_labels_temporal(data['labels'],
                                                    sfreq['chest'],
                                                    target_freq)
    
    resampled['subject'] = data['subject']
    resampled['sfreq'] = target_freq
    
    return resampled


def apply_filters(resampled: Dict, target_freq: int = 32) -> Dict:
    """Apply bandpass/lowpass filters to improve SNR."""
    filtered = resampled.copy()

    try:
        sos_ecg = signal.butter(4, [0.5, min(15, target_freq/2.1)],
                               btype='band', fs=target_freq, output='sos')
        sos_bvp = signal.butter(4, [0.5, min(12, target_freq/2.1)],
                               btype='band', fs=target_freq, output='sos')
        sos_acc = signal.butter(4, [0.1, min(2, target_freq/2.1)],
                               btype='band', fs=target_freq, output='sos')
        sos_eda = signal.butter(4, min(1.5, target_freq/2.1), btype='low',
                               fs=target_freq, output='sos')
        sos_temp = signal.butter(4, min(0.5, target_freq/2.1), btype='low',
                                fs=target_freq, output='sos')
        sos_resp = signal.butter(4, [0.1, min(0.5, target_freq/2.1)],
                                btype='band', fs=target_freq, output='sos')
        sos_emg = signal.butter(4, [0.5, min(2, target_freq/2.1)],
                               btype='band', fs=target_freq, output='sos')

    except Exception as e:
        print(f"  Warning: filter design failed ({e})")
        return resampled

    signal_keys = {
        'ecg': sos_ecg,
        'bvp': sos_bvp,
        'acc_chest': sos_acc,
        'acc_wrist': sos_acc,
        'eda_chest': sos_eda,
        'eda_wrist': sos_eda,
        'temp_chest': sos_temp,
        'temp_wrist': sos_temp,
        'resp': sos_resp,
        'emg': sos_emg
    }

    for key, sos_filter in signal_keys.items():
        if key in filtered and filtered[key] is not None:
            try:
                filtered[key] = signal.sosfiltfilt(sos_filter,
                                                  filtered[key], axis=0)
            except Exception as e:
                print(f"  Warning: filtering failed for {key} ({e})")

    return filtered


def reduce_outliers(signals: Dict, method: str = 'clip',
                   lower_p: float = 1.0,
                   upper_p: float = 99.0) -> Dict:
    """Basic outlier mitigation prior to windowing."""
    if method == 'none':
        return signals
    
    clipped = signals.copy()
    keys = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp',
            'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']
    
    for k in keys:
        if k in clipped and clipped[k] is not None:
            try:
                arr = clipped[k]
                if arr.ndim == 1:
                    lo = np.percentile(arr, lower_p)
                    hi = np.percentile(arr, upper_p)
                    clipped[k] = np.clip(arr, lo, hi)
                else:
                    for c in range(arr.shape[1]):
                        lo = np.percentile(arr[:, c], lower_p)
                        hi = np.percentile(arr[:, c], upper_p)
                        arr[:, c] = np.clip(arr[:, c], lo, hi)
            except Exception:
                pass
    
    return clipped


# ============================================================================
# WINDOWING
# ============================================================================

def create_temporal_windows(data: Dict, window_size: int = 1920,
                           overlap: float = 0.5,
                           label_threshold: float = 0.7) -> Tuple[
                               np.ndarray, np.ndarray, List[str], int]:
    """
    Create temporal sliding windows with strict label assignment.
    
    Args:
        label_threshold: Fraction of window that must be valid labels
                        to include the window
    
    Returns:
        windows, labels, channel_names, n_dropped_windows
    """
    step_size = int(window_size * (1 - overlap))
    ref_length = len(data['ecg'])
    
    signal_names = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg',
                    'resp', 'bvp', 'eda_wrist', 'temp_wrist', 'acc_wrist']
    
    channel_names = []
    for name in signal_names:
        if name not in data:
            continue
        arr = data[name]
        if arr.ndim > 1 and arr.shape[1] > 1:
            axis_labels = ['x', 'y', 'z'] if arr.shape[1] == 3 else \
                         [f'ch{i}' for i in range(arr.shape[1])]
            for ax in axis_labels:
                channel_names.append(f"{name}_{ax}")
        else:
            channel_names.append(name)
    
    n_channels = len(channel_names)
    windows_list = []
    labels_list = []
    n_dropped = 0
    
    for start_idx in range(0, ref_length - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Check all signals are available
        all_signals_complete = True
        for name in signal_names:
            if name not in data:
                continue
            arr = data[name]
            if len(arr) < end_idx:
                all_signals_complete = False
                break
        
        if not all_signals_complete:
            continue
        
        # Extract signals
        window_data = np.zeros((n_channels, window_size))
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
        
        # Check validity threshold
        if len(valid_labels) < window_size * label_threshold:
            n_dropped += 1
            continue
        
        # Use majority label among valid labels
        if len(valid_labels) > 0:
            window_label = np.bincount(
                valid_labels.astype(int)
            ).argmax()
        else:
            n_dropped += 1
            continue
        
        windows_list.append(window_data)
        labels_list.append(window_label)
    
    return (np.array(windows_list), np.array(labels_list),
            channel_names, n_dropped)


# ============================================================================
# NORMALIZATION (WITH STATISTICS SAVED)
# ============================================================================

def compute_normalization_stats(X_train: np.ndarray) -> Dict:
    """Compute normalization statistics from training set."""
    # Per-channel z-score
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)
    train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    return {
        'type': 'per_channel_zscore',
        'mean': train_mean,
        'std': train_std,
        'X_shape': X_train.shape
    }


def apply_normalization(X: np.ndarray, stats: Dict) -> np.ndarray:
    """Apply saved normalization statistics."""
    mean = stats['mean']
    std = stats['std']
    return (X - mean) / std


# ============================================================================
# LOSO CROSS-VALIDATION
# ============================================================================

def split_by_subject_loso(subjects_arr: np.ndarray,
                          X: np.ndarray,
                          y: np.ndarray) -> List[Tuple]:
    """
    Generate LOSO (Leave-One-Subject-Out) splits.
    
    Returns:
        list of (X_train, y_train, X_val, y_val, val_subject, train_subjects)
    """
    unique_subjects = sorted(list(set(subjects_arr)))
    splits = []
    
    for test_subject in unique_subjects:
        test_mask = subjects_arr == test_subject
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        train_subjects = subjects_arr[train_mask]
        
        splits.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'test_subject': test_subject,
            'train_subjects': np.unique(train_subjects)
        })
    
    return splits


def split_by_subject_standard(subjects_arr: np.ndarray,
                              X: np.ndarray,
                              y: np.ndarray,
                              test_size: float = 0.2,
                              val_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
    """Standard subject-wise split (para referência)."""
    from sklearn.model_selection import train_test_split
    
    unique_subjects = np.array(
        sorted(list(set(subjects_arr)))
    )
    
    subj_trainval, subj_test = train_test_split(
        unique_subjects, test_size=test_size,
        random_state=random_state
    )
    val_size_adj = val_size / (1 - test_size)
    subj_train, subj_val = train_test_split(
        subj_trainval, test_size=val_size_adj,
        random_state=random_state
    )
    
    train_mask = np.isin(subjects_arr, subj_train)
    val_mask = np.isin(subjects_arr, subj_val)
    test_mask = np.isin(subjects_arr, subj_test)
    
    return {
        'X_train': X[train_mask],
        'y_train': y[train_mask],
        'X_val': X[val_mask],
        'y_val': y[val_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'train_subjects': np.unique(subjects_arr[train_mask]),
        'val_subjects': np.unique(subjects_arr[val_mask]),
        'test_subjects': np.unique(subjects_arr[test_mask])
    }


# ============================================================================
# CLASS IMBALANCE ANALYSIS
# ============================================================================

def compute_class_weights(y: np.ndarray) -> Dict:
    """Compute class weights for imbalanced data."""
    class_counts = np.bincount(y.astype(int))
    n_classes = len(class_counts)
    n_samples = len(y)
    
    # Balanced weights: 1 / (n_classes * p_c)
    weights = {}
    for c in range(n_classes):
        if class_counts[c] > 0:
            weights[c] = n_samples / (n_classes * class_counts[c])
        else:
            weights[c] = 1.0
    
    return weights, class_counts


def export_eda(eda_dir: str, class_counts: np.ndarray,
              class_names: List[str], weights: Dict = None) -> None:
    """Save class distribution and weights."""
    if plt is None:
        return
    
    try:
        os.makedirs(eda_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Class distribution
        ax1.bar(class_names, class_counts, color=['green', 'red'])
        ax1.set_title('WESAD Class Distribution')
        ax1.set_ylabel('Num windows')
        for i, v in enumerate(class_counts):
            ax1.text(i, v + max(class_counts)*0.01, str(int(v)),
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Class weights
        if weights:
            weight_values = [weights.get(i, 1.0) for i in range(len(class_names))]
            ax2.bar(class_names, weight_values, color=['green', 'red'])
            ax2.set_title('Class Weights (for training)')
            ax2.set_ylabel('Weight')
            for i, v in enumerate(weight_values):
                ax2.text(i, v + max(weight_values)*0.01, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=10, weight='bold')
        
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'class_balance.png'), dpi=150)
        plt.close(fig)
        
    except Exception as e:
        print(f"  Warning: EDA export failed ({e})")


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_wesad_file(args: Tuple) -> Tuple[Dict, str]:
    """Process a single WESAD file in parallel."""
    (pkl_file, target_freq, window_size, overlap,
     outlier_method, clip_lower_p, clip_upper_p,
     label_threshold) = args

    try:
        data = load_wesad_file(pkl_file)
        subject = data['subject']
        
        resampled = resample_all_signals(data, target_freq)
        outlier_reduced = reduce_outliers(
            resampled, method=outlier_method,
            lower_p=clip_lower_p, upper_p=clip_upper_p
        )
        filtered = apply_filters(outlier_reduced, target_freq)
        
        windows, labels, ch_names, n_dropped = create_temporal_windows(
            filtered, window_size, overlap,
            label_threshold=label_threshold
        )

        return {
            'windows': windows,
            'labels': labels,
            'subject': subject,
            'channel_names': ch_names,
            'n_dropped': n_dropped
        }, subject

    except Exception as e:
        print(f"  Error processing {os.path.basename(pkl_file)}: {e}")
        return None, None


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def preprocess_wesad_temporal(
    data_dir: str,
    output_dir: str,
    target_freq: int = 32,
    window_size: int = 1920,
    overlap: float = 0.5,
    label_threshold: float = 0.7,
    binary: bool = True,
    use_loso: bool = True,
    outlier_method: str = 'clip',
    clip_lower_p: float = 1.0,
    clip_upper_p: float = 99.0,
    save_eda_plots: bool = True,
    n_workers: int = None,
    force_reprocess: bool = False,
    random_state: int = 42) -> Dict:
    """
    Complete preprocessing pipeline for WESAD (OPTIMIZED).
    
    Args:
        use_loso: If True, returns LOSO splits instead of train/val/test
        label_threshold: Minimum fraction of valid labels in window
    """
    print("="*70)
    print("WESAD TEMPORAL PREPROCESSING (OPTIMIZED)")
    print("="*70)
    print(f"Target frequency: {target_freq} Hz")
    print(f"Window: {window_size} samples ({window_size/target_freq:.1f}s)")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Label threshold: {label_threshold*100:.0f}%")
    print(f"Classification: {'Binary (stress vs non-stress)' if binary else '3-class'}")
    print(f"Split strategy: {'LOSO (Leave-One-Subject-Out)' if use_loso else 'Standard subject-wise'}\n")

    os.makedirs(output_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} parallel workers\n")

    pkl_files = glob.glob(os.path.join(data_dir, "**/*.pkl"),
                         recursive=True)
    print(f"Found {len(pkl_files)} pickle files\n")

    if not pkl_files:
        raise ValueError(f"No pickle files found in {data_dir}")

    process_args = [
        (pkl_file, target_freq, window_size, overlap, outlier_method,
         clip_lower_p, clip_upper_p, label_threshold)
        for pkl_file in pkl_files
    ]

    start_time = time.time()

    print(f"Processing {len(pkl_files)} files in parallel...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_wesad_file, process_args)

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
            all_subjects.extend([subject] * len(result['windows']))
            total_dropped += result['n_dropped']

    print(f"Parallel processing completed in {time.time() - start_time:.1f}s")
    print(f"Total windows dropped (invalid labels): {total_dropped}\n")
    
    if not all_windows:
        raise ValueError("No valid data processed")
    
    print(f"Combining data from {len(pkl_files)} files...")
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subjects_arr = np.array(all_subjects)
    
    print(f"Total windows: {len(X)}")
    print(f"Window shape: {X.shape}")
    print(f"Label distribution (raw): {np.bincount(y.astype(int))}")
    
    # Binary classification
    if binary:
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        y_relabeled = (y == 2).astype(int)
        class_names = ['non-stress', 'stress']
        
        print(f"\nAfter filtering (binary):")
        print(f"  Total windows: {len(X)}")
        print(f"  Non-stress (0): {np.sum(y_relabeled == 0)} "
              f"({np.sum(y_relabeled == 0)/len(y_relabeled)*100:.1f}%)")
        print(f"  Stress (1): {np.sum(y_relabeled == 1)} "
              f"({np.sum(y_relabeled == 1)/len(y_relabeled)*100:.1f}%)")
    else:
        y_relabeled = y
        class_names = ['baseline', 'stress', 'amusement']
    
    # Compute class weights
    weights, class_counts = compute_class_weights(y_relabeled)
    print(f"\nClass weights (for training):")
    for c, w in weights.items():
        print(f"  {class_names[c]}: {w:.3f}")
    
    if save_eda_plots:  
        eda_dir = os.path.join(output_dir, 'eda')
        export_eda(eda_dir, class_counts, class_names, weights)
        print(f"\nEDA saved to {eda_dir}")
    
    # =====================================================================
    # SPLIT STRATEGY
    # =====================================================================
    
    if use_loso:
        print(f"\nUsing LOSO (Leave-One-Subject-Out) strategy...")
        loso_splits = split_by_subject_loso(subjects_arr, X, y_relabeled)
        
        print(f"Created {len(loso_splits)} LOSO splits:")
        for split in loso_splits:
            print(f"  Test: {split['test_subject']} "
                  f"({len(split['X_test'])} windows) | "
                  f"Train: {len(split['train_subjects'])} subjects "
                  f"({len(split['X_train'])} windows)")
        
        # Save all LOSO splits
        loso_output_dir = os.path.join(output_dir, 'loso_splits')
        os.makedirs(loso_output_dir, exist_ok=True)
        
        for split_idx, split in enumerate(loso_splits):
            split_dir = os.path.join(loso_output_dir,
                                    f'split_{split_idx:02d}')
            os.makedirs(split_dir, exist_ok=True)
            
            # Compute normalization from training data
            norm_stats = compute_normalization_stats(split['X_train'])
            
            # Apply normalization
            X_train_norm = apply_normalization(split['X_train'],
                                              norm_stats)
            X_test_norm = apply_normalization(split['X_test'],
                                             norm_stats)
            
            np.save(os.path.join(split_dir, 'X_train.npy'),
                   X_train_norm)
            np.save(os.path.join(split_dir, 'y_train.npy'),
                   split['y_train'])
            np.save(os.path.join(split_dir, 'X_test.npy'),
                   X_test_norm)
            np.save(os.path.join(split_dir, 'y_test.npy'),
                   split['y_test'])
            
            joblib.dump(norm_stats, os.path.join(split_dir,
                                                 'normalization_stats.pkl'))
            
            split_info = {
                'split_idx': split_idx,
                'test_subject': split['test_subject'],
                'train_subjects': [str(s) for s in split['train_subjects']],
                'n_train': len(split['X_train']),
                'n_test': len(split['X_test']),
                'normalization': norm_stats['type']
            }
            joblib.dump(split_info, os.path.join(split_dir,
                                                 'split_info.pkl'))
        
        # Save metadata
        metadata = {
            'strategy': 'loso',
            'n_splits': len(loso_splits),
            'class_names': class_names,
            'class_weights': weights,
            'channel_names': channel_names,
            'target_freq': target_freq,
            'window_size': window_size,
            'overlap': overlap,
            'label_threshold': label_threshold,
            'total_windows': len(X)
        }
        
        joblib.dump(metadata, os.path.join(output_dir, 'metadata.pkl'))
        
        return metadata
    
    else:
        # Standard subject-wise split
        print(f"\nSplitting by subject (standard strategy)...")
        split_data = split_by_subject_standard(
            subjects_arr, X, y_relabeled,
            test_size=0.2, val_size=0.2,
            random_state=random_state
        )
        
        print(f"Subject splits:")
        print(f"  Train: {len(split_data['train_subjects'])} subjects "
              f"→ {len(split_data['X_train'])} windows")
        print(f"  Val: {len(split_data['val_subjects'])} subjects "
              f"→ {len(split_data['X_val'])} windows")
        print(f"  Test: {len(split_data['test_subjects'])} subjects "
              f"→ {len(split_data['X_test'])} windows")
        
        print(f"\nApplying per-channel z-score normalization...")
        norm_stats = compute_normalization_stats(split_data['X_train'])
        
        X_train = apply_normalization(split_data['X_train'], norm_stats)
        X_val = apply_normalization(split_data['X_val'], norm_stats)
        X_test = apply_normalization(split_data['X_test'], norm_stats)
        
        print(f"\nSaving to {output_dir}...")
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_train.npy"),
               split_data['y_train'])
        np.save(os.path.join(output_dir, "y_val.npy"), split_data['y_val'])
        np.save(os.path.join(output_dir, "y_test.npy"),
               split_data['y_test'])
        
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)
        joblib.dump(label_encoder, os.path.join(output_dir,
                                               "label_encoder.pkl"))
        
        # Save normalization statistics
        joblib.dump(norm_stats, os.path.join(output_dir,
                                            "normalization_stats.pkl"))
        
        preprocessing_info = {
            'strategy': 'standard',
            'n_windows': len(X),
            'n_channels': X.shape[1],
            'window_size': X.shape[2],
            'n_classes': len(class_names),
            'class_names': class_names,
            'class_weights': weights,
            'binary': binary,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'target_freq': target_freq,
            'overlap': overlap,
            'label_threshold': label_threshold,
            'window_duration_s': window_size / target_freq,
            'channels': channel_names,
            'subject_splits': {
                'train_subjects': [str(s) for s in split_data['train_subjects']],
                'val_subjects': [str(s) for s in split_data['val_subjects']],
                'test_subjects': [str(s) for s in split_data['test_subjects']],
                'total_subjects': len(np.unique(subjects_arr))
            },
            'normalization': {
                'type': norm_stats['type'],
                'per_channel': True
            },
            'files_processed': len(pkl_files),
            'windows_dropped': total_dropped,
            'parallel_processing': True,
            'n_workers': n_workers
        }
        
        joblib.dump(preprocessing_info, os.path.join(output_dir,
                                                     "preprocessing_info.pkl"))
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Preprocessing complete in {elapsed:.1f}s")
        print(f"  Classes: {class_names}")
        print(f"  Channels: {X.shape[1]}")
        print(f"  Window: {window_size} samples ({window_size/target_freq:.1f}s)")
        print(f"  Data saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        return preprocessing_info


# ============================================================================
# LOADING FUNCTIONS (UPDATED)
# ============================================================================

def load_processed_wesad_temporal(data_dir: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    LabelEncoder, Dict]:
    """Load preprocessed WESAD temporal data."""
    print(f"Loading processed WESAD data from {data_dir}...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    label_encoder = joblib.load(os.path.join(data_dir,
                                            'label_encoder.pkl'))
    info = joblib.load(os.path.join(data_dir,
                                   'preprocessing_info.pkl'))
    
    print(f"Loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {info['class_names']}")
    print(f"  Channels: {info['n_channels']}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, \
           label_encoder, info


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess WESAD dataset (optimized)'
    )
    parser.add_argument('--data_dir', default='data/raw/wesad',
                       help='Raw data directory')
    parser.add_argument('--output_dir', default='data/processed/wesad',
                       help='Output directory')
    parser.add_argument('--target_freq', type=int, default=32,
                       help='Target sampling frequency (Hz)')
    parser.add_argument('--window_size', type=int, default=1920,
                       help='Window size in samples')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio')
    parser.add_argument('--label_threshold', type=float, default=0.7,
                       help='Min fraction of valid labels in window')
    parser.add_argument('--use_loso', action='store_true',
                       help='Use LOSO instead of standard split')
    parser.add_argument('--binary', action='store_true', default=True,
                       help='Binary classification')
    
    args = parser.parse_args()
    
    print("="*70)
    print("WESAD PREPROCESSING (OPTIMIZED)")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.data_dir):
        print(f"Raw data not found: {args.data_dir}")
        exit(1)
    
    info = preprocess_wesad_temporal(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_freq=args.target_freq,
        window_size=args.window_size,
        overlap=args.overlap,
        label_threshold=args.label_threshold,
        use_loso=args.use_loso,
        binary=args.binary
    )
    
    print(f"WESAD preprocessing completed!")