"""
WESAD Dataset Preprocessing Module (FINAL VERSION - INTEGRATED)

Features:
- Load pickle files with physiological signals from RespiBAN and Empatica E4
- Resample to uniform frequency (32 Hz - optimal for ECG/BVP preservation)
- Temporal windowing (60s windows, 50% overlap)
- Subject-wise splitting (LOSO-style to avoid leakage)
- Per-channel z-score normalization (train-only)
- ✅ AUTO-CREATE AUGMENTED DATA (integrated like Sleep-EDF)
- Binary (stress vs non-stress) or 3-class (baseline/stress/amusement)

WESAD contains:
- 15 subjects (~100 min each)
- RespiBAN (chest): ECG, EDA, EMG, Temp, Resp, ACC (700 Hz)
- Empatica E4 (wrist): ACC, BVP, EDA, TEMP (various frequencies)
- Labels: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=other

Usage:
    python -m src.preprocessing.wesad --data_dir data/raw/wesad --output_dir data/processed/wesad
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

def resample_signal(arr: np.ndarray, orig_freq: int, target_freq: int) -> np.ndarray:
    """Resample signal using polyphase filtering."""
    if arr.ndim == 1:
        return signal.resample_poly(arr, int(target_freq), int(orig_freq), axis=0)
    resampled_cols = []
    for c in range(arr.shape[1]):
        resampled_cols.append(
            signal.resample_poly(arr[:, c], int(target_freq), int(orig_freq), axis=0)
        )
    return np.stack(resampled_cols, axis=1)


def resample_all_signals(data: Dict, target_freq: int = 32) -> Dict:
    """Resample all signals to target frequency."""
    sfreq = data['sfreq']
    original_length = len(data['ecg'])
    target_length = int(np.round(original_length * (target_freq / sfreq['chest'])))
    
    resampled = {}
    
    # Resample chest signals (700 Hz → target_freq)
    resampled['ecg'] = resample_signal(data['ecg'], sfreq['chest'], target_freq)
    resampled['eda_chest'] = resample_signal(data['eda_chest'], sfreq['chest'], target_freq)
    resampled['temp_chest'] = resample_signal(data['temp_chest'], sfreq['chest'], target_freq)
    resampled['acc_chest'] = resample_signal(data['acc_chest'], sfreq['chest'], target_freq)
    resampled['emg'] = resample_signal(data['emg'], sfreq['chest'], target_freq)
    resampled['resp'] = resample_signal(data['resp'], sfreq['chest'], target_freq)
    
    # Resample wrist signals
    resampled['acc_wrist'] = resample_signal(data['acc_wrist'], sfreq['acc_wrist'], target_freq)
    resampled['bvp'] = resample_signal(data['bvp'], sfreq['bvp'], target_freq)
    resampled['eda_wrist'] = resample_signal(data['eda_wrist'], sfreq['eda_wrist'], target_freq)
    resampled['temp_wrist'] = resample_signal(data['temp_wrist'], sfreq['temp_wrist'], target_freq)
    
    # Downsample labels by nearest-neighbor indexing
    indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    resampled['labels'] = data['labels'][indices]
    
    resampled['subject'] = data['subject']
    resampled['sfreq'] = target_freq
    
    return resampled


def apply_filters(resampled: Dict, target_freq: int = 32) -> Dict:
    """Apply bandpass/lowpass filters to improve SNR."""
    filtered = resampled.copy()

    try:
        # Pre-compute filter coefficients
        sos_ecg = signal.butter(4, [0.5, min(15, target_freq/2.1)], btype='band', 
                               fs=target_freq, output='sos')
        sos_bvp = signal.butter(4, [0.5, min(12, target_freq/2.1)], btype='band', 
                               fs=target_freq, output='sos')
        sos_acc = signal.butter(4, [0.1, min(2, target_freq/2.1)], btype='band', 
                               fs=target_freq, output='sos')
        sos_eda = signal.butter(4, min(1.5, target_freq/2.1), btype='low', 
                               fs=target_freq, output='sos')
        sos_temp = signal.butter(4, min(0.5, target_freq/2.1), btype='low', 
                                fs=target_freq, output='sos')
        sos_resp = signal.butter(4, [0.1, min(0.5, target_freq/2.1)], btype='band', 
                                fs=target_freq, output='sos')
        sos_emg = signal.butter(4, [0.5, min(2, target_freq/2.1)], btype='band', 
                               fs=target_freq, output='sos')

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
                filtered[key] = signal.sosfiltfilt(sos_filter, filtered[key], axis=0)
            except Exception as e:
                print(f"  Warning: filtering failed for {key} ({e})")

    return filtered


def _clip_array_percentiles(arr: np.ndarray, lower_p: float, upper_p: float) -> np.ndarray:
    """Clip array values to percentiles per channel."""
    if arr is None:
        return arr
    if arr.ndim == 1:
        lo = np.percentile(arr, lower_p)
        hi = np.percentile(arr, upper_p)
        return np.clip(arr, lo, hi)
    clipped_cols = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        lo = np.percentile(col, lower_p)
        hi = np.percentile(col, upper_p)
        clipped_cols.append(np.clip(col, lo, hi))
    return np.stack(clipped_cols, axis=1)


def reduce_outliers(signals: Dict, method: str = 'clip', 
                   lower_p: float = 0.5, upper_p: float = 99.5) -> Dict:
    """Basic outlier mitigation prior to windowing."""
    if method == 'none':
        return signals
    clipped = signals.copy()
    keys = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp', 
            'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']
    for k in keys:
        if k in clipped and clipped[k] is not None:
            try:
                clipped[k] = _clip_array_percentiles(clipped[k], lower_p, upper_p)
            except Exception:
                pass
    return clipped


# ============================================================================
# WINDOWING
# ============================================================================

def create_temporal_windows(data: Dict, window_size: int = 1920, 
                           overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create temporal sliding windows for LSTM/CNN.
    Only includes complete windows to avoid zero-padding artifacts.
    """
    step_size = int(window_size * (1 - overlap))
    ref_length = len(data['ecg'])
    
    signal_names = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp',
                    'bvp', 'eda_wrist', 'temp_wrist', 'acc_wrist']
    
    channel_names = []
    for name in signal_names:
        if name not in data:
            continue
        arr = data[name]
        if arr.ndim > 1 and arr.shape[1] > 1:
            axis_labels = ['x', 'y', 'z'] if arr.shape[1] == 3 else [f'ch{i}' for i in range(arr.shape[1])]
            for ax in axis_labels:
                channel_names.append(f"{name}_{ax}")
        else:
            channel_names.append(name)
    
    n_channels = len(channel_names)
    windows_list = []
    labels_list = []
    
    # Slide windows - only process complete windows
    for start_idx in range(0, ref_length - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
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
        
        # Get label (majority vote, excluding undefined=0)
        window_labels = data['labels'][start_idx:end_idx]
        valid_labels = window_labels[window_labels != 0]
        if len(valid_labels) > 0:
            window_label = np.bincount(valid_labels.astype(int)).argmax()
        else:
            window_label = 0
        
        windows_list.append(window_data)
        labels_list.append(window_label)
    
    return np.array(windows_list), np.array(labels_list), channel_names


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_wesad_file(args: Tuple) -> Tuple[Dict, str]:
    """Processa um único arquivo WESAD para paralelização."""
    pkl_file, target_freq, window_size, overlap, outlier_method, clip_lower_p, clip_upper_p = args

    try:
        data = load_wesad_file(pkl_file)
        subject = data['subject']
        resampled = resample_all_signals(data, target_freq)
        outlier_reduced = reduce_outliers(resampled, method=outlier_method, 
                                         lower_p=clip_lower_p, upper_p=clip_upper_p)
        filtered = apply_filters(outlier_reduced, target_freq)
        windows, labels, ch_names = create_temporal_windows(filtered, window_size, overlap)

        return {
            'windows': windows,
            'labels': labels,
            'subject': subject,
            'channel_names': ch_names
        }, subject

    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(pkl_file)}: {e}")
        return None, None


# ============================================================================
# AUGMENTATION
# ============================================================================

def _augment_temporal(X: np.ndarray, noise_std: float = 0.01, 
                     max_time_shift: int = 8, seed: int = 42) -> np.ndarray:
    """
    Apply deterministic temporal augmentation with fixed seed.
    
    Args:
        X: Input data (samples, channels, timesteps)
        noise_std: Standard deviation for Gaussian noise
        max_time_shift: Maximum time shift in samples
        seed: Random seed for reproducibility
        
    Returns:
        Augmented data with same shape as input
    """
    rng = np.random.default_rng(seed)
    X_aug = X.copy()

    n_samples = X_aug.shape[0]
    noise = rng.normal(0, noise_std, size=X_aug.shape)
    shifts = rng.integers(-max_time_shift, max_time_shift + 1, size=n_samples)

    X_aug += noise

    if max_time_shift > 0:
        for i in range(n_samples):
            shift = shifts[i]
            if shift != 0:
                if shift > 0:
                    X_aug[i] = np.pad(X_aug[i], ((0, 0), (shift, 0)), 
                                     mode='edge')[:, :-shift]
                else:
                    X_aug[i] = np.pad(X_aug[i], ((0, 0), (0, -shift)), 
                                     mode='edge')[:, -shift:]

    return X_aug


def _create_augmented_data_internal(X_train: np.ndarray, y_train: np.ndarray,
                                    output_dir: str, n_augmentations: int = 2) -> None:
    """
    Internal function called by preprocess_wesad_temporal.
    Creates augmented training data and saves it.
    """
    print(f"  Creating augmented data ({n_augmentations} augmentations per sample)...")
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for i in range(n_augmentations):
        seed = 42 + i
        X_aug = _augment_temporal(X_train, noise_std=0.01, max_time_shift=8, seed=seed)
        X_aug_list.append(X_aug)
        y_aug_list.append(y_train)

    # Concatenate all augmented data
    X_all = np.concatenate(X_aug_list, axis=0)
    y_all = np.concatenate(y_aug_list, axis=0)

    print(f"    Augmented: {X_all.shape[0]} samples (was {X_train.shape[0]})")

    # Save augmented data
    np.save(os.path.join(output_dir, 'X_train_augmented.npy'), X_all)
    np.save(os.path.join(output_dir, 'y_train_augmented.npy'), y_all)

    # Save augmentation info
    aug_info = {
        'n_augmentations': n_augmentations,
        'original_samples': X_train.shape[0],
        'augmented_samples': X_all.shape[0],
        'augmentation_factor': len(X_aug_list),
        'augmentation_method': 'temporal (noise + time shift)'
    }
    joblib.dump(aug_info, os.path.join(output_dir, 'augmentation_info.pkl'))


# ============================================================================
# CACHE & EDA
# ============================================================================

def check_wesad_cache_status(data_dir: str, output_dir: str) -> Tuple[bool, Dict]:
    """Verifica se os dados WESAD já foram processados."""
    import hashlib

    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'label_encoder.pkl', 'preprocessing_info.pkl'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            missing_files.append(file)

    if missing_files:
        return False, {'missing_files': missing_files}

    try:
        input_files = glob.glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)

        if not input_files:
            return False, {'error': 'No input files found'}

        input_hash = hashlib.md5()
        for file_path in sorted(input_files):
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                input_hash.update(f"{file_path}:{stat.st_size}:{stat.st_mtime}".encode())

        cache_file = os.path.join(output_dir, '.wesad_cache_info.pkl')
        if os.path.exists(cache_file):
            cache_info = joblib.load(cache_file)
            if cache_info.get('input_hash') == input_hash.hexdigest():
                return True, cache_info

        cache_info = {
            'input_hash': input_hash.hexdigest(),
            'input_files': len(input_files),
            'cache_time': time.time()
        }
        joblib.dump(cache_info, cache_file)

        return False, cache_info

    except Exception as e:
        return False, {'error': str(e)}


def export_basic_eda(eda_dir: str, class_counts: np.ndarray, 
                    class_names: List[str]) -> None:
    """Save a simple class distribution bar chart."""
    if plt is None:
        return
    try:
        os.makedirs(eda_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(class_names, class_counts)
        ax.set_title('WESAD Class Distribution')
        ax.set_ylabel('Num windows')
        for i, v in enumerate(class_counts):
            ax.text(i, v + max(class_counts)*0.01, str(int(v)), 
                   ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'class_distribution.png'), dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def preprocess_wesad_temporal(data_dir: str, output_dir: str,
                              target_freq: int = 32,
                              window_size: int = 1920,
                              overlap: float = 0.5,
                              test_size: float = 0.2,
                              val_size: float = 0.2,
                              binary: bool = True,
                              random_state: int = 42,
                              outlier_method: str = 'clip',
                              clip_lower_p: float = 1.0,
                              clip_upper_p: float = 99.0,
                              export_eda: bool = True,
                              n_workers: int = None,
                              force_reprocess: bool = False,
                              create_augmentations: bool = True,  # ✅ NOVO
                              n_augmentations: int = 2) -> Dict:  # ✅ NOVO
    """
    Complete preprocessing pipeline for WESAD (temporal windows for LSTM/CNN).
    
    Args:
        data_dir: Directory containing WESAD pickle files (S2/, S3/, ...)
        output_dir: Directory to save processed data
        target_freq: Target sampling frequency (Hz) - default 32 Hz
        window_size: Window size in samples (default: 1920 = 60s at 32 Hz)
        overlap: Overlap ratio (0.0-0.9)
        test_size: Test set size ratio (subject-wise split)
        val_size: Validation set size ratio (subject-wise split)
        binary: If True, binary classification (stress vs non-stress).
                If False, 3-class (baseline/stress/amusement)
        random_state: Random seed
        outlier_method: Method for outlier handling ('clip' or 'none')
        clip_lower_p: Lower percentile for clipping (default: 1.0%)
        clip_upper_p: Upper percentile for clipping (default: 99.0%)
        export_eda: Whether to export EDA plots
        n_workers: Number of parallel workers
        force_reprocess: Force reprocessing
        create_augmentations: Automatically create augmented data (default=True)
        n_augmentations: Number of augmentations per sample (default=2)
    
    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("WESAD TEMPORAL PREPROCESSING (FINAL VERSION)")
    print("="*70)
    print(f"Target frequency: {target_freq} Hz")
    print(f"Window size: {window_size} samples ({window_size/target_freq:.1f} seconds)")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Classification: {'Binary (stress vs non-stress)' if binary else '3-class (baseline/stress/amusement)'}")
    print(f"Create augmentations: {create_augmentations} (factor: {n_augmentations+1}×)\n")

    os.makedirs(output_dir, exist_ok=True)

    # Check cache status
    if not force_reprocess:
        is_cached, cache_info = check_wesad_cache_status(data_dir, output_dir)
        if is_cached:
            print("✓ Cache hit! Loading preprocessed WESAD data...")
            return load_processed_wesad_temporal(output_dir)[7]

        print("○ Cache miss or outdated - reprocessing WESAD data...")

    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} parallel workers\n")

    # Find all pickle files
    pkl_files = glob.glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} pickle files\n")

    if not pkl_files:
        raise ValueError(f"No pickle files found in {data_dir}")

    # Prepare arguments
    process_args = [
        (pkl_file, target_freq, window_size, overlap, outlier_method, 
         clip_lower_p, clip_upper_p)
        for pkl_file in pkl_files
    ]

    start_time = time.time()

    print(f"Processing {len(pkl_files)} files in parallel...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_wesad_file, process_args)

    # Combine results
    all_windows = []
    all_labels = []
    all_subjects = []
    channel_names = None

    for result, subject in results:
        if result is not None:
            if channel_names is None:
                channel_names = result['channel_names']

            all_windows.append(result['windows'])
            all_labels.append(result['labels'])
            all_subjects.extend([subject] * len(result['windows']))

    print(f"Parallel processing completed in {time.time() - start_time:.1f}s")
    
    if not all_windows:
        raise ValueError("No valid data processed")
    
    # Combine all data
    print(f"\nCombining data from {len(pkl_files)} files...")
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subjects_arr = np.array(all_subjects)
    
    print(f"Total windows: {len(X)}")
    print(f"Window shape: {X.shape} (n_windows, n_channels, window_size)")
    print(f"Label distribution (raw): {np.bincount(y.astype(int))}")
    
    # Filter and relabel
    if binary:
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        y_relabeled = (y == 2).astype(int)
        class_names = ['non-stress', 'stress']
        
        print(f"\nAfter filtering and relabeling (binary):")
        print(f"  Total windows: {len(X)}")
        print(f"  Non-stress (0): {np.sum(y_relabeled == 0)} ({np.sum(y_relabeled == 0)/len(y_relabeled)*100:.1f}%)")
        print(f"  Stress (1): {np.sum(y_relabeled == 1)} ({np.sum(y_relabeled == 1)/len(y_relabeled)*100:.1f}%)")
        if export_eda:
            export_basic_eda(os.path.join(output_dir, 'eda'), np.array([
                np.sum(y_relabeled == 0), np.sum(y_relabeled == 1)
            ]), class_names)
    else:
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        y_relabeled = y - 1
        class_names = ['baseline', 'stress', 'amusement']
        
        print(f"\nAfter filtering and relabeling (3-class):")
        print(f"  Total windows: {len(X)}")
        for i, name in enumerate(class_names):
            count = np.sum(y_relabeled == i)
            print(f"  {name} ({i}): {count} ({count/len(y_relabeled)*100:.1f}%)")
        if export_eda:
            counts = np.array([np.sum(y_relabeled == i) for i in range(len(class_names))])
            export_basic_eda(os.path.join(output_dir, 'eda'), counts, class_names)
    
    # Subject-wise split (LOSO-style)
    print(f"\nSplitting by subject (test={test_size}, val={val_size})...")
    unique_subjects = np.array(sorted(list(set(subjects_arr))))
    print(f"Total subjects: {len(unique_subjects)}")
    
    subj_trainval, subj_test = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )
    val_size_adjusted = val_size / (1 - test_size)
    subj_train, subj_val = train_test_split(
        subj_trainval, test_size=val_size_adjusted, random_state=random_state
    )
    
    # Create masks
    train_mask = np.isin(subjects_arr, subj_train)
    val_mask = np.isin(subjects_arr, subj_val)
    test_mask = np.isin(subjects_arr, subj_test)
    
    X_train, y_train = X[train_mask], y_relabeled[train_mask]
    X_val, y_val = X[val_mask], y_relabeled[val_mask]
    X_test, y_test = X[test_mask], y_relabeled[test_mask]
    
    # ✅ EXTRACT SUBJECT IDs per split
    subjects_train = subjects_arr[train_mask]
    subjects_val = subjects_arr[val_mask]
    subjects_test = subjects_arr[test_mask]
    
    print(f"Subject splits:")
    print(f"  Train: {len(subj_train)} subjects → {len(X_train)} windows")
    print(f"  Val: {len(subj_val)} subjects → {len(X_val)} windows")
    print(f"  Test: {len(subj_test)} subjects → {len(X_test)} windows")
    
    # Per-channel z-score normalization (using train statistics only)
    print("\nApplying per-channel z-score normalization (train-only stats)...")
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)
    train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    print(f"Normalized shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # ✅ SAVE BASIC DATA + SUBJECT IDs
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # ✅ SAVE SUBJECT IDs
    np.save(os.path.join(output_dir, 'subjects_train.npy'), subjects_train)
    np.save(os.path.join(output_dir, 'subjects_val.npy'), subjects_val)
    np.save(os.path.join(output_dir, 'subjects_test.npy'), subjects_test)
    
    # Save label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    
    # Save preprocessing info
    preprocessing_info = {
        'n_windows': len(X),
        'n_channels': X.shape[1],
        'window_size': X.shape[2],
        'n_classes': len(class_names),
        'class_names': class_names,
        'binary': binary,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'target_freq': target_freq,
        'overlap': overlap,
        'window_duration_s': window_size / target_freq,
        'channels': channel_names,
        'subject_splits': {
            'train_subjects': [str(s) for s in subj_train],
            'val_subjects': [str(s) for s in subj_val],
            'test_subjects': [str(s) for s in subj_test],
            'total_subjects': len(unique_subjects)
        },
        'normalization': 'per-channel z-score (train-only)',
        'files_processed': len(pkl_files),
        'parallel_processing': True,
        'n_workers': n_workers,
        'has_subject_ids': True  # ✅ Flag
    }
    
    joblib.dump(preprocessing_info, os.path.join(output_dir, "preprocessing_info.pkl"))
    
    print(f"\n✅ Basic preprocessing complete!")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ✅ AUTO-CREATE AUGMENTED DATA
    if create_augmentations:
        print(f"\n📊 Creating augmented training data (factor={n_augmentations+1}×)...")
        try:
            _create_augmented_data_internal(X_train, y_train, output_dir, n_augmentations)
            print("✅ Augmented data created!")
            preprocessing_info['has_augmented_data'] = True
            preprocessing_info['n_augmentations'] = n_augmentations
            joblib.dump(preprocessing_info, os.path.join(output_dir, "preprocessing_info.pkl"))
        except Exception as e:
            print(f"⚠️  Augmented data creation failed: {e}")
            preprocessing_info['has_augmented_data'] = False
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Preprocessing complete in {elapsed:.1f}s")
    print(f"  Classes: {class_names}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Window: {window_size} samples ({window_size/target_freq:.1f}s)")
    print(f"  Data saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return preprocessing_info


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_processed_wesad_temporal(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray,
                                                          LabelEncoder, Dict]:
    """Load preprocessed WESAD temporal data (basic format)."""
    print(f"Loading processed WESAD data from {data_dir}...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))
    
    print(f"Loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {info['class_names']}")
    print(f"  Channels: {info['n_channels']}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info


def load_augmented_wesad_temporal(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray,
                                                           LabelEncoder, Dict]:
    """
    Load preprocessed WESAD data with pre-calculated augmentation (training only).
    
    Returns:
        Tuple of (X_train_aug, X_val, X_test, y_train_aug, y_val, y_test, label_encoder, info)
    """
    print(f"Loading augmented WESAD data from {data_dir}...")

    # Load augmented training data
    X_train = np.load(os.path.join(data_dir, 'X_train_augmented.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_augmented.npy'))

    # Load non-augmented validation and test data
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))

    print(f"Loaded augmented data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['class_names']}")
    
    if 'n_augmentations' in info:
        print(f"  Augmentation factor: {info['n_augmentations'] + 1}×")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD dataset')
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
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, 
                       help='Validation set size')
    parser.add_argument('--binary', action='store_true', 
                       help='Binary classification (stress vs non-stress)')
    parser.add_argument('--multiclass', dest='binary', action='store_false', 
                       help='3-class (baseline/stress/amusement)')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--no_augmentations', action='store_true', 
                       help='Skip automatic augmentation')
    parser.add_argument('--n_augmentations', type=int, default=2, 
                       help='Number of augmentations per sample')
    parser.set_defaults(binary=True)
    
    args = parser.parse_args()
    
    print("="*70)
    print("WESAD PREPROCESSING")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.data_dir):
        print(f"Raw data not found: {args.data_dir}")
        exit(1)
    
    print(f"Raw data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target frequency: {args.target_freq} Hz")
    print(f"Window size: {args.window_size} samples")
    print(f"Create augmentations: {not args.no_augmentations}")
    if not args.no_augmentations:
        print(f"Augmentations per sample: {args.n_augmentations}")
    
    # Run preprocessing
    info = preprocess_wesad_temporal(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_freq=args.target_freq,
        window_size=args.window_size,
        overlap=args.overlap,
        test_size=args.test_size,
        val_size=args.val_size,
        binary=args.binary,
        random_state=args.random_state,
        create_augmentations=not args.no_augmentations,
        n_augmentations=args.n_augmentations
    )
    
    print(f"✅ WESAD preprocessing completed!")
    print(f"Preprocessing info:\n{info}")