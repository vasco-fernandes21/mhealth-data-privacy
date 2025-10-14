"""
WESAD Dataset Preprocessing Module (Optimized)

Streamlined preprocessing for WESAD stress detection:
- Load pickle files with physiological signals from RespiBAN and Empatica E4
- Resample to uniform frequency (32 Hz - optimal for ECG/BVP preservation)
- Temporal windowing (60s windows, 50% overlap)
- Subject-wise splitting (LOSO-style to avoid leakage)
- Per-channel z-score normalization (train-only)
- Binary (stress vs non-stress) or 3-class (baseline/stress/amusement)

WESAD contains:
- 15 subjects (~100 min each)
- RespiBAN (chest): ECG, EDA, EMG, Temp, Resp, ACC (700 Hz)
- Empatica E4 (wrist): ACC, BVP, EDA, TEMP (various frequencies)
- Labels: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=other
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
warnings.filterwarnings('ignore')
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_wesad_file(pkl_path: str) -> Dict:
    """
    Load a single WESAD pickle file.
    
    Returns:
        Dictionary with signals, labels, and subject ID
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract chest signals (RespiBAN) - 700 Hz
    chest = data['signal']['chest']
    # Extract wrist signals (Empatica E4) - various frequencies
    wrist = data['signal']['wrist']
    
    return {
        # Chest signals (700 Hz)
        'ecg': chest['ECG'],
        'eda_chest': chest['EDA'],
        'temp_chest': chest['Temp'],
        'acc_chest': chest['ACC'],
        'emg': chest['EMG'],
        'resp': chest['Resp'],
        # Wrist signals (Empatica E4)
        'acc_wrist': wrist['ACC'],      # 32 Hz
        'bvp': wrist['BVP'],            # 64 Hz
        'eda_wrist': wrist['EDA'],      # 4 Hz
        'temp_wrist': wrist['TEMP'],    # 4 Hz
        # Labels and metadata
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


def resample_signal(arr: np.ndarray, orig_freq: int, target_freq: int) -> np.ndarray:
    """
    Resample signal using polyphase filtering.
    Handles 1D and 2D (multi-channel) arrays.
    """
    if arr.ndim == 1:
        return signal.resample_poly(arr, int(target_freq), int(orig_freq), axis=0)
    # Multi-channel: resample each column
    resampled_cols = []
    for c in range(arr.shape[1]):
        resampled_cols.append(signal.resample_poly(arr[:, c], int(target_freq), int(orig_freq), axis=0))
    return np.stack(resampled_cols, axis=1)


def resample_all_signals(data: Dict, target_freq: int = 32) -> Dict:
    """
    Resample all signals to target frequency.
    
    Args:
        data: Dictionary from load_wesad_file
        target_freq: Target sampling frequency (Hz) - default 32 Hz for optimal signal quality
    
    Returns:
        Dictionary with resampled signals
    """
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
    """
    Apply bandpass/lowpass filters to improve SNR.
    Optimized for 32 Hz sampling to preserve signal quality.
    
    Args:
        resampled: Dictionary with resampled signals
        target_freq: Sampling frequency
    
    Returns:
        Dictionary with filtered signals
    """
    filtered = resampled.copy()
    
    try:
        # ECG: 0.5-15 Hz (capture heart rate and R-peaks)
        # With 32 Hz, we can preserve up to 15 Hz without aliasing
        sos_ecg = signal.butter(4, [0.5, min(15, target_freq/2.1)], btype='band', fs=target_freq, output='sos')
        filtered['ecg'] = signal.sosfiltfilt(sos_ecg, resampled['ecg'], axis=0)
        
        # BVP: 0.5-12 Hz (pulse waveform details)
        # Higher frequency limit to preserve pulse morphology
        sos_bvp = signal.butter(4, [0.5, min(12, target_freq/2.1)], btype='band', fs=target_freq, output='sos')
        filtered['bvp'] = signal.sosfiltfilt(sos_bvp, resampled['bvp'], axis=0)
        
        # ACC: 0.1-2 Hz (body movement, slightly higher for 32 Hz)
        sos_acc = signal.butter(4, [0.1, min(2, target_freq/2.1)], btype='band', fs=target_freq, output='sos')
        filtered['acc_chest'] = signal.sosfiltfilt(sos_acc, resampled['acc_chest'], axis=0)
        filtered['acc_wrist'] = signal.sosfiltfilt(sos_acc, resampled['acc_wrist'], axis=0)
        
        # EDA: lowpass 1.5 Hz (slow variations, slightly higher for 32 Hz)
        sos_eda = signal.butter(4, min(1.5, target_freq/2.1), btype='low', fs=target_freq, output='sos')
        filtered['eda_chest'] = signal.sosfiltfilt(sos_eda, resampled['eda_chest'], axis=0)
        filtered['eda_wrist'] = signal.sosfiltfilt(sos_eda, resampled['eda_wrist'], axis=0)
        
        # TEMP: lowpass 0.5 Hz (very slow)
        sos_temp = signal.butter(4, min(0.5, target_freq/2.1), btype='low', fs=target_freq, output='sos')
        filtered['temp_chest'] = signal.sosfiltfilt(sos_temp, resampled['temp_chest'], axis=0)
        filtered['temp_wrist'] = signal.sosfiltfilt(sos_temp, resampled['temp_wrist'], axis=0)
        
        # Resp: 0.1-0.5 Hz (breathing rate)
        sos_resp = signal.butter(4, [0.1, min(0.5, target_freq/2.1)], btype='band', fs=target_freq, output='sos')
        filtered['resp'] = signal.sosfiltfilt(sos_resp, resampled['resp'], axis=0)
        
        # EMG: 0.5-2 Hz (muscle activity, higher limit for 32 Hz)
        sos_emg = signal.butter(4, [0.5, min(2, target_freq/2.1)], btype='band', fs=target_freq, output='sos')
        filtered['emg'] = signal.sosfiltfilt(sos_emg, resampled['emg'], axis=0)
        
    except Exception as e:
        print(f"  Warning: filtering failed ({e}), using unfiltered signals")
        return resampled
    
    return filtered


def _clip_array_percentiles(arr: np.ndarray, lower_p: float, upper_p: float) -> np.ndarray:
    """
    Clip array values to [lower_p, upper_p] percentiles per channel to reduce outliers.
    Works for 1D and 2D arrays (time x channels).
    """
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


def reduce_outliers(signals: Dict, method: str = 'clip', lower_p: float = 0.5, upper_p: float = 99.5) -> Dict:
    """
    Basic outlier mitigation prior to windowing.
    - method='clip': percentile clipping per signal/channel (default)
    - method='none': no change
    """
    if method == 'none':
        return signals
    clipped = signals.copy()
    keys = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp', 'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']
    for k in keys:
        if k in clipped and clipped[k] is not None:
            try:
                clipped[k] = _clip_array_percentiles(clipped[k], lower_p, upper_p)
            except Exception:
                # If clipping fails for any reason, keep original
                pass
    return clipped


def export_basic_eda(eda_dir: str, class_counts: np.ndarray, class_names: List[str]) -> None:
    """
    Save a simple class distribution bar chart to eda_dir, if matplotlib is available.
    """
    if plt is None:
        return
    try:
        os.makedirs(eda_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(class_names, class_counts)
        ax.set_title('WESAD Class Distribution')
        ax.set_ylabel('Num windows')
        for i, v in enumerate(class_counts):
            ax.text(i, v + max(class_counts)*0.01, str(int(v)), ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'class_distribution.png'), dpi=150)
        plt.close(fig)
    except Exception:
        pass

def create_temporal_windows(data: Dict, window_size: int = 1920, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create temporal sliding windows for LSTM/CNN.
    Only includes complete windows to avoid zero-padding artifacts.
    
    Args:
        data: Dictionary with filtered signals
        window_size: Window size in samples (default: 1920 = 60s at 32 Hz)
        overlap: Overlap ratio (0.0 to 0.9)
    
    Returns:
        Tuple of (windows, labels, channel_names)
        - windows: (n_windows, n_channels, window_size)
        - labels: (n_windows,)
        - channel_names: list of channel names
    """
    step_size = int(window_size * (1 - overlap))
    ref_length = len(data['ecg'])
    
    # Define signals to use and expand multi-channel
    signal_names = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp',
                    'bvp', 'eda_wrist', 'temp_wrist', 'acc_wrist']
    
    channel_names = []
    for name in signal_names:
        if name not in data:
            continue
        arr = data[name]
        if arr.ndim > 1 and arr.shape[1] > 1:
            # Multi-channel: label axes
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
        
        # Check if all signals have enough data for this window
        all_signals_complete = True
        for name in signal_names:
            if name not in data:
                continue
            arr = data[name]
            if len(arr) < end_idx:
                all_signals_complete = False
                break
        
        # Skip incomplete windows to avoid zero-padding artifacts
        if not all_signals_complete:
            continue
        
        window_data = np.zeros((n_channels, window_size))
        ch_idx = 0
        
        for name in signal_names:
            if name not in data:
                continue
            arr = data[name]
            
            if arr.ndim > 1 and arr.shape[1] > 1:
                # Multi-channel
                for c in range(arr.shape[1]):
                    window_data[ch_idx] = arr[start_idx:end_idx, c]
                    ch_idx += 1
            else:
                # Single channel
                series = arr if arr.ndim == 1 else arr[:, 0]
                window_data[ch_idx] = series[start_idx:end_idx]
                ch_idx += 1
        
        # Get label for this window (majority vote, excluding undefined=0)
        window_labels = data['labels'][start_idx:end_idx]
        valid_labels = window_labels[window_labels != 0]
        if len(valid_labels) > 0:
            window_label = np.bincount(valid_labels.astype(int)).argmax()
        else:
            window_label = 0  # undefined
        
        windows_list.append(window_data)
        labels_list.append(window_label)
    
    return np.array(windows_list), np.array(labels_list), channel_names


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
                               export_eda: bool = True) -> Dict:
    """
    Complete preprocessing pipeline for WESAD (temporal windows for LSTM/CNN).
    
    Args:
        data_dir: Directory containing WESAD pickle files (S2/, S3/, ...)
        output_dir: Directory to save processed data
        target_freq: Target sampling frequency (Hz) - default 32 Hz (optimal signal quality)
        window_size: Window size in samples (default: 1920 = 60s at 32 Hz)
        overlap: Overlap ratio (0.0-0.9)
        test_size: Test set size ratio (subject-wise split)
        val_size: Validation set size ratio (subject-wise split)
        binary: If True, binary classification (stress vs non-stress).
                If False, 3-class (baseline/stress/amusement)
        random_state: Random seed
        outlier_method: Method for outlier handling ('clip' or 'none')
        clip_lower_p: Lower percentile for clipping (default: 1.0% - less aggressive)
        clip_upper_p: Upper percentile for clipping (default: 99.0% - less aggressive)
        export_eda: Whether to export EDA plots
    
    Returns:
        Dictionary with preprocessing info
    
    Note:
        Default 32 Hz chosen for optimal signal quality:
        - ECG: Preserves R-peak resolution and HRV analysis
        - BVP: Captures pulse waveform details without aliasing
        - ACC: Preserves movement dynamics
        - Less aggressive clipping (1-99%) preserves important signal peaks
        - Use 16 Hz for efficiency or 4 Hz for edge deployment if needed.
    """
    print("="*70)
    print("WESAD TEMPORAL PREPROCESSING (Optimized)")
    print("="*70)
    print(f"Target frequency: {target_freq} Hz")
    print(f"Window size: {window_size} samples ({window_size/target_freq:.1f} seconds)")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Classification: {'Binary (stress vs non-stress)' if binary else '3-class (baseline/stress/amusement)'}")
    print(f"\nNote: {target_freq} Hz preserves R-peaks (ECG), pulse details (BVP), and movement dynamics (ACC)")
    print(f"Outlier handling: {outlier_method} (clip {clip_lower_p}-{clip_upper_p} percentiles - less aggressive)\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    pkl_files = glob.glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} pickle files\n")
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {data_dir}")
    
    all_windows = []
    all_labels = []
    all_subjects = []
    channel_names = None
    
    start_time = time.time()
    
    for file_idx, pkl_file in enumerate(pkl_files, 1):
        try:
            print(f"[{file_idx}/{len(pkl_files)}] Processing {os.path.basename(pkl_file)}...")
            
            # Load and resample
            data = load_wesad_file(pkl_file)
            subject = data['subject']
            resampled = resample_all_signals(data, target_freq)
            
            # Apply simple outlier mitigation before filtering
            outlier_reduced = reduce_outliers(resampled, method=outlier_method, lower_p=clip_lower_p, upper_p=clip_upper_p)
            
            # Apply filters
            filtered = apply_filters(outlier_reduced, target_freq)
            
            # Create temporal windows
            windows, labels, ch_names = create_temporal_windows(filtered, window_size, overlap)
            
            if channel_names is None:
                channel_names = ch_names
            
            all_windows.append(windows)
            all_labels.append(labels)
            all_subjects.extend([subject] * len(windows))
            
            print(f"  → {len(windows)} windows, shape: {windows.shape}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
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
        # Binary: stress (2) vs non-stress (1=baseline, 3=amusement)
        # Keep labels 1, 2, 3
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        # Relabel: 1,3 → 0 (non-stress), 2 → 1 (stress)
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
        # 3-class: baseline (1), stress (2), amusement (3)
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]
        
        # Relabel: 1→0 (baseline), 2→1 (stress), 3→2 (amusement)
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
    
    # Subject-wise split (LOSO-style to avoid leakage)
    print(f"\nSplitting by subject (test={test_size}, val={val_size})...")
    unique_subjects = np.array(sorted(list(set(subjects_arr))))
    print(f"Total subjects: {len(unique_subjects)}")
    
    # Split subjects: train+val vs test
    subj_trainval, subj_test = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )
    # Split train+val: train vs val
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
    
    print(f"Subject splits:")
    print(f"  Train: {len(subj_train)} subjects → {len(X_train)} windows")
    print(f"  Val: {len(subj_val)} subjects → {len(X_val)} windows")
    print(f"  Test: {len(subj_test)} subjects → {len(X_test)} windows")
    
    # Per-channel z-score normalization (using train statistics only)
    print("\nApplying per-channel z-score normalization (train-only stats)...")
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # shape (1, n_channels, 1)
    train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    print(f"Normalized shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Save data
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
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
            'test_subjects': [str(s) for s in subj_test]
        },
        'normalization': 'per-channel z-score (train-only)',
        'files_processed': len(pkl_files)
    }
    
    joblib.dump(preprocessing_info, os.path.join(output_dir, "preprocessing_info.pkl"))
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Preprocessing complete in {elapsed:.1f}s")
    print(f"  Classes: {class_names}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Window: {window_size} samples ({window_size/target_freq:.1f}s)")
    print(f"  Data saved to: {output_dir}")
    print(f"{'='*70}")
    
    return preprocessing_info


def load_processed_wesad_temporal(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray,
                                                          LabelEncoder, Dict]:
    """
    Load preprocessed WESAD temporal data.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info)
    """
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD dataset (optimized)')
    parser.add_argument('--data_dir', default='data/raw/wesad', help='Raw data directory')
    parser.add_argument('--output_dir', default='data/processed/wesad', help='Output directory')
    parser.add_argument('--target_freq', type=int, default=32, 
                       help='Target sampling frequency (Hz). 32 Hz optimal for signal quality, 16 Hz for efficiency, 4 Hz for edge deployment.')
    parser.add_argument('--window_size', type=int, default=1920, 
                       help='Window size in samples (1920=60s at 32Hz, 960=60s at 16Hz, 240=60s at 4Hz)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--binary', action='store_true', help='Binary classification (stress vs non-stress)')
    parser.add_argument('--multiclass', dest='binary', action='store_false', help='3-class (baseline/stress/amusement)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.set_defaults(binary=True)
    
    args = parser.parse_args()
    
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
        random_state=args.random_state
    )
    
    print(f"\n✓ Done! Preprocessing info saved.")
