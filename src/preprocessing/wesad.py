#!/usr/bin/env python3
"""
WESAD Dataset Preprocessing (FEATURES ONLY)
Direct conversion from raw signals to 140D features (10 per channel × 14 channels)
Optimized for memory efficiency and stability.
"""

import numpy as np
import pickle
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import glob
import gc
from typing import Tuple, Dict, List, Optional
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


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


def resample_signal(
    arr: np.ndarray, orig_freq: int, target_freq: int
) -> np.ndarray:
    """Resample signal using polyphase filtering."""
    if arr.ndim == 1:
        return signal.resample_poly(
            arr, int(target_freq), int(orig_freq), axis=0
        ).astype(np.float32)

    resampled_cols = []
    for c in range(arr.shape[1]):
        resampled_cols.append(
            signal.resample_poly(
                arr[:, c], int(target_freq), int(orig_freq), axis=0
            ).astype(np.float32)
        )
    return np.stack(resampled_cols, axis=1)


def downsample_labels_temporal(
    labels: np.ndarray, orig_freq: int, target_freq: int
) -> np.ndarray:
    """Downsample labels using temporal mapping."""
    orig_length = len(labels)
    target_length = int(
        np.round(orig_length * (target_freq / orig_freq))
    )
    indices = np.round(
        np.linspace(0, orig_length - 1, target_length)
    ).astype(int)
    return labels[indices]


def resample_all_signals(
    data: Dict, target_freq: int = 32
) -> Dict:
    """Resample all signals to target frequency."""
    sfreq = data['sfreq']
    resampled = {}

    resampled['ecg'] = resample_signal(
        data['ecg'], sfreq['chest'], target_freq
    )
    resampled['eda_chest'] = resample_signal(
        data['eda_chest'], sfreq['chest'], target_freq
    )
    resampled['temp_chest'] = resample_signal(
        data['temp_chest'], sfreq['chest'], target_freq
    )
    resampled['acc_chest'] = resample_signal(
        data['acc_chest'], sfreq['chest'], target_freq
    )
    resampled['emg'] = resample_signal(
        data['emg'], sfreq['chest'], target_freq
    )
    resampled['resp'] = resample_signal(
        data['resp'], sfreq['chest'], target_freq
    )

    resampled['acc_wrist'] = resample_signal(
        data['acc_wrist'], sfreq['acc_wrist'], target_freq
    )
    resampled['bvp'] = resample_signal(
        data['bvp'], sfreq['bvp'], target_freq
    )
    resampled['eda_wrist'] = resample_signal(
        data['eda_wrist'], sfreq['eda_wrist'], target_freq
    )
    resampled['temp_wrist'] = resample_signal(
        data['temp_wrist'], sfreq['temp_wrist'], target_freq
    )

    resampled['labels'] = downsample_labels_temporal(
        data['labels'], sfreq['chest'], target_freq
    )
    resampled['subject'] = data['subject']
    resampled['sfreq'] = target_freq

    return resampled


def apply_filters(
    resampled: Dict, target_freq: int = 32
) -> Dict:
    """Apply bandpass/lowpass filters."""
    filtered = resampled.copy()

    try:
        sos_ecg = signal.butter(
            4, [0.5, 15], btype='band', fs=target_freq, output='sos'
        )
        sos_bvp = signal.butter(
            4, [0.5, 12], btype='band', fs=target_freq, output='sos'
        )
        sos_acc = signal.butter(
            4, [0.1, 2], btype='band', fs=target_freq, output='sos'
        )
        sos_eda = signal.butter(
            4, 1.5, btype='low', fs=target_freq, output='sos'
        )
        sos_resp = signal.butter(
            4, [0.1, 0.5], btype='band', fs=target_freq, output='sos'
        )
    except Exception as e:
        print(f"WARNING: filter design failed ({e})")
        return resampled

    signal_filters = {
        'ecg': sos_ecg, 'bvp': sos_bvp, 'acc_chest': sos_acc,
        'acc_wrist': sos_acc, 'eda_chest': sos_eda, 'eda_wrist': sos_eda,
        'resp': sos_resp, 'emg': sos_ecg, 'temp_chest': sos_eda,
        'temp_wrist': sos_eda
    }

    for key, sos_filter in signal_filters.items():
        if key in filtered and filtered[key] is not None:
            try:
                filtered[key] = signal.sosfiltfilt(
                    sos_filter, filtered[key], axis=0
                )
            except Exception:
                pass

    return filtered


def reduce_outliers(
    signals: Dict, lower_p: float = 1.0, upper_p: float = 99.0
) -> Dict:
    """Clip outliers using percentile method."""
    clipped = signals.copy()
    keys = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp',
            'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']

    for k in keys:
        if k in clipped and clipped[k] is not None:
            arr = clipped[k]
            if arr.ndim == 1:
                lo, hi = np.percentile(arr, lower_p), \
                         np.percentile(arr, upper_p)
                clipped[k] = np.clip(arr, lo, hi)
            else:
                for c in range(arr.shape[1]):
                    lo, hi = np.percentile(arr[:, c], lower_p), \
                             np.percentile(arr[:, c], upper_p)
                    arr[:, c] = np.clip(arr[:, c], lo, hi)

    return clipped


def extract_10_features(series: np.ndarray) -> List[float]:
    """
    Extract 10 stable statistical features from a signal.
    Avoids problematic features like skewness and entropy.
    """
    series = series.astype(np.float64)

    features = [
        float(np.mean(series)),
        float(np.std(series)),
        float(np.min(series)),
        float(np.max(series)),
        float(np.median(series)),
        float(np.percentile(series, 25)),
        float(np.percentile(series, 75)),
        float(np.ptp(series)),
        float(np.sqrt(np.mean(series**2))),
        float(np.sum(np.abs(np.diff(series))))
    ]

    return features


def extract_window_features(
    signal_dict: Dict, start_idx: int, end_idx: int
) -> Optional[np.ndarray]:
    """
    Extract 10 features from each of 14 channels.
    Returns array of shape (140,) for 14 channels × 10 features.

    Canais:
      1D: ecg, eda_chest, temp_chest, emg, resp, bvp, eda_wrist, temp_wrist
      3D: acc_chest (x,y,z), acc_wrist (x,y,z)
      Total: 8 × 1 + 2 × 3 = 14 canais
    """
    features = []

    signal_names_1d = ['ecg', 'eda_chest', 'temp_chest', 'emg', 'resp',
                       'bvp', 'eda_wrist', 'temp_wrist']
    signal_names_3d = [('acc_chest', 3), ('acc_wrist', 3)]

    # 1D signals
    for name in signal_names_1d:
        if name not in signal_dict:
            continue

        arr = signal_dict[name]
        series = arr if arr.ndim == 1 else arr[:, 0]
        series = series[start_idx:end_idx]

        if len(series) == 0:
            return None

        ch_features = extract_10_features(series)
        features.extend(ch_features)

    # 3D signals (ACC)
    for name, n_dims in signal_names_3d:
        if name not in signal_dict:
            continue

        arr = signal_dict[name]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[1] != n_dims:
            continue

        for c in range(n_dims):
            series = arr[start_idx:end_idx, c]
            if len(series) == 0:
                return None

            ch_features = extract_10_features(series)
            features.extend(ch_features)

    # Verificar dimensionalidade final
    if len(features) != 140:
        return None

    return np.array(features, dtype=np.float32)


def create_features_from_windows(
    data: Dict, window_size: int = 1024, overlap: float = 0.75,
    label_threshold: float = 0.7, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Create features from sliding windows.
    Returns shape (n_windows, 140) for 140D features.
    """
    np.random.seed(random_state)

    step_size = int(window_size * (1 - overlap))
    ref_length = len(data['ecg'])

    signal_names = ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg',
                    'resp', 'bvp', 'eda_wrist', 'temp_wrist', 'acc_wrist']

    features_list = []
    labels_list = []
    n_dropped = 0

    for start_idx in range(0, ref_length - window_size + 1, step_size):
        end_idx = start_idx + window_size

        if not all(
            len(data.get(name, [])) >= end_idx
            for name in signal_names if name in data
        ):
            continue

        window_features = extract_window_features(data, start_idx, end_idx)

        if window_features is None or len(window_features) != 140:
            n_dropped += 1
            continue

        window_labels = data['labels'][start_idx:end_idx]
        valid_labels = window_labels[window_labels != 0]

        if len(valid_labels) < window_size * label_threshold:
            n_dropped += 1
            continue

        if len(valid_labels) > 0:
            window_label = int(np.bincount(valid_labels.astype(int)).argmax())
        else:
            n_dropped += 1
            continue

        features_list.append(window_features)
        labels_list.append(window_label)

    if not features_list:
        return (
            np.empty((0, 140), dtype=np.float32),
            np.array([], dtype=np.uint8),
            n_dropped
        )

    return (
        np.stack(features_list, axis=0).astype(np.float32),
        np.array(labels_list, dtype=np.uint8),
        n_dropped
    )


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


def process_single_wesad_file(
    args: Tuple
) -> Tuple[Optional[Dict], Optional[str]]:
    """Process a single WESAD file directly to features."""
    (pkl_file, target_freq, window_size, overlap,
     label_threshold, random_state) = args

    try:
        data = load_wesad_file(pkl_file)
        subject = data['subject']

        resampled = resample_all_signals(data, target_freq)
        outlier_reduced = reduce_outliers(resampled)
        filtered = apply_filters(outlier_reduced, target_freq)

        features, labels, n_dropped = create_features_from_windows(
            filtered, window_size, overlap, label_threshold, random_state
        )

        del data, resampled, outlier_reduced, filtered
        gc.collect()

        if len(features) == 0:
            return None, subject

        return {
            'features': features,
            'labels': labels,
            'subject': subject,
            'n_dropped': n_dropped
        }, subject

    except Exception as e:
        print(f"  ERROR: {os.path.basename(pkl_file)}: {e}")
        return None, None


def preprocess_wesad(
    data_dir: str,
    output_dir: str,
    target_freq: int = 32,
    window_size: int = 1024,
    overlap: float = 0.75,
    label_threshold: float = 0.7,
    binary: bool = True,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_workers: int = None,
    force_reprocess: bool = False
) -> Dict:
    """
    Preprocess WESAD directly to 140D features (10 per channel × 14 channels).
    No intermediate temporal storage - memory efficient.
    """
    print("="*70)
    print("WESAD PREPROCESSING (140D FEATURES)")
    print("="*70)
    print(f"Target freq: {target_freq} Hz")
    print(f"Window: {window_size/target_freq:.1f}s ({window_size} samples)")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Features: 140D (10 per channel × 14 channels)")
    print(f"  - 8 channels 1D: ecg, eda_chest, temp_chest, emg, resp, bvp, "
          f"eda_wrist, temp_wrist")
    print(f"  - 6 channels 3D: acc_chest(x,y,z), acc_wrist(x,y,z)\n")

    os.makedirs(output_dir, exist_ok=True)

    required_files = ['X_train.npy', 'X_val.npy', 'X_test.npy',
                     'y_train.npy', 'y_val.npy', 'y_test.npy']
    all_exist = all(
        os.path.exists(os.path.join(output_dir, f))
        for f in required_files
    )

    if force_reprocess:
        print("Force reprocess enabled - cleaning up old files...")
        for f in required_files + ['metadata.pkl', 'scaler.pkl']:
            filepath = os.path.join(output_dir, f)
            if os.path.exists(filepath):
                os.remove(filepath)

    if all_exist and not force_reprocess:
        print("Already preprocessed!")
        try:
            info = joblib.load(os.path.join(output_dir, "metadata.pkl"))
            return info
        except Exception:
            return {}

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    pkl_files = glob.glob(
        os.path.join(data_dir, "**/*.pkl"), recursive=True
    )
    print(f"Found {len(pkl_files)} pickle files\n")

    if not pkl_files:
        raise ValueError(f"No pickle files found in {data_dir}")

    process_args = [
        (pkl_file, target_freq, window_size, overlap, label_threshold,
         random_state + i)
        for i, pkl_file in enumerate(pkl_files)
    ]

    print(f"Processing with {n_workers} workers...")
    start_time = time.time()

    all_features = []
    all_labels = []
    all_subjects = []

    with Pool(processes=n_workers) as pool:
        for idx, (result, subject) in enumerate(
            pool.imap_unordered(
                process_single_wesad_file, process_args, chunksize=2
            )
        ):
            if result is not None:
                all_features.append(result['features'])
                all_labels.append(result['labels'])
                all_subjects.extend([result['subject']] * len(result['labels']))

                del result
                gc.collect()

            if (idx + 1) % 2 == 0:
                gc.collect()

    elapsed_processing = time.time() - start_time
    print(f"Processed in {elapsed_processing:.1f}s\n")

    print("Combining features...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    subjects_arr = np.array(all_subjects, dtype=object)

    del all_features, all_labels, all_subjects
    gc.collect()

    print(f"Combined: {X.shape}\n")

    # Binary classification
    if binary:
        valid_mask = (y >= 1) & (y <= 3)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects_arr = subjects_arr[valid_mask]

        y_binary = (y == 2).astype(np.uint8)
        class_names = ['non-stress', 'stress']

        n_nonstress = np.sum(y_binary == 0)
        n_stress = np.sum(y_binary == 1)
        print(f"Binary classification (stress detection):")
        print(f"  Non-stress: {n_nonstress:5d} "
              f"({100*n_nonstress/len(y_binary):.1f}%)")
        print(f"  Stress:     {n_stress:5d} "
              f"({100*n_stress/len(y_binary):.1f}%)\n")
    else:
        y_binary = y
        class_names = ['baseline', 'stress', 'amusement']

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

    X_val = X[val_mask]
    y_val = y_binary[val_mask]

    X_test = X[test_mask]
    y_test = y_binary[test_mask]

    print(f"  Train: {len(subj_train)} subjects -> {len(X_train)} samples")
    print(f"  Val:   {len(subj_val)} subjects -> {len(X_val)} samples")
    print(f"  Test:  {len(subj_test)} subjects -> {len(X_test)} samples\n")

    del X, y, y_binary, subjects_arr
    gc.collect()

    print(f"Normalization: StandardScaler (global)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print("Normalized\n")

    weights, class_counts = compute_class_weights(y_train)
    print(f"Class weights (for loss function):")
    for c, w in weights.items():
        print(f"  {class_names[c]}: {w:.3f}")
    print()

    print(f"Saving to {output_dir}...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    total_size_mb = (
        X_train.nbytes + X_val.nbytes + X_test.nbytes
    ) / 1e6

    metadata = {
        'n_features': 140,
        'features_per_channel': 10,
        'n_channels': 14,
        'channel_description': {
            '1D_channels': ['ecg', 'eda_chest', 'temp_chest', 'emg', 'resp',
                           'bvp', 'eda_wrist', 'temp_wrist'],
            '3D_channels': ['acc_chest (x,y,z)', 'acc_wrist (x,y,z)']
        },
        'class_names': class_names,
        'class_weights': weights,
        'class_counts_train': {
            class_names[int(c)]: int(cnt)
            for c, cnt in enumerate(np.bincount(y_train))
        },
        'n_classes': len(class_names),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_size_mb': total_size_mb,
        'total_subjects': len(unique_subjects),
        'processing_time_s': elapsed_processing,
    }

    joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))
    print("Saved\n")

    print("="*70)
    print(f"PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total samples: {len(X_train) + len(X_val) + len(X_test)}")
    print(f"Features: 140D (10 per channel × 14 channels)")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Processing time: {elapsed_processing:.1f}s\n")

    return metadata


def load_processed_wesad(data_dir: str) -> Tuple:
    """
    Load preprocessed WESAD data (features-only).

    Args:
        data_dir: Path to processed WESAD directory

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, info)
    """
    print(f"Loading WESAD from {data_dir}...")

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    info = joblib.load(os.path.join(data_dir, 'metadata.pkl'))

    print(f"Loaded WESAD:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    print(f"  Total size: {info.get('total_size_mb', 0):.1f} MB\n")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess WESAD to features (140D)'
    )
    parser.add_argument('--data_dir', default='data/raw/wesad')
    parser.add_argument('--output_dir', default='data/processed/wesad')
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size (1024 @ 32Hz = 32s)')
    parser.add_argument('--overlap', type=float, default=0.75,
                       help='Window overlap (0.75 = 75%)')
    parser.add_argument('--target_freq', type=int, default=32,
                       help='Target sampling frequency')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--force_reprocess', action='store_true')

    args = parser.parse_args()

    info = preprocess_wesad(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        target_freq=args.target_freq,
        test_size=args.test_size,
        val_size=args.val_size,
        n_workers=args.n_workers,
        force_reprocess=args.force_reprocess
    )

    print(f"Preprocessing complete!")