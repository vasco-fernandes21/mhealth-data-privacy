#!/usr/bin/env python3
"""
Sleep-EDF Dataset Preprocessing
- Non-overlapping temporal windows based on epochs
- HDF5 optimized storage for fast loading
- Pre-allocation of arrays
- Pre-computed windows for fast loading
"""

import numpy as np
import pyedflib
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import h5py
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
            except Exception:
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
            except Exception:
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

        del signals, filtered_signals
        gc.collect()

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
    window_epochs: int = 10,
    stride: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping windows from epochs with majority label."""
    if stride is None:
        stride = window_epochs

    n_epochs = X.shape[0]
    n_channels, samples_per_epoch = X.shape[1:]

    if n_epochs < window_epochs:
        return (
            np.empty(
                (0, window_epochs, n_channels * samples_per_epoch),
                dtype=np.float32
            ),
            np.array([], dtype=y.dtype),
            np.array([], dtype=object)
        )

    windows_list = []
    labels_list = []
    subjects_list = []

    for start_idx in range(0, n_epochs - window_epochs + 1, stride):
        end_idx = start_idx + window_epochs

        window = X[start_idx:end_idx].reshape(
            window_epochs, n_channels * samples_per_epoch
        )
        windows_list.append(window)

        window_labels = y[start_idx:end_idx]
        label_majority = int(np.bincount(window_labels).argmax())
        labels_list.append(label_majority)

        subjects_list.append(subjects[end_idx - 1])

    if not windows_list:
        return (
            np.empty(
                (0, window_epochs, n_channels * samples_per_epoch),
                dtype=np.float32
            ),
            np.array([], dtype=y.dtype),
            np.array([], dtype=object)
        )

    X_windows = np.stack(windows_list, axis=0).astype(np.float32)
    y_windows = np.array(labels_list, dtype=y.dtype)
    subj_windows = np.array(subjects_list, dtype=object)

    n_dropped = n_epochs - (len(windows_list) * stride + window_epochs - 1)
    if n_dropped > 0:
        print(f"    Created {len(windows_list)} windows "
              f"(dropped {n_dropped} epochs)")
    else:
        print(f"    Created {len(windows_list)} windows")

    gc.collect()

    return X_windows, y_windows, subj_windows


def save_windowed_data_optimized(
    output_dir: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    subjects_train, subjects_val, subjects_test,
    scaler, metadata
):
    """Save data in optimized HDF5 format for fast training."""
    print("Saving in optimized HDF5 format...")

    hdf5_path = os.path.join(output_dir, 'sleep_edf_data.h5')

    with h5py.File(hdf5_path, 'w') as f:
        print("  Saving training data...")
        f.create_dataset('X_train', data=X_train,
                        compression='gzip', compression_opts=1,
                        chunks=True, dtype=np.float32)
        f.create_dataset('X_val', data=X_val,
                        compression='gzip', compression_opts=1,
                        chunks=True, dtype=np.float32)
        f.create_dataset('X_test', data=X_test,
                        compression='gzip', compression_opts=1,
                        chunks=True, dtype=np.float32)

        f.create_dataset('y_train', data=y_train, dtype=np.uint8)
        f.create_dataset('y_val', data=y_val, dtype=np.uint8)
        f.create_dataset('y_test', data=y_test, dtype=np.uint8)

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('subjects_train', data=subjects_train, dtype=dt)
        f.create_dataset('subjects_val', data=subjects_val, dtype=dt)
        f.create_dataset('subjects_test', data=subjects_test, dtype=dt)

        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                f.attrs[key] = str(value)
            else:
                f.attrs[key] = value

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    old_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy'
    ]

    for f_name in old_files:
        filepath = os.path.join(output_dir, f_name)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

    file_size_gb = os.path.getsize(hdf5_path) / (1024**3)
    print(f"  Saved to {hdf5_path}")
    print(f"  Size: {file_size_gb:.2f} GB")
    print(f"  Estimated compression: ~70%")


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
    """Complete Sleep-EDF preprocessing pipeline."""
    print("="*70)
    print("SLEEP-EDF PREPROCESSING (OPTIMIZED HDF5)")
    print("="*70 + "\n")

    start_time_total = time.time()
    os.makedirs(output_dir, exist_ok=True)

    hdf5_path = os.path.join(output_dir, 'sleep_edf_data.h5')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    if os.path.exists(hdf5_path) and os.path.exists(scaler_path) and \
       not force_reprocess:
        print(f"Already preprocessed (optimized)! Loading metadata...\n")

        with h5py.File(hdf5_path, 'r') as f:
            info = dict(f.attrs)
            for key, value in info.items():
                if isinstance(value, str) and value.startswith('['):
                    try:
                        info[key] = eval(value)
                    except Exception:
                        pass

        return info

    if force_reprocess:
        print("Force reprocess enabled - cleaning up old files...")
        old_files = [
            'sleep_edf_data.h5',
            'X_train.npy', 'X_val.npy', 'X_test.npy',
            'y_train.npy', 'y_val.npy', 'y_test.npy',
            'subjects_train.npy', 'subjects_val.npy', 'subjects_test.npy',
            'preprocessing_info.pkl', 'scaler.pkl', 'metadata.pkl'
        ]

        removed_count = 0
        for f_name in old_files:
            filepath = os.path.join(output_dir, f_name)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Could not remove {f_name}: {e}")

        if removed_count > 0:
            print(f"  Removed {removed_count} old file(s)\n")

    print("Matching PSG-Hypnogram files...")
    psg_files = sorted(glob.glob(
        os.path.join(data_dir, '**/*-PSG.edf'), recursive=True
    ))
    hypno_files = sorted(glob.glob(
        os.path.join(data_dir, '**/*-Hypnogram.edf'), recursive=True
    ))

    print(f"Found {len(psg_files)} PSG files, {len(hypno_files)} Hypnogram "
          f"files")

    psg_by_subject = {extract_subject_id(f): f for f in psg_files}
    hypno_by_subject = {extract_subject_id(f): f for f in hypno_files}

    matched_subjects = set(psg_by_subject.keys()) & \
        set(hypno_by_subject.keys())
    file_pairs = [
        (psg_by_subject[s], hypno_by_subject[s])
        for s in sorted(matched_subjects)
    ]

    print(f"Matched {len(file_pairs)}/{max(len(psg_files), len(hypno_files))} "
          f"pairs\n")

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

    n_success = 0
    n_failed = 0

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
                n_success += 1
            else:
                n_failed += 1

            if (idx + 1) % 2 == 0:
                gc.collect()

    elapsed_processing = time.time() - start_time
    print(f"\nProcessed in {elapsed_processing:.1f}s")
    print(f"  Success: {n_success}/{len(file_pairs)} files")
    if n_failed > 0:
        print(f"  Failed:  {n_failed}/{len(file_pairs)} files\n")
    else:
        print()

    if len(all_epochs) == 0:
        raise RuntimeError(
            "No valid epochs found! All files failed to process. "
            "Check the raw data files."
        )

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

    n_windows_train = len(X_train_win)
    n_windows_val = len(X_val_win)
    n_windows_test = len(X_test_win)
    window_shape = X_train_win.shape[1:]

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
        'window_type': 'pre_computed_non_overlapping_hdf5',
        'train_subjects': [str(s) for s in split_info['train_subjects']],
        'val_subjects': [str(s) for s in split_info['val_subjects']],
        'test_subjects': [str(s) for s in split_info['test_subjects']],
        'total_subjects': len(unique_subjects),
        'n_workers': n_workers,
        'processing_time_s': elapsed_processing,
        'total_time_s': elapsed_total
    }

    save_windowed_data_optimized(
        output_dir,
        X_train_win, X_val_win, X_test_win,
        y_train_win, y_val_win, y_test_win,
        subj_train_win, subj_val_win, subj_test_win,
        scaler, metadata
    )

    del X_train_win, X_val_win, X_test_win
    del y_train_win, y_val_win, y_test_win
    gc.collect()

    print("\n" + "="*70)
    print(f"PREPROCESSING COMPLETE (HDF5 OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_total:.1f}s (processing: {elapsed_processing:.1f}s)\n")

    return metadata


def load_windowed_sleep_edf_optimized(data_dir: str) -> Tuple:
    """Load from optimized HDF5 format for maximum speed."""
    print(f"Loading Sleep-EDF from optimized format...")

    hdf5_path = os.path.join(data_dir, 'sleep_edf_data.h5')

    with h5py.File(hdf5_path, 'r') as f:
        print("  Loading training data...")
        X_train = f['X_train'][:]
        X_val = f['X_val'][:]
        X_test = f['X_test'][:]

        y_train = f['y_train'][:]
        y_val = f['y_val'][:]
        y_test = f['y_test'][:]

        subjects_train = f['subjects_train'][:].astype(str)
        subjects_val = f['subjects_val'][:].astype(str)
        subjects_test = f['subjects_test'][:].astype(str)

        info = dict(f.attrs)
        for key, value in info.items():
            if isinstance(value, str) and value.startswith('['):
                try:
                    info[key] = eval(value)
                except Exception:
                    pass

    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    subjects = np.concatenate([subjects_train, subjects_val, subjects_test])

    print(f"Loaded (optimized):")
    print(f"   Train: {X_train.shape} ({X_train.nbytes/(1024**2):.1f} MB)")
    print(f"   Val:   {X_val.shape} ({X_val.nbytes/(1024**2):.1f} MB)")
    print(f"   Test:  {X_test.shape} ({X_test.nbytes/(1024**2):.1f} MB)\n")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, \
        subjects


def load_windowed_sleep_edf(data_dir: str) -> Tuple:
    """Load preprocessed Sleep-EDF windowed data."""
    hdf5_path = os.path.join(data_dir, 'sleep_edf_data.h5')

    if os.path.exists(hdf5_path):
        return load_windowed_sleep_edf_optimized(data_dir)

    print(f"Loading Sleep-EDF from legacy format...")

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
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

    info_path = os.path.join(data_dir, 'preprocessing_info.pkl')
    if os.path.exists(info_path):
        info = joblib.load(info_path)
    else:
        info = {
            'window_epochs': 10,
            'n_channels': 3,
            'samples_per_epoch': 3000,
            'n_classes': 5,
            'class_names': ['W', 'N1', 'N2', 'N3', 'R']
        }

    print(f"Loaded (legacy):")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}\n")

    subjects = np.concatenate([subjects_train, subjects_val, subjects_test])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, \
        subjects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess Sleep-EDF dataset (HDF5 optimized)'
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