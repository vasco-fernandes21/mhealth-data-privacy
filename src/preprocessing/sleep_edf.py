"""
Sleep-EDF Dataset Preprocessing Module (FINAL VERSION)

Features:
- Subject-wise splitting (critical for FL)
- Subject ID tracking for each epoch
- Automatic windowed data creation
- Parallel processing for speed
- Cache system to avoid reprocessing

Usage:
    python -m src.preprocessing.sleep_edf --data_dir data/raw/sleep-edf --output_dir data/processed/sleep-edf
"""

import numpy as np
import pandas as pd
import pyedflib
import mne
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import glob
import time
import pickle
from typing import Tuple, Dict, List
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')


# ============================================================================
# LOADING FUNCTIONS (unchanged from original)
# ============================================================================

def load_sleep_edf_expanded_hypnogram(hypno_file: str, target_epoch_duration: int = 30) -> Tuple[List[str], List[int], int, int]:
    """Carrega hypnogram do Sleep-EDF Expanded"""
    if not os.path.exists(hypno_file):
        print(f'Ficheiro não encontrado: {hypno_file}')
        return None, None, None, None
    
    try:
        f = pyedflib.EdfReader(hypno_file)
        annotations = f.read_annotation()
        f.close()
        
        sleep_stages = []
        epoch_durations = []
        
        for onset, duration, description in annotations:
            desc_str = description.decode('utf-8') if isinstance(description, bytes) else str(description)
            
            if 'Sleep stage' in desc_str:
                if 'Sleep stage W' in desc_str:
                    stage = 'W'
                elif 'Sleep stage 1' in desc_str:
                    stage = '1'
                elif 'Sleep stage 2' in desc_str:
                    stage = '2'
                elif 'Sleep stage 3' in desc_str:
                    stage = '3'
                elif 'Sleep stage 4' in desc_str:
                    stage = '4'
                elif 'Sleep stage R' in desc_str:
                    stage = 'R'
                elif 'Sleep stage M' in desc_str:
                    stage = 'M'
                else:
                    stage = '?'
                
                if isinstance(duration, bytes):
                    try:
                        duration_sec = int(duration.decode('utf-8'))
                    except:
                        duration_sec = 30
                else:
                    duration_sec = int(duration)
                
                sleep_stages.append(stage)
                epoch_durations.append(duration_sec)
        
        total_duration = sum(epoch_durations)
        n_epochs = int(total_duration / target_epoch_duration)
        
        return sleep_stages, epoch_durations, total_duration, n_epochs
        
    except Exception as e:
        print(f'Erro ao ler hypnogram: {e}')
        return None, None, None, None


def convert_hypnogram_to_30s_epochs(sleep_stages: List[str], epoch_durations: List[int], target_epoch_duration: int = 30) -> List[str]:
    """Converte hypnogram com durações variáveis para épocas de 30s"""
    target_epochs = []
    
    for stage, duration in zip(sleep_stages, epoch_durations):
        n_epochs = int(duration / target_epoch_duration)
        
        for _ in range(n_epochs):
            target_epochs.append(stage)
        
        remaining_time = duration % target_epoch_duration
        if remaining_time >= target_epoch_duration / 2:
            target_epochs.append(stage)
    
    return target_epochs


def load_sleep_edf_expanded_file(psg_path: str, hypno_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load a single Sleep-EDF Expanded file"""
    print(f'Loading: {os.path.basename(psg_path)}')
    
    try:
        f = pyedflib.EdfReader(psg_path)
        
        info = {
            'n_channels': f.signals_in_file,
            'duration': f.file_duration,
            'start_time': f.getStartdatetime(),
            'sample_rates': [f.getSampleFrequency(i) for i in range(f.signals_in_file)],
            'channel_labels': [f.getLabel(i) for i in range(f.signals_in_file)]
        }
        
        eeg_fpz_cz = f.readSignal(0)
        eeg_pz_oz = f.readSignal(1)
        eog = f.readSignal(2)
        
        f.close()
        
        signals = np.array([eeg_fpz_cz, eeg_pz_oz, eog])
        
        print(f'   • Signals: {signals.shape}')
        print(f'   • Channels: {info["channel_labels"][:3]}')
        
    except Exception as e:
        print(f'Error loading PSG: {e}')
        return None, None, None
    
    sleep_stages, epoch_durations, total_duration, n_epochs = load_sleep_edf_expanded_hypnogram(hypno_path)
    
    if sleep_stages is None:
        print(f'Error loading hypnogram')
        return None, None, None
    
    labels_30s = convert_hypnogram_to_30s_epochs(sleep_stages, epoch_durations)
    
    label_mapping = {'W': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'R': 5, 'M': 6, '?': 7}
    labels = np.array([label_mapping.get(stage, 7) for stage in labels_30s])
    
    print(f'   • Labels: {len(labels)} epochs')
    
    return signals, labels, info


def load_sleep_edf_file(edf_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load wrapper"""
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"PSG file not found: {edf_path}")
    if not os.path.exists(hyp_path):
        raise FileNotFoundError(f"Hypnogram file not found: {hyp_path}")
    
    return load_sleep_edf_expanded_file(edf_path, hyp_path)


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def filter_signals(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """Apply Butterworth bandpass filters"""
    if signals.shape[1] < 100:
        return signals.copy()

    filtered_signals = np.zeros_like(signals)

    try:
        b_eeg, a_eeg = signal.butter(3, [0.5, 32], btype='band', fs=sfreq)
        b_eog, a_eog = signal.butter(3, [0.5, 10], btype='band', fs=sfreq)
    except ValueError:
        return signals.copy()

    for i in range(signals.shape[0]):
        try:
            if i < 2:  # EEG channels
                filtered_signals[i] = signal.filtfilt(b_eeg, a_eeg, signals[i])
            else:  # EOG channel
                filtered_signals[i] = signal.filtfilt(b_eog, a_eog, signals[i])
        except (ValueError, RuntimeError):
            filtered_signals[i] = signals[i]

    return filtered_signals


def segment_epochs(signals: np.ndarray, labels: np.ndarray, sfreq: float, 
                   epoch_duration: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Segment signals into epochs"""
    n_samples_epoch = int(sfreq * epoch_duration)
    n_epochs = signals.shape[1] // n_samples_epoch
    
    n_available_labels = len(labels)
    n_epochs_to_use = min(n_epochs, n_available_labels)
    
    if n_epochs_to_use == 0:
        return np.empty((signals.shape[0], 0, n_samples_epoch)), np.array([])
    
    epochs = signals[:, :n_epochs_to_use * n_samples_epoch].reshape(signals.shape[0], n_epochs_to_use, n_samples_epoch)
    epoch_labels = labels[:n_epochs_to_use]
    
    return epochs, epoch_labels


def extract_sleep_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract features from a single epoch"""
    n_channels = epoch.shape[0]
    features = np.zeros(n_channels * 8)

    for i, channel in enumerate(epoch):
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        max_val = np.max(channel)
        min_val = np.min(channel)

        nperseg = min(256, len(channel))
        freqs, psd = signal.welch(channel, sfreq, nperseg=nperseg)

        band_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
        band_powers = []

        for low, high in band_ranges:
            mask = (freqs >= low) & (freqs < high)
            band_powers.append(np.mean(psd[mask]) if np.any(mask) else 0.0)

        start_idx = i * 8
        features[start_idx:start_idx + 8] = [
            mean_val, std_val, max_val, min_val,
            band_powers[0], band_powers[1], band_powers[2], band_powers[3]
        ]

    return features


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_file(args: Tuple) -> Tuple[List, List, str]:
    """Process single file for parallelization"""
    psg_file, hypno_file, file_idx, total_files = args

    try:
        print(f"[{file_idx}/{total_files}] Processing {os.path.basename(psg_file)}...")

        signals, labels, info = load_sleep_edf_file(psg_file, hypno_file)

        if signals is None or labels is None:
            print(f"[{file_idx}/{total_files}] Error: Failed to load")
            return [], [], None

        filtered_signals = filter_signals(signals, 100.0)
        epochs, epoch_labels = segment_epochs(filtered_signals, labels, 100.0)

        if epochs.shape[1] == 0:
            print(f"[{file_idx}/{total_files}] Warning: No epochs created")
            return [], [], None

        batch_size = min(100, epochs.shape[1])
        features_list = []
        labels_list = []

        for batch_start in range(0, epochs.shape[1], batch_size):
            batch_end = min(batch_start + batch_size, epochs.shape[1])

            for i in range(batch_start, batch_end):
                features = extract_sleep_features(epochs[:, i], 100.0)
                features_list.append(features)
                labels_list.append(epoch_labels[i])

        # Extract subject ID
        psg_basename = os.path.basename(psg_file)
        psg_prefix = psg_basename.replace('-PSG.edf', '')
        if psg_prefix.startswith('SC'):
            subject_id = psg_prefix[:-1][:6]
        elif psg_prefix.startswith('ST'):
            subject_id = psg_prefix[:-1][:6]
        else:
            subject_id = psg_prefix[:6] if len(psg_prefix) >= 6 else psg_prefix

        print(f"[{file_idx}/{total_files}] Completed: {len(features_list)} epochs")

        return features_list, labels_list, subject_id

    except Exception as e:
        print(f"[{file_idx}/{total_files}] Error: {e}")
        return [], [], None


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_sleep_edf(data_dir: str, output_dir: str,
                        test_size: float = 0.15, val_size: float = 0.15,
                        random_state: int = 42, n_workers: int = None, 
                        force_reprocess: bool = False,
                        create_windows: bool = True,  # ✅ NOVO
                        window_size: int = 10) -> Dict:  # ✅ NOVO
    """
    Complete preprocessing pipeline with automatic windowing

    Args:
        data_dir: Directory containing Sleep-EDF Expanded files
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        n_workers: Number of parallel workers
        force_reprocess: Force reprocessing
        create_windows: Automatically create windowed data (default=True)
        window_size: Window size for LSTM (default=10)

    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("SLEEP-EDF PREPROCESSING (FINAL VERSION)")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Find all PSG and Hypnogram files
    psg_files = glob.glob(os.path.join(data_dir, '**/*-PSG.edf'), recursive=True)
    hypno_files = glob.glob(os.path.join(data_dir, '**/*-Hypnogram.edf'), recursive=True)

    print(f"Found {len(psg_files)} PSG files and {len(hypno_files)} Hypnogram files")

    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} parallel workers")

    # Prepare file pairs
    file_pairs = []
    for file_idx, psg_file in enumerate(psg_files, 1):
        psg_basename = os.path.basename(psg_file)
        psg_prefix = psg_basename.replace('-PSG.edf', '')

        if psg_prefix.startswith('SC'):
            base_prefix = psg_prefix[:-1]
        elif psg_prefix.startswith('ST'):
            base_prefix = psg_prefix[:-1]
        else:
            base_prefix = psg_prefix

        subject_id = base_prefix[:6] if len(base_prefix) >= 6 else base_prefix

        hypno_file = None
        for hypno_path in hypno_files:
            hypno_basename = os.path.basename(hypno_path)
            hypno_prefix = hypno_basename.replace('-Hypnogram.edf', '')

            if hypno_prefix.startswith('SC'):
                hypno_base_prefix = hypno_prefix[:-1]
            elif hypno_prefix.startswith('ST'):
                hypno_base_prefix = hypno_prefix[:-1]
            else:
                hypno_base_prefix = hypno_prefix

            if base_prefix == hypno_base_prefix:
                hypno_file = hypno_path
                break

        if hypno_file is None:
            print(f"[{file_idx}/{len(psg_files)}] Warning: No hypnogram for {psg_basename}")
            continue

        file_pairs.append((psg_file, hypno_file, file_idx, len(psg_files)))

    print(f"Processing {len(file_pairs)} file pairs in parallel...")

    start_time = time.time()

    # Process files in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_file, file_pairs)

    # Combine results
    all_features = []
    all_labels = []
    all_subjects = []

    for features_list, labels_list, subject_id in results:
        if features_list and labels_list and subject_id:
            all_features.extend(features_list)
            all_labels.extend(labels_list)
            all_subjects.extend([subject_id] * len(features_list))

    print(f"\nParallel processing completed in {time.time() - start_time:.1f}s")
    print(f"Total epochs processed: {len(all_features)}")

    if len(all_features) == 0:
        print("No data processed - check file formats and paths")
        return None

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    subjects_arr = np.array(all_subjects)

    print(f"Feature shape: {X.shape}")

    if len(y) > 0:
        print(f"Label distribution: {np.bincount(y.astype(int))}")
    else:
        print("No data processed")
        return None

    # Map labels to standard 5-class format
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 0, 7: 0}
    y_encoded = np.array([label_mapping.get(label, 0) for label in y])

    # ✅ SUBJECT-WISE SPLIT
    print(f"\nPerforming subject-wise split...")
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

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_val, y_val = X[val_mask], y_encoded[val_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]
    
    # ✅ EXTRACT SUBJECT IDs per split
    subjects_train = subjects_arr[train_mask]
    subjects_val = subjects_arr[val_mask]
    subjects_test = subjects_arr[test_mask]

    print(f"Subject splits:")
    print(f"  Train: {len(subj_train)} subjects → {len(X_train)} epochs")
    print(f"  Val: {len(subj_val)} subjects → {len(X_val)} epochs")
    print(f"  Test: {len(subj_test)} subjects → {len(X_test)} epochs")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ✅ SAVE BASIC DATA + SUBJECT IDs
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # ✅ SAVE SUBJECT IDs
    np.save(os.path.join(output_dir, 'subjects_train.npy'), subjects_train)
    np.save(os.path.join(output_dir, 'subjects_val.npy'), subjects_val)
    np.save(os.path.join(output_dir, 'subjects_test.npy'), subjects_test)

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    preprocessing_info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'class_names': ['W', 'N1', 'N2', 'N3', 'R'],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'files_processed': len(file_pairs),
        'subject_splits': {
            'train_subjects': [str(s) for s in subj_train],
            'val_subjects': [str(s) for s in subj_val],
            'test_subjects': [str(s) for s in subj_test],
            'total_subjects': len(unique_subjects)
        },
        'split_type': 'subject-wise',
        'parallel_processing': True,
        'n_workers': n_workers,
        'has_subject_ids': True  # ✅ Flag
    }

    joblib.dump(preprocessing_info, os.path.join(output_dir, 'preprocessing_info.pkl'))

    print(f"\n✅ Basic preprocessing complete!")
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # ✅ AUTO-CREATE WINDOWED DATA
    if create_windows:
        print(f"\n📊 Creating windowed data (window_size={window_size})...")
        try:
            _create_windowed_data_internal(
                X_train_scaled, y_train, subjects_train,
                X_val_scaled, y_val, subjects_val,
                X_test_scaled, y_test, subjects_test,
                output_dir, window_size
            )
            print("✅ Windowed data created!")
            preprocessing_info['has_windowed_data'] = True
            preprocessing_info['window_size'] = window_size
            joblib.dump(preprocessing_info, os.path.join(output_dir, 'preprocessing_info.pkl'))
        except Exception as e:
            print(f"⚠️  Windowed data creation failed: {e}")
            preprocessing_info['has_windowed_data'] = False
    
    print(f"\nData saved to: {output_dir}")

    return preprocessing_info


# ============================================================================
# WINDOWING FUNCTIONS
# ============================================================================

def _create_windowed_data_internal(X_train, y_train, subjects_train,
                                   X_val, y_val, subjects_val,
                                   X_test, y_test, subjects_test,
                                   output_dir, window_size):
    """Internal function called by preprocess_sleep_edf"""
    from numpy.lib.stride_tricks import sliding_window_view
    
    def create_windows(X, y, subjects, window_size):
        n_samples, n_features = X.shape
        n_windows = n_samples - window_size + 1

        X_windows = sliding_window_view(X, window_size, axis=0)
        X_windows = X_windows.transpose(0, 2, 1)
        y_windows = y[window_size-1:]
        subjects_windows = subjects[window_size-1:]

        return X_windows, y_windows, subjects_windows

    print("  Processing train data...")
    X_train_windows, y_train_windows, subjects_train_windows = create_windows(
        X_train, y_train, subjects_train, window_size
    )
    print(f"    Train windows: {X_train_windows.shape}")

    print("  Processing validation data...")
    X_val_windows, y_val_windows, subjects_val_windows = create_windows(
        X_val, y_val, subjects_val, window_size
    )
    print(f"    Val windows: {X_val_windows.shape}")

    print("  Processing test data...")
    X_test_windows, y_test_windows, subjects_test_windows = create_windows(
        X_test, y_test, subjects_test, window_size
    )
    print(f"    Test windows: {X_test_windows.shape}")

    # Save
    np.save(os.path.join(output_dir, 'X_train_windows.npy'), X_train_windows)
    np.save(os.path.join(output_dir, 'y_train_windows.npy'), y_train_windows)
    np.save(os.path.join(output_dir, 'X_val_windows.npy'), X_val_windows)
    np.save(os.path.join(output_dir, 'y_val_windows.npy'), y_val_windows)
    np.save(os.path.join(output_dir, 'X_test_windows.npy'), X_test_windows)
    np.save(os.path.join(output_dir, 'y_test_windows.npy'), y_test_windows)
    
    np.save(os.path.join(output_dir, 'subjects_train_windows.npy'), subjects_train_windows)
    np.save(os.path.join(output_dir, 'subjects_val_windows.npy'), subjects_val_windows)
    np.save(os.path.join(output_dir, 'subjects_test_windows.npy'), subjects_test_windows)


def create_windowed_data(data_dir: str, window_size: int = 10) -> None:
    """
    Standalone function to create windowed data from existing preprocessed data
    
    Use this if you want to create windows with different size later
    """
    print(f"Creating windowed data for Sleep-EDF (window_size={window_size})...")

    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    subjects_train = np.load(os.path.join(data_dir, 'subjects_train.npy'))
    subjects_val = np.load(os.path.join(data_dir, 'subjects_val.npy'))
    subjects_test = np.load(os.path.join(data_dir, 'subjects_test.npy'))

    _create_windowed_data_internal(
        X_train, y_train, subjects_train,
        X_val, y_val, subjects_val,
        X_test, y_test, subjects_test,
        data_dir, window_size
    )
    
    # Update preprocessing info
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))
    info['has_windowed_data'] = True
    info['window_size'] = window_size
    joblib.dump(info, os.path.join(data_dir, 'preprocessing_info.pkl'))
    
    print(f"✅ Windowed data saved! Window size: {window_size}")


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_processed_sleep_edf(data_dir: str) -> Tuple:
    """Load preprocessed Sleep-EDF data (basic format)"""
    print(f"Loading processed Sleep-EDF data from {data_dir}...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))
    
    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info


def load_windowed_sleep_edf(data_dir: str) -> Tuple:
    """
    Load preprocessed windowed Sleep-EDF data WITH subject IDs
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects_train)
    """
    print(f"Loading windowed Sleep-EDF data from {data_dir}...")

    X_train = np.load(os.path.join(data_dir, 'X_train_windows.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_windows.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val_windows.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val_windows.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_windows.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_windows.npy'))
    
    # ✅ Load subject IDs if available
    subjects_train = None
    try:
        subjects_train = np.load(os.path.join(data_dir, 'subjects_train_windows.npy'))
        print("✅ Subject IDs loaded - FL subject-aware partitioning available!")
    except FileNotFoundError:
        print("⚠️  Subject IDs not found - FL will use random partitioning")

    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))

    print(f"Loaded windowed data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects_train


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Sleep-EDF dataset')
    parser.add_argument('--data_dir', default='../../data/raw/sleep-edf', help='Raw data directory')
    parser.add_argument('--output_dir', default='../../data/processed/sleep-edf', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--no_windows', action='store_true', help='Skip automatic windowing')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for LSTM')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SLEEP-EDF PREPROCESSING")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.data_dir):
        print(f"Raw data not found: {args.data_dir}")
        exit(1)
    
    print(f"Raw data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print(f"Create windows: {not args.no_windows}")
    if not args.no_windows:
        print(f"Window size: {args.window_size}")
    
    # Run preprocessing
    info = preprocess_sleep_edf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        create_windows=not args.no_windows,
        window_size=args.window_size
    )
    
    print(f"\n✅ Sleep-EDF preprocessing completed!")
    print(f"Preprocessing info: {info}")