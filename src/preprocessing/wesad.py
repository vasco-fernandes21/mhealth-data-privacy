"""
WESAD Dataset Preprocessing Module

This module handles the preprocessing of WESAD dataset:
- Loading pickle files with physiological signals from RespiBAN and Empatica E4
- Signal filtering and resampling
- Comprehensive feature extraction (time, frequency, and statistical domain)
- Sliding window approach for data augmentation
- Normalization and train/val/test splitting

WESAD contains:
- 15 subjects with ~100 minutes of data each
- RespiBAN (chest): ECG, EDA, EMG, Temp, Resp, ACC (3D) at 700 Hz
- Empatica E4 (wrist): ACC (3D), BVP, EDA, TEMP at different frequencies
- 8 classes: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=other
"""

import numpy as np
import pandas as pd
import pickle
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import time
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_wesad_file(pkl_path: str) -> Dict:
    """
    Load a single WESAD pickle file with all available signals.
    
    Args:
        pkl_path: Path to .pkl file
    
    Returns:
        Dictionary with signals and labels from both RespiBAN and Empatica E4
    """
    print(f"Loading {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract signals from RespiBAN (chest device) - synchronized with labels
    chest_signals = data['signal']['chest']
    ecg = chest_signals['ECG']  # Electrocardiogram (700 Hz)
    eda_chest = chest_signals['EDA']  # Electrodermal Activity (700 Hz)
    temp_chest = chest_signals['Temp']  # Temperature (700 Hz)
    acc_chest = chest_signals['ACC']  # Acceleration 3D (700 Hz)
    emg = chest_signals['EMG']  # Electromyography (700 Hz)
    resp = chest_signals['Resp']  # Respiration (700 Hz)
    
    # Extract signals from Empatica E4 (wrist device) - different frequencies
    wrist_signals = data['signal']['wrist']
    acc_wrist = wrist_signals['ACC']  # Acceleration 3D (32 Hz)
    bvp = wrist_signals['BVP']  # Blood Volume Pulse (64 Hz)
    eda_wrist = wrist_signals['EDA']  # Electrodermal Activity (4 Hz)
    temp_wrist = wrist_signals['TEMP']  # Temperature (4 Hz)
    
    # Extract labels (synchronized with chest signals)
    labels = data['label']  # Labels: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=other
    
    return {
        # Chest signals (RespiBAN) - 700 Hz
        'ecg': ecg,
        'eda_chest': eda_chest,
        'temp_chest': temp_chest,
        'acc_chest': acc_chest,
        'emg': emg,
        'resp': resp,
        
        # Wrist signals (Empatica E4) - different frequencies
        'acc_wrist': acc_wrist,
        'bvp': bvp,
        'eda_wrist': eda_wrist,
        'temp_wrist': temp_wrist,
        
        'labels': labels,
        'subject': data.get('subject', 'unknown'),
        'sfreq': {
            'chest': 700,  # All chest signals at 700 Hz
            'acc_wrist': 32,
            'bvp': 64,
            'eda_wrist': 4,
            'temp_wrist': 4
        }
    }


def resample_signals(signals_dict: Dict, target_freq: float = 4) -> Dict:
    """
    Resample all signals to the same frequency.
    
    Args:
        signals_dict: Dictionary with signals and sampling frequencies
        target_freq: Target sampling frequency (Hz)
    
    Returns:
        Dictionary with resampled signals
    """
    print(f"Resampling signals to {target_freq} Hz...")
    
    resampled = {}
    sfreq = signals_dict['sfreq']
    
    # Use chest signals as reference (700 Hz) for labels
    original_length = len(signals_dict['ecg'])
    target_length = int(original_length * target_freq / sfreq['chest'])
    
    # Resample chest signals from 700 Hz to target frequency
    resampled['ecg'] = signal.resample(signals_dict['ecg'], target_length)
    resampled['eda_chest'] = signal.resample(signals_dict['eda_chest'], target_length)
    resampled['temp_chest'] = signal.resample(signals_dict['temp_chest'], target_length)
    resampled['acc_chest'] = signal.resample(signals_dict['acc_chest'], target_length)
    resampled['emg'] = signal.resample(signals_dict['emg'], target_length)
    resampled['resp'] = signal.resample(signals_dict['resp'], target_length)
    
    # Resample wrist signals to target frequency
    resampled['acc_wrist'] = signal.resample(signals_dict['acc_wrist'], 
                                           int(len(signals_dict['acc_wrist']) * target_freq / sfreq['acc_wrist']))
    resampled['bvp'] = signal.resample(signals_dict['bvp'], 
                                     int(len(signals_dict['bvp']) * target_freq / sfreq['bvp']))
    resampled['eda_wrist'] = signal.resample(signals_dict['eda_wrist'], 
                                           int(len(signals_dict['eda_wrist']) * target_freq / sfreq['eda_wrist']))
    resampled['temp_wrist'] = signal.resample(signals_dict['temp_wrist'], 
                                            int(len(signals_dict['temp_wrist']) * target_freq / sfreq['temp_wrist']))
    
    # Use stratified sampling for labels to maintain class proportions
    indices = np.linspace(0, original_length-1, target_length, dtype=int)
    resampled['labels'] = signals_dict['labels'][indices]
    
    resampled['subject'] = signals_dict['subject']
    resampled['sfreq'] = target_freq
    
    return resampled


def extract_comprehensive_features(signal_data: np.ndarray, signal_name: str) -> np.ndarray:
    """
    Extract comprehensive features from a signal.
    
    Args:
        signal_data: Signal data (can be 1D or 2D)
        signal_name: Name of the signal for logging
    
    Returns:
        Array of extracted features
    """
    features = []
    
    # Handle multi-dimensional signals (like ACC)
    if signal_data.ndim > 1:
        for i in range(signal_data.shape[1]):
            channel_features = _extract_single_channel_features(signal_data[:, i], f"{signal_name}_ch{i}")
            features.extend(channel_features)
    else:
        features = _extract_single_channel_features(signal_data, signal_name)
    
    return np.array(features)


def _extract_single_channel_features(signal_data: np.ndarray, signal_name: str) -> List[float]:
    """
    Extract features from a single channel signal.
    
    Args:
        signal_data: 1D signal data
        signal_name: Name of the signal
    
    Returns:
        List of extracted features
    """
    features = []
    
    # Statistical features
    features.extend([
        np.mean(signal_data),           # Mean
        np.std(signal_data),            # Standard deviation
        np.var(signal_data),            # Variance
        np.median(signal_data),         # Median
        np.percentile(signal_data, 25), # 25th percentile
        np.percentile(signal_data, 75), # 75th percentile
        skew(signal_data),              # Skewness
        kurtosis(signal_data),          # Kurtosis
        np.min(signal_data),            # Minimum
        np.max(signal_data),            # Maximum
        np.max(signal_data) - np.min(signal_data),  # Range
    ])
    
    # Time domain features
    features.extend([
        np.sum(np.abs(np.diff(signal_data))),  # Total variation
        np.mean(np.abs(np.diff(signal_data))), # Mean absolute difference
        np.std(np.diff(signal_data)),          # Std of differences
    ])
    
    # Frequency domain features
    try:
        # Power spectral density
        freqs, psd = signal.welch(signal_data, nperseg=min(256, len(signal_data)//4))
        
        # Spectral features
        features.extend([
            np.sum(psd),                    # Total power
            np.mean(psd),                   # Mean power
            np.std(psd),                    # Power std
            freqs[np.argmax(psd)],          # Dominant frequency
            np.max(psd),                    # Peak power
        ])
        
        # Frequency band powers (if signal is long enough)
        if len(freqs) > 10:
            # Low frequency (0-0.1 Hz)
            lf_mask = freqs <= 0.1
            lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
            
            # High frequency (0.1-0.5 Hz)
            hf_mask = (freqs > 0.1) & (freqs <= 0.5)
            hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
            
            features.extend([lf_power, hf_power])
            
            # LF/HF ratio
            if hf_power > 0:
                features.append(lf_power / hf_power)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])  # Placeholder for short signals
            
    except Exception as e:
        # If spectral analysis fails, add zeros
        features.extend([0] * 8)
    
    return features


def create_sliding_windows(signals_dict: Dict, window_size: int = 60, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from signals for data augmentation.
    
    Args:
        signals_dict: Dictionary with resampled signals
        window_size: Window size in samples (default: 60 samples = 15 seconds at 4 Hz)
        overlap: Overlap ratio between windows (0.0 to 0.9)
    
    Returns:
        Tuple of (features_array, labels_array)
    """
    print(f"Creating sliding windows (size={window_size}, overlap={overlap})...")
    
    # Get reference length from chest signals
    ref_length = len(signals_dict['ecg'])
    step_size = int(window_size * (1 - overlap))
    
    features_list = []
    labels_list = []
    
    # Create windows
    for start_idx in range(0, ref_length - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract window data
        window_data = {}
        for signal_name in ['ecg', 'eda_chest', 'temp_chest', 'acc_chest', 'emg', 'resp', 
                           'acc_wrist', 'bvp', 'eda_wrist', 'temp_wrist']:
            if signal_name in signals_dict:
                signal_data = signals_dict[signal_name]
                # Handle different signal lengths
                if len(signal_data) >= end_idx:
                    window_data[signal_name] = signal_data[start_idx:end_idx]
                else:
                    # Pad with zeros if signal is shorter
                    padded_signal = np.zeros(window_size)
                    available_length = min(len(signal_data) - start_idx, window_size)
                    if available_length > 0:
                        padded_signal[:available_length] = signal_data[start_idx:start_idx + available_length]
                    window_data[signal_name] = padded_signal
        
        # Get label for this window (use majority vote)
        window_labels = signals_dict['labels'][start_idx:end_idx]
        # Remove undefined labels (0) for voting
        valid_labels = window_labels[window_labels != 0]
        if len(valid_labels) > 0:
            window_label = np.bincount(valid_labels).argmax()
        else:
            window_label = 0  # Default to undefined if no valid labels
        
        # Extract features from this window
        window_features = []
        for signal_name, signal_data in window_data.items():
            features = extract_comprehensive_features(signal_data, signal_name)
            window_features.extend(features)
        
        features_list.append(window_features)
        labels_list.append(window_label)
    
    return np.array(features_list), np.array(labels_list)


def filter_signals(signals_dict: Dict) -> Dict:
    """
    Apply bandpass filters to signals.
    
    Args:
        signals_dict: Dictionary with resampled signals
    
    Returns:
        Dictionary with filtered signals
    """
    print("Applying bandpass filters...")
    
    filtered = {}
    sfreq = signals_dict['sfreq']
    
    # ECG: 0.5-1.5 Hz (heart rate range adapted for 4Hz sampling)
    filtered['ecg'] = signal.sosfilt(signal.butter(4, [0.5, 1.5], btype='band', fs=sfreq, output='sos'), 
                                    signals_dict['ecg'])
    
    # EDA: 0.05-1 Hz (slow variations)
    filtered['eda'] = signal.sosfilt(signal.butter(4, [0.05, 1], btype='band', fs=sfreq, output='sos'), 
                                    signals_dict['eda'])
    
    # Temperature: no filtering (very slow variations)
    filtered['temp'] = signals_dict['temp']
    
    # Acceleration: 0.1-1.5 Hz (body movement adapted for 4Hz sampling)
    filtered['acc'] = signal.sosfilt(signal.butter(4, [0.1, 1.5], btype='band', fs=sfreq, output='sos'), 
                                    signals_dict['acc'])
    
    filtered['labels'] = signals_dict['labels']
    filtered['sfreq'] = sfreq
    
    return filtered


def create_windows(signals_dict: Dict, window_size: int = 240, stride: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from signals.
    
    Args:
        signals_dict: Dictionary with filtered signals
        window_size: Window size in samples (60 seconds at 4 Hz)
        stride: Stride between windows (30 seconds at 4 Hz)
    
    Returns:
        Tuple of (windows, window_labels)
    """
    print(f"Creating windows (size={window_size}, stride={stride})...")
    
    ecg = signals_dict['ecg']
    eda = signals_dict['eda']
    temp = signals_dict['temp']
    acc = signals_dict['acc']
    labels = signals_dict['labels']
    
    windows = []
    window_labels = []
    
    for i in range(0, len(ecg) - window_size, stride):
        # Extract window
        ecg_window = ecg[i:i+window_size, 0]  # Take first channel
        eda_window = eda[i:i+window_size, 0]  # Take first channel
        temp_window = temp[i:i+window_size, 0]  # Take first channel
        acc_window = acc[i:i+window_size, 0]  # Take first channel (x-axis)
        
        # Get majority label for window
        window_label = np.bincount(labels[i:i+window_size].astype(int)).argmax()
        
        # Store window data
        window_data = np.array([ecg_window, eda_window, temp_window, acc_window])
        windows.append(window_data)
        window_labels.append(window_label)
    
    return np.array(windows), np.array(window_labels)


def extract_wesad_features(window: np.ndarray, sfreq: float = 4) -> np.ndarray:
    """
    Extract features from a single window.
    
    Args:
        window: Array of shape (n_channels, n_samples)
        sfreq: Sampling frequency
    
    Returns:
        Feature vector of length 27 (9 features per channel)
    """
    features = []
    
    for channel in window:
        # Time domain features
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        # Frequency domain features (Power Spectral Density)
        freqs, psd = signal.welch(channel, sfreq, nperseg=min(64, len(channel)))
        
        # Use first 5 PSD values as features
        psd_features = psd[:5]
        
        # Combine features for this channel
        channel_features = [mean_val, std_val, min_val, max_val] + list(psd_features)
        features.extend(channel_features)
    
    return np.array(features)


def preprocess_wesad(data_dir: str, output_dir: str,
                    test_size: float = 0.15, val_size: float = 0.15,
                    random_state: int = 42) -> Dict:
    """
    Complete preprocessing pipeline for WESAD dataset.
    
    Args:
        data_dir: Directory containing WESAD pickle files
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("WESAD PREPROCESSING")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files in subdirectories
    import glob
    pkl_files = glob.glob(os.path.join(data_dir, '**/*.pkl'), recursive=True)
    print(f"Found {len(pkl_files)} pickle files")
    
    all_features = []
    all_labels = []
    
    # Process each file
    total_files = len(pkl_files)
    print(f"Processing {total_files} WESAD files...")
    
    start_time = time.time()
    
    for file_idx, pkl_file in enumerate(pkl_files, 1):
        pkl_path = os.path.join(data_dir, pkl_file)
        
        try:
            # Load file
            data = load_wesad_file(pkl_path)
            
            # Resample signals
            resampled = resample_signals(data)
            
            # Filter signals
            filtered = filter_signals(resampled)
            
            # Create windows
            windows, window_labels = create_windows(filtered)
            
            # Extract features for each window
            for i in range(len(windows)):
                features = extract_wesad_features(windows[i])
                all_features.append(features)
                all_labels.append(window_labels[i])
            
            # Calculate progress and time estimates
            elapsed_time = time.time() - start_time
            progress_pct = (file_idx / total_files) * 100
            
            if file_idx > 1:
                avg_time_per_file = elapsed_time / file_idx
                remaining_files = total_files - file_idx
                estimated_remaining_time = avg_time_per_file * remaining_files
                remaining_minutes = int(estimated_remaining_time // 60)
                remaining_seconds = int(estimated_remaining_time % 60)
                
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {pkl_file}: {len(windows)} windows | ETA: {remaining_minutes}m {remaining_seconds}s")
            else:
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {pkl_file}: {len(windows)} windows")
            
        except Exception as e:
            print(f"[{file_idx}/{total_files}] Error processing {pkl_file}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nTotal windows processed: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    if len(y) > 0:
        print(f"Label distribution: {np.bincount(y.astype(int))}")
    else:
        print("No data processed - check file formats and paths")
        return None
    
    # Keep only the 3 main classes: baseline (1), stress (2), amusement (3)
    # Remove: not defined/transient (0), meditation (4) and other states (5,6,7)
    valid_mask = (y >= 1) & (y <= 3)
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    
    # Relabel: baseline=0, stress=1, amusement=2
    y_filtered = y_filtered - 1
    
    print(f"After filtering: {len(X_filtered)} windows")
    print(f"New label distribution: {np.bincount(y_filtered)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_filtered)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_filtered, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    
    # Save preprocessing info
    preprocessing_info = {
        'n_samples': len(X_filtered),
        'n_features': X_filtered.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'class_names': ['baseline', 'stress', 'amusement'][:len(np.unique(y_encoded))],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'files_processed': len(pkl_files),
        'original_labels': '0=baseline, 1=stress, 2=amusement, 3=meditation',
        'filtered_labels': '0=baseline, 1=stress, 2=amusement'
    }
    
    joblib.dump(preprocessing_info, os.path.join(output_dir, 'preprocessing_info.pkl'))
    
    print(f"\nPreprocessing complete!")
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Classes: {preprocessing_info['n_classes']} ({preprocessing_info['class_names']})")
    print(f"Data saved to: {output_dir}")
    
    return preprocessing_info


def load_processed_wesad(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray, np.ndarray, 
                                                 StandardScaler, LabelEncoder, Dict]:
    """
    Load preprocessed WESAD data.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, info)
    """
    print(f"Loading processed WESAD data from {data_dir}...")
    
    # Load arrays
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load scaler and label encoder
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))
    
    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, info


def preprocess_wesad_improved(data_dir: str, output_dir: str, target_freq: float = 4,
                             window_size: int = 60, overlap: float = 0.5,
                             test_size: float = 0.15, val_size: float = 0.15,
                             random_state: int = 42) -> Dict:
    """
    Improved preprocessing pipeline for WESAD dataset with comprehensive feature extraction.
    
    Args:
        data_dir: Directory containing WESAD pickle files
        output_dir: Directory to save processed data
        target_freq: Target sampling frequency (Hz)
        window_size: Window size in samples (default: 60 = 15 seconds at 4 Hz)
        overlap: Overlap ratio between windows
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("WESAD IMPROVED PREPROCESSING")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files in subdirectories
    import glob
    pkl_files = glob.glob(os.path.join(data_dir, '**/*.pkl'), recursive=True)
    print(f"Found {len(pkl_files)} pickle files")
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    # Process each file
    total_files = len(pkl_files)
    print(f"Processing {total_files} WESAD files...")
    
    start_time = time.time()
    
    for file_idx, pkl_file in enumerate(pkl_files, 1):
        pkl_path = os.path.join(data_dir, pkl_file)
        
        try:
            # Load file
            data = load_wesad_file(pkl_path)
            subject = data['subject']
            
            # Resample signals
            resampled = resample_signals(data, target_freq)
            
            # Create sliding windows with comprehensive features
            features, labels = create_sliding_windows(resampled, window_size, overlap)
            
            all_features.append(features)
            all_labels.append(labels)
            all_subjects.extend([subject] * len(features))
            
            # Calculate progress and time estimates
            elapsed_time = time.time() - start_time
            progress_pct = (file_idx / total_files) * 100
            
            if file_idx > 1:
                avg_time_per_file = elapsed_time / file_idx
                remaining_files = total_files - file_idx
                estimated_remaining_time = avg_time_per_file * remaining_files
                remaining_minutes = int(estimated_remaining_time // 60)
                remaining_seconds = int(estimated_remaining_time % 60)
                
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {pkl_file}: {len(features)} windows | ETA: {remaining_minutes}m {remaining_seconds}s")
            else:
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {pkl_file}: {len(features)} windows")
            
        except Exception as e:
            print(f"[{file_idx}/{total_files}] Error processing {pkl_file}: {e}")
            continue
    
    # Combine all data
    X = np.vstack(all_features) if all_features else np.array([])
    y = np.hstack(all_labels) if all_labels else np.array([])
    subjects = np.array(all_subjects)
    
    print(f"\nTotal windows processed: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Number of features per window: {X.shape[1] if len(X) > 0 else 0}")
    
    if len(y) > 0:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Classes found: {unique_classes}")
        print(f"Class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = count / len(y) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Remove undefined labels (class 0) for training
    valid_mask = y != 0
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    subjects_valid = subjects[valid_mask]
    
    print(f"\nAfter removing undefined labels:")
    print(f"Valid samples: {len(X_valid)}")
    
    if len(y_valid) > 0:
        unique_classes, class_counts = np.unique(y_valid, return_counts=True)
        print(f"Valid classes: {unique_classes}")
        print(f"Valid class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = count / len(y_valid) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Split data
    if len(X_valid) > 0:
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state, stratify=y_valid
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Save processed data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train_encoded)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val_encoded)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test_encoded)
        
        # Save preprocessing objects
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        
        # Create preprocessing info
        preprocessing_info = {
            'n_features': X_train_scaled.shape[1],
            'n_classes': len(label_encoder.classes_),
            'class_names': label_encoder.classes_.tolist(),
            'train_samples': len(X_train_scaled),
            'val_samples': len(X_val_scaled),
            'test_samples': len(X_test_scaled),
            'target_freq': target_freq,
            'window_size': window_size,
            'overlap': overlap,
            'total_subjects': len(np.unique(subjects_valid)),
            'subjects': np.unique(subjects_valid).tolist()
        }
        
        joblib.dump(preprocessing_info, os.path.join(output_dir, 'preprocessing_info.pkl'))
        
        print(f"\nPreprocessing complete!")
        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"Features: {X_train_scaled.shape[1]}")
        print(f"Classes: {len(label_encoder.classes_)} ({label_encoder.classes_})")
        print(f"Data saved to: {output_dir}")
        
        return preprocessing_info
    else:
        print("No valid data found!")
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD dataset')
    parser.add_argument('--data_dir', default='../../data/raw/wesad', help='Raw data directory')
    parser.add_argument('--output_dir', default='../../data/processed/wesad', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    print("="*70)
    print("WESAD PREPROCESSING")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if raw data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Raw data not found: {args.data_dir}")
        exit(1)
    
    print(f"📁 Raw data: {args.data_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"📊 Test size: {args.test_size}")
    print(f"📊 Validation size: {args.val_size}")
    
    # Run preprocessing
    info = preprocess_wesad(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    print(f"\n✅ WESAD preprocessing completed!")
    print(f"Preprocessing info: {info}")
