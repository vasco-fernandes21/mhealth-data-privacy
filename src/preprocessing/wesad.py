"""
WESAD Dataset Preprocessing Module

This module handles the preprocessing of WESAD dataset:
- Loading pickle files with physiological signals
- Signal filtering and resampling
- Feature extraction (time and frequency domain)
- Normalization and train/val/test splitting
"""

import numpy as np
import pandas as pd
import pickle
from scipy import signal
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
    Load a single WESAD pickle file.
    
    Args:
        pkl_path: Path to .pkl file
    
    Returns:
        Dictionary with signals and labels
    """
    print(f"Loading {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract signals from RespiBAN (chest device) - already synchronized with labels
    ecg = data['signal']['chest']['ECG']  # Electrocardiogram (700 Hz)
    eda = data['signal']['chest']['EDA']  # Electrodermal Activity (700 Hz)
    temp = data['signal']['chest']['Temp']  # Temperature (700 Hz)
    acc = data['signal']['chest']['ACC']  # Acceleration (700 Hz)
    
    # Extract labels
    labels = data['label']  # Labels: 0=not defined, 1=baseline, 2=stress, 3=amusement, 4=meditation
    
    return {
        'ecg': ecg,
        'eda': eda,
        'temp': temp,
        'acc': acc,
        'labels': labels,
        'sfreq': {
            'ecg': 700,
            'eda': 700,
            'temp': 700,
            'acc': 700
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
    
    # All signals are at 700 Hz, resample to target frequency
    original_length = len(signals_dict['ecg'])
    target_length = int(original_length * target_freq / sfreq['ecg'])
    
    # Resample all signals from 700 Hz to target frequency
    resampled['ecg'] = signal.resample(signals_dict['ecg'], target_length)
    resampled['eda'] = signal.resample(signals_dict['eda'], target_length)
    resampled['temp'] = signal.resample(signals_dict['temp'], target_length)
    resampled['acc'] = signal.resample(signals_dict['acc'], target_length)
    
    # Use stratified sampling for labels to maintain class proportions
    original_length = len(signals_dict['labels'])
    indices = np.linspace(0, original_length-1, target_length, dtype=int)
    resampled['labels'] = signals_dict['labels'][indices]
    
    resampled['sfreq'] = target_freq
    
    return resampled


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
    
    # Find all pickle files
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
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


if __name__ == "__main__":
    # Example usage
    data_dir = "/content/drive/MyDrive/mhealth-data/raw/wesad"
    output_dir = "/content/drive/MyDrive/mhealth-data/processed/wesad"
    
    # Run preprocessing
    info = preprocess_wesad(data_dir, output_dir)
    print(f"Preprocessing info: {info}")
