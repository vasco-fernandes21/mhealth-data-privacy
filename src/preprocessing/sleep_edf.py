"""
Sleep-EDF Dataset Preprocessing Module

This module handles the preprocessing of Sleep-EDF dataset:
- Loading EDF files with EEG/EOG signals
- Filtering and segmentation
- Feature extraction (time and frequency domain)
- Normalization and train/val/test splitting
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
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_sleep_edf_file(edf_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a single Sleep-EDF file and its corresponding hypnogram.
    Supports both .edf files and .rec/.hyp files (PhysioNet format).
    
    Args:
        edf_path: Path to .edf or .rec file (recording)
        hyp_path: Path to .edf or .hyp file (hypnogram)
    
    Returns:
        Tuple of (signals, labels, info_dict)
    """
    print(f"Loading {edf_path}...")
    
    # Check file format
    if edf_path.endswith('.rec'):
        return load_physionet_file(edf_path, hyp_path)
    else:
        return load_edf_file(edf_path, hyp_path)


def load_edf_file(edf_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load EDF format files.
    """
    # Load recording
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Get signal info
    sfreq = raw.info['sfreq']
    signal_labels = raw.ch_names
    
    # Extract signals (assuming standard channel names)
    # Fpz-Cz, Pz-Oz (EEG), ROC-LOC (EOG)
    eeg_fpz_cz = raw.get_data(picks='Fpz-Cz')[0] if 'Fpz-Cz' in signal_labels else raw.get_data()[0]
    eeg_pz_oz = raw.get_data(picks='Pz-Oz')[0] if 'Pz-Oz' in signal_labels else raw.get_data()[1]
    eog = raw.get_data(picks='ROC-LOC')[0] if 'ROC-LOC' in signal_labels else raw.get_data()[2]
    
    # Load hypnogram
    hyp_raw = mne.io.read_raw_edf(hyp_path, preload=True, verbose=False)
    hypnogram = hyp_raw.get_data()[0]
    
    # Create info dict
    info = {
        'sfreq': sfreq,
        'channels': signal_labels,
        'duration': len(eeg_fpz_cz) / sfreq
    }
    
    # Combine signals
    signals_combined = np.vstack([eeg_fpz_cz, eeg_pz_oz, eog])
    
    return signals_combined, hypnogram, info


def load_physionet_file(rec_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load EDF files with .rec/.hyp extensions (not true PhysioNet format).
    """
    import pyedflib
    
    # Load recording using pyedflib
    f = pyedflib.EdfReader(rec_path)
    
    # Get signal info
    n_signals = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sfreq = f.getSampleFrequency(0)  # Assume all signals have same frequency
    
    # Extract signals
    eeg_fpz_cz = f.readSignal(0)  # First signal
    eeg_pz_oz = f.readSignal(1) if n_signals > 1 else eeg_fpz_cz  # Second signal
    eog = f.readSignal(2) if n_signals > 2 else eeg_pz_oz  # Third signal
    
    f.close()
    
    # Load hypnogram
    labels = load_physionet_hypnogram(hyp_path)
    
    # Create info dict
    info = {
        'sfreq': sfreq,
        'channels': signal_labels,
        'duration': len(eeg_fpz_cz) / sfreq
    }
    
    # Combine signals
    signals_combined = np.vstack([eeg_fpz_cz, eeg_pz_oz, eog])
    
    return signals_combined, labels, info


def load_physionet_hypnogram(hyp_path: str) -> np.ndarray:
    """
    Load hypnogram from PhysioNet .hyp file.
    """
    with open(hyp_path, 'r') as f:
        content = f.read()
    
    # Extract numeric values from the entire content
    # The hypnogram data is at the end of the file as individual digits
    hypnogram_data = []
    
    # Look for digits in the content (skip header information)
    # The hypnogram starts after the header and contains only digits 0-6
    for char in content:
        if char.isdigit() and int(char) <= 6:  # Valid sleep stage values
            hypnogram_data.append(int(char))
    
    # Convert to numpy array and map to standard sleep stages
    # PhysioNet: 0=W, 1=S1, 2=S2, 3=S3, 4=S4, 5=R, 6=M
    # Standard: 0=W, 1=N1, 2=N2, 3=N3, 4=R
    stage_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 0}  # Map S3/S4 to N3, M to W
    
    labels = np.array([stage_mapping.get(stage, 0) for stage in hypnogram_data])
    
    return labels
    
    # Convert hypnogram to 30-second epochs
    epoch_duration = 30  # seconds
    n_samples_epoch = int(sfreq * epoch_duration)
    n_epochs = len(hypnogram) // n_samples_epoch
    
    # Truncate to complete epochs
    signals = np.array([eeg_fpz_cz, eeg_pz_oz, eog])[:, :n_epochs * n_samples_epoch]
    labels = hypnogram[:n_epochs * n_samples_epoch]
    
    info = {
        'sfreq': sfreq,
        'n_epochs': n_epochs,
        'epoch_duration': epoch_duration,
        'signal_labels': signal_labels
    }
    
    return signals, labels, info


def filter_signals(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Apply Butterworth bandpass filters to signals.
    
    Args:
        signals: Array of shape (n_channels, n_samples)
        sfreq: Sampling frequency
    
    Returns:
        Filtered signals
    """
    filtered_signals = np.zeros_like(signals)
    
    # EEG: 0.5-32 Hz
    b_eeg, a_eeg = signal.butter(3, [0.5, 32], btype='band', fs=sfreq)
    filtered_signals[0] = signal.filtfilt(b_eeg, a_eeg, signals[0])  # Fpz-Cz
    filtered_signals[1] = signal.filtfilt(b_eeg, a_eeg, signals[1])  # Pz-Oz
    
    # EOG: 0.5-10 Hz
    b_eog, a_eog = signal.butter(3, [0.5, 10], btype='band', fs=sfreq)
    filtered_signals[2] = signal.filtfilt(b_eog, a_eog, signals[2])  # ROC-LOC
    
    return filtered_signals


def segment_epochs(signals: np.ndarray, labels: np.ndarray, sfreq: float, 
                   epoch_duration: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment signals into epochs and extract epoch labels.
    
    Args:
        signals: Array of shape (n_channels, n_samples)
        labels: Hypnogram labels
        sfreq: Sampling frequency
        epoch_duration: Duration of each epoch in seconds
    
    Returns:
        Tuple of (epochs, epoch_labels)
    """
    n_samples_epoch = int(sfreq * epoch_duration)
    n_epochs = signals.shape[1] // n_samples_epoch
    
    # Reshape into epochs
    epochs = signals[:, :n_epochs * n_samples_epoch].reshape(3, n_epochs, n_samples_epoch)
    
    # Labels are already in epoch format (one label per 30-second epoch)
    # Take the minimum between available labels and epochs
    n_available_labels = len(labels)
    n_epochs_to_use = min(n_epochs, n_available_labels)
    
    # Reshape epochs to match available labels
    epochs = epochs[:, :n_epochs_to_use, :]
    epoch_labels = labels[:n_epochs_to_use]
    
    return epochs, epoch_labels


def extract_sleep_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract features from a single epoch.
    
    Args:
        epoch: Array of shape (n_channels, n_samples)
        sfreq: Sampling frequency
    
    Returns:
        Feature vector of length 24 (8 features per channel)
    """
    features = []
    
    for channel in epoch:
        # Time domain features
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        max_val = np.max(channel)
        min_val = np.min(channel)
        
        # Frequency domain features (Power Spectral Density)
        freqs, psd = signal.welch(channel, sfreq, nperseg=min(256, len(channel)))
        
        # Band power features
        delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
        
        # Combine features for this channel
        channel_features = [mean_val, std_val, max_val, min_val, delta, theta, alpha, beta]
        features.extend(channel_features)
    
    return np.array(features)


def preprocess_sleep_edf(data_dir: str, output_dir: str, 
                        test_size: float = 0.15, val_size: float = 0.15,
                        random_state: int = 42) -> Dict:
    """
    Complete preprocessing pipeline for Sleep-EDF dataset.
    
    Args:
        data_dir: Directory containing Sleep-EDF files
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("SLEEP-EDF PREPROCESSING")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all recording files (EDF or REC format)
    edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf') and not f.endswith('.hyp.edf')]
    rec_files = [f for f in os.listdir(data_dir) if f.endswith('.rec')]
    hyp_files = [f for f in os.listdir(data_dir) if f.endswith('.hyp.edf')]
    hyp_physionet_files = [f for f in os.listdir(data_dir) if f.endswith('.hyp') and not f.endswith('.hyp.edf')]
    
    # Combine all recording files
    recording_files = edf_files + rec_files
    hypnogram_files = hyp_files + hyp_physionet_files
    
    print(f"Found {len(recording_files)} recording files and {len(hypnogram_files)} hypnogram files")
    
    all_features = []
    all_labels = []
    all_info = []
    
    # Process each file
    for recording_file in recording_files:
        # Find corresponding hypnogram
        subject_id = recording_file.replace('.edf', '').replace('.rec', '')
        
        # Find corresponding hypnogram file
        hyp_file = None
        if recording_file.endswith('.edf'):
            hyp_file = f"{subject_id}-PSG.edf.hyp.edf"
        else:  # .rec file
            hyp_file = f"{subject_id}.hyp"
        
        if hyp_file not in hypnogram_files:
            print(f"Warning: No hypnogram found for {recording_file}: {hyp_file}")
            continue
        
        recording_path = os.path.join(data_dir, recording_file)
        hyp_path = os.path.join(data_dir, hyp_file)
        
        try:
            # Load file
            signals, labels, info = load_sleep_edf_file(recording_path, hyp_path)
            
            # Filter signals
            filtered_signals = filter_signals(signals, info['sfreq'])
            
            # Segment into epochs
            epochs, epoch_labels = segment_epochs(filtered_signals, labels, info['sfreq'])
            
            # Extract features for each epoch
            for i in range(epochs.shape[1]):
                features = extract_sleep_features(epochs[:, i], info['sfreq'])
                all_features.append(features)
                all_labels.append(epoch_labels[i])
            
            all_info.append(info)
            print(f"Processed {recording_file}: {epochs.shape[1]} epochs")
            
        except Exception as e:
            print(f"Error processing {recording_file}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nTotal epochs processed: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    if len(y) > 0:
        print(f"Label distribution: {np.bincount(y.astype(int))}")
    else:
        print("No data processed - check file formats and paths")
        return None
    
    # Encode labels (W=0, N1=1, N2=2, N3=3, R=4)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
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
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'class_names': label_encoder.classes_,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'files_processed': len(all_info)
    }
    
    joblib.dump(preprocessing_info, os.path.join(output_dir, 'preprocessing_info.pkl'))
    
    print(f"\nPreprocessing complete!")
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Classes: {preprocessing_info['n_classes']} ({preprocessing_info['class_names']})")
    print(f"Data saved to: {output_dir}")
    
    return preprocessing_info


def load_processed_sleep_edf(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray, np.ndarray, 
                                                     StandardScaler, LabelEncoder, Dict]:
    """
    Load preprocessed Sleep-EDF data.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, info)
    """
    print(f"Loading processed Sleep-EDF data from {data_dir}...")
    
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
    data_dir = "/content/drive/MyDrive/mhealth-data/raw/sleep-edf"
    output_dir = "/content/drive/MyDrive/mhealth-data/processed/sleep-edf"
    
    # Run preprocessing
    info = preprocess_sleep_edf(data_dir, output_dir)
    print(f"Preprocessing info: {info}")
