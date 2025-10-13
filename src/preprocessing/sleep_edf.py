"""
Sleep-EDF Dataset Preprocessing Module

This module handles the preprocessing of Sleep-EDF dataset:
- Loading EDF files with EEG/EOG signals
- Loading hypnograms from EDF+ annotations
- Filtering and segmentation
- Feature extraction (time and frequency domain)
- Normalization and train/val/test splitting

Supports both original Sleep-EDF and Sleep-EDF Expanded datasets.
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
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_sleep_edf_expanded_hypnogram(hypno_file: str, target_epoch_duration: int = 30) -> Tuple[List[str], List[int], int, int]:
    """
    Carrega hypnogram do Sleep-EDF Expanded baseado nas anotações EDF+
    
    Args:
        hypno_file: Caminho para o ficheiro *-Hypnogram.edf
        target_epoch_duration: Duração alvo das épocas (30s)
        
    Returns:
        tuple: (sleep_stages, epoch_durations, total_duration, n_epochs)
    """
    if not os.path.exists(hypno_file):
        print(f'❌ Ficheiro não encontrado: {hypno_file}')
        return None, None, None, None
    
    try:
        f = pyedflib.EdfReader(hypno_file)
        annotations = f.read_annotation()
        f.close()
        
        # Processar anotações
        sleep_stages = []
        epoch_durations = []
        
        for onset, duration, description in annotations:
            desc_str = description.decode('utf-8') if isinstance(description, bytes) else str(description)
            
            if 'Sleep stage' in desc_str:
                # Mapear para sleep stage
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
                
                # Converter duration para segundos
                if isinstance(duration, bytes):
                    try:
                        duration_sec = int(duration.decode('utf-8'))
                    except:
                        duration_sec = 30
                else:
                    duration_sec = int(duration)
                
                sleep_stages.append(stage)
                epoch_durations.append(duration_sec)
        
        # Calcular duração total e número de épocas
        total_duration = sum(epoch_durations)
        n_epochs = int(total_duration / target_epoch_duration)
        
        return sleep_stages, epoch_durations, total_duration, n_epochs
        
    except Exception as e:
        print(f'❌ Erro ao ler hypnogram: {e}')
        return None, None, None, None


def convert_hypnogram_to_30s_epochs(sleep_stages: List[str], epoch_durations: List[int], target_epoch_duration: int = 30) -> List[str]:
    """
    Converte hypnogram com durações variáveis para épocas de 30s
    
    Args:
        sleep_stages: Lista de sleep stages
        epoch_durations: Lista de durações em segundos
        target_epoch_duration: Duração alvo das épocas (30s)
        
    Returns:
        list: Sleep stages para épocas de 30s
    """
    target_epochs = []
    
    for stage, duration in zip(sleep_stages, epoch_durations):
        # Calcular quantas épocas de 30s cabem nesta duração
        n_epochs = int(duration / target_epoch_duration)
        
        # Adicionar épocas
        for _ in range(n_epochs):
            target_epochs.append(stage)
        
        # Se sobrar tempo, adicionar uma época extra
        remaining_time = duration % target_epoch_duration
        if remaining_time >= target_epoch_duration / 2:  # Se sobrar pelo menos 15s
            target_epochs.append(stage)
    
    return target_epochs


def load_sleep_edf_expanded_file(psg_path: str, hypno_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a single Sleep-EDF Expanded file (PSG + Hypnogram).
    
    Args:
        psg_path: Path to the PSG .edf file
        hypno_path: Path to the Hypnogram .edf file
        
    Returns:
        tuple: (signals, labels, info)
    """
    print(f'📁 Loading: {os.path.basename(psg_path)}')
    
    # Load PSG signals
    try:
        f = pyedflib.EdfReader(psg_path)
        
        # Extract information
        info = {
            'n_channels': f.signals_in_file,
            'duration': f.file_duration,
            'start_time': f.getStartdatetime(),
            'sample_rates': [f.getSampleFrequency(i) for i in range(f.signals_in_file)],
            'channel_labels': [f.getLabel(i) for i in range(f.signals_in_file)]
        }
        
        # Read EEG/EOG signals (channels 0-2)
        eeg_fpz_cz = f.readSignal(0)  # EEG Fpz-Cz
        eeg_pz_oz = f.readSignal(1)   # EEG Pz-Oz
        eog = f.readSignal(2)         # EOG horizontal
        
        f.close()
        
        # Combine signals (transpose to get channels x samples format)
        signals = np.array([eeg_fpz_cz, eeg_pz_oz, eog])
        
        print(f'   • Signals: {signals.shape}')
        print(f'   • Channels: {info["channel_labels"][:3]}')
        print(f'   • Sample rates: {info["sample_rates"][:3]} Hz')
        
    except Exception as e:
        print(f'❌ Error loading PSG: {e}')
        return None, None, None
    
    # Load hypnogram
    sleep_stages, epoch_durations, total_duration, n_epochs = load_sleep_edf_expanded_hypnogram(hypno_path)
    
    if sleep_stages is None:
        print(f'❌ Error loading hypnogram')
        return None, None, None
    
    # Convert to 30s epochs
    labels_30s = convert_hypnogram_to_30s_epochs(sleep_stages, epoch_durations)
    
    # Map labels to numbers
    label_mapping = {'W': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'R': 5, 'M': 6, '?': 7}
    labels = np.array([label_mapping.get(stage, 7) for stage in labels_30s])
    
    print(f'   • Labels: {len(labels)} epochs')
    print(f'   • Distribution: {np.bincount(labels)}')
    
    return signals, labels, info


def load_sleep_edf_file(edf_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a single Sleep-EDF file and its corresponding hypnogram.
    Now supports only Sleep-EDF Expanded format (.edf files).
    
    Args:
        edf_path: Path to PSG .edf file (recording)
        hyp_path: Path to Hypnogram .edf file (hypnogram)
    
    Returns:
        Tuple of (signals, labels, info_dict)
    """
    print(f"Loading {edf_path}...")
    
    # Check if files exist
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"PSG file not found: {edf_path}")
    if not os.path.exists(hyp_path):
        raise FileNotFoundError(f"Hypnogram file not found: {hyp_path}")
    
    # Use the new expanded format loader
    return load_sleep_edf_expanded_file(edf_path, hyp_path)


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
    
    # Check if signal is long enough for filtering
    min_length = 100  # Minimum samples needed for stable filtering
    
    for i in range(signals.shape[0]):
        if len(signals[i]) < min_length:
            # If signal is too short, just copy it without filtering
            filtered_signals[i] = signals[i]
            continue
            
        if i < 2:  # EEG channels
            # EEG: 0.5-32 Hz
            try:
                b_eeg, a_eeg = signal.butter(3, [0.5, 32], btype='band', fs=sfreq)
                filtered_signals[i] = signal.filtfilt(b_eeg, a_eeg, signals[i])
            except ValueError:
                # If filtering fails, use original signal
                filtered_signals[i] = signals[i]
        else:  # EOG channel
            # EOG: 0.5-10 Hz
            try:
                b_eog, a_eog = signal.butter(3, [0.5, 10], btype='band', fs=sfreq)
                filtered_signals[i] = signal.filtfilt(b_eog, a_eog, signals[i])
            except ValueError:
                # If filtering fails, use original signal
                filtered_signals[i] = signals[i]
    
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
    
    # Labels are already in epoch format (one label per 30-second epoch)
    # Take the minimum between available labels and epochs
    n_available_labels = len(labels)
    n_epochs_to_use = min(n_epochs, n_available_labels)
    
    if n_epochs_to_use == 0:
        # Return empty arrays if no epochs can be created
        return np.empty((signals.shape[0], 0, n_samples_epoch)), np.array([])
    
    # Reshape into epochs
    epochs = signals[:, :n_epochs_to_use * n_samples_epoch].reshape(signals.shape[0], n_epochs_to_use, n_samples_epoch)
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
    Complete preprocessing pipeline for Sleep-EDF Expanded dataset.
    
    Args:
        data_dir: Directory containing Sleep-EDF Expanded files
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with preprocessing info
    """
    print("="*70)
    print("SLEEP-EDF EXPANDED PREPROCESSING")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PSG and Hypnogram files
    psg_files = glob.glob(os.path.join(data_dir, '**/*-PSG.edf'), recursive=True)
    hypno_files = glob.glob(os.path.join(data_dir, '**/*-Hypnogram.edf'), recursive=True)
    
    print(f"Found {len(psg_files)} PSG files and {len(hypno_files)} Hypnogram files")
    
    all_features = []
    all_labels = []
    all_info = []
    
    # Process each PSG file
    total_files = len(psg_files)
    print(f"Processing {total_files} PSG files...")
    
    start_time = time.time()
    
    for file_idx, psg_file in enumerate(psg_files, 1):
        # Find corresponding hypnogram file by matching base prefix
        psg_basename = os.path.basename(psg_file)
        psg_prefix = psg_basename.replace('-PSG.edf', '')
        
        # Extract base prefix (subject + night + type, ignoring annotator)
        # SC4ssNEO -> SC4ssNE, ST7ssNJ0 -> ST7ssNJ
        if psg_prefix.startswith('SC'):
            base_prefix = psg_prefix[:-1]  # Remove last character (annotator)
        elif psg_prefix.startswith('ST'):
            base_prefix = psg_prefix[:-1]  # Remove last character (annotator)
        else:
            base_prefix = psg_prefix
        
        # Find matching hypnogram file
        hypno_file = None
        for hypno_path in hypno_files:
            hypno_basename = os.path.basename(hypno_path)
            hypno_prefix = hypno_basename.replace('-Hypnogram.edf', '')
            
            # Extract base prefix for hypnogram
            if hypno_prefix.startswith('SC'):
                hypno_base_prefix = hypno_prefix[:-1]  # Remove last character (annotator)
            elif hypno_prefix.startswith('ST'):
                hypno_base_prefix = hypno_prefix[:-1]  # Remove last character (annotator)
            else:
                hypno_base_prefix = hypno_prefix
            
            if base_prefix == hypno_base_prefix:
                hypno_file = hypno_path
                break
        
        if hypno_file is None:
            print(f"[{file_idx}/{total_files}] Warning: No hypnogram found for {psg_basename} (base: {base_prefix})")
            continue
        
        try:
            # Load file
            signals, labels, info = load_sleep_edf_file(psg_file, hypno_file)
            
            if signals is None or labels is None:
                print(f"[{file_idx}/{total_files}] Error: Failed to load {os.path.basename(psg_file)}")
                continue
            
            # Filter signals (EEG/EOG at 100 Hz)
            filtered_signals = filter_signals(signals, 100.0)  # Fixed sample rate for Expanded
            
            # Segment into epochs (30s epochs)
            epochs, epoch_labels = segment_epochs(filtered_signals, labels, 100.0)
            
            # Extract features for each epoch
            for i in range(epochs.shape[1]):
                features = extract_sleep_features(epochs[:, i], 100.0)
                all_features.append(features)
                all_labels.append(epoch_labels[i])
            
            all_info.append(info)
            
            # Calculate progress and time estimates
            elapsed_time = time.time() - start_time
            progress_pct = (file_idx / total_files) * 100
            
            if file_idx > 1:
                avg_time_per_file = elapsed_time / file_idx
                remaining_files = total_files - file_idx
                estimated_remaining_time = avg_time_per_file * remaining_files
                remaining_minutes = int(estimated_remaining_time // 60)
                remaining_seconds = int(estimated_remaining_time % 60)
                
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {os.path.basename(psg_file)}: {epochs.shape[1]} epochs | ETA: {remaining_minutes}m {remaining_seconds}s")
            else:
                print(f"[{file_idx}/{total_files}] ({progress_pct:.1f}%) Processed {os.path.basename(psg_file)}: {epochs.shape[1]} epochs")
            
        except Exception as e:
            print(f"[{file_idx}/{total_files}] Error processing {os.path.basename(psg_file)}: {e}")
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
    
    # Labels are already encoded (W=0, 1=1, 2=2, 3=3, 4=4, R=5, M=6, ?=7)
    # Map to standard 5-class format: W=0, N1=1, N2=2, N3=3, R=4
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 0, 7: 0}  # Map 4->3 (N3), M->W, ?->W
    y_encoded = np.array([label_mapping.get(label, 0) for label in y])
    
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
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save preprocessing info
    preprocessing_info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'class_names': ['W', 'N1', 'N2', 'N3', 'R'],
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
                                                     StandardScaler, Dict]:
    """
    Load preprocessed Sleep-EDF data.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, info)
    """
    print(f"Loading processed Sleep-EDF data from {data_dir}...")
    
    # Load arrays
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load scaler and info
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    info = joblib.load(os.path.join(data_dir, 'preprocessing_info.pkl'))
    
    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, info


if __name__ == "__main__":
    # Example usage
    data_dir = "/content/drive/MyDrive/mhealth-data/raw/sleep-edf"
    output_dir = "/content/drive/MyDrive/mhealth-data/processed/sleep-edf"
    
    # Run preprocessing
    info = preprocess_sleep_edf(data_dir, output_dir)
    print(f"Preprocessing info: {info}")
