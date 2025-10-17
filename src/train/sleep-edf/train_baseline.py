#!/usr/bin/env python3
"""
Train PyTorch LSTM baseline for Sleep-EDF dataset

Environment Variables (for multiple runs):
- TRAIN_SEED: Random seed (default: 42)
- MODEL_DIR: Directory to save model (default: models/sleep-edf/baseline)
- RESULTS_DIR: Directory to save results (default: results/sleep-edf/baseline)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Tuple

# Add src to path and import device utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from device_utils import get_optimal_device, print_device_info
from preprocessing.sleep_edf import load_processed_sleep_edf

# --- Progress Bar with ETA ---
class ProgressBar:
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, n: int = 1):
        self.current += n
        now = time.time()

        # Update every 1% or at least every 5 seconds
        if (self.current % max(1, self.total // 100) == 0 or
            now - self.last_update >= 5.0):
            self._display()
            self.last_update = now

    def _display(self):
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        percentage = min(100, (self.current / self.total) * 100)

        # Format time
        if elapsed < 60:
            elapsed_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed/60:.1f}m"
        else:
            elapsed_str = f"{elapsed/3600:.1f}h"

        if remaining < 60:
            eta_str = f"{remaining:.0f}s"
        elif remaining < 3600:
            eta_str = f"{remaining/60:.0f}m"
        else:
            eta_str = f"{remaining/3600:.0f}h"

        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"\r{self.description}: [{bar}] {percentage:5.1f}% | "
              f"{self.current}/{self.total} | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str}", end="", flush=True)

    def finish(self):
        self.update(self.total - self.current)
        print()  # New line

# --- Simple LSTM Model ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# --- Data Loading ---
class SleepEDFDataLoader:
    """Simple data loader for Sleep-EDF dataset"""

    def __init__(self, data_dir: str, batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Load arrays
        self.X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        self.y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        self.X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        self.y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        self.X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

        print(f"Dataset loaded: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def get_dataloaders(self, window_size: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with windowing"""

        print(f"Creating LSTM windows (window_size={window_size})...")

        # Create windows
        def create_windows(X, y, window_size):
            n_samples, n_features = X.shape
            n_windows = n_samples - window_size + 1

            X_windows = np.zeros((n_windows, window_size, n_features), dtype=X.dtype)
            y_windows = np.zeros(n_windows, dtype=y.dtype)

            # Progress bar for window creation
            progress = ProgressBar(n_windows, "Windows")

            for i in range(n_windows):
                X_windows[i] = X[i:i+window_size]
                y_windows[i] = y[i+window_size-1]
                progress.update(1)

            progress.finish()
            return X_windows, y_windows

        # Create windows
        X_train_windows, y_train_windows = create_windows(self.X_train, self.y_train, window_size)
        X_val_windows, y_val_windows = create_windows(self.X_val, self.y_val, window_size)
        X_test_windows, y_test_windows = create_windows(self.X_test, self.y_test, window_size)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_windows, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_windows, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_windows, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_windows, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_windows, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_windows, dtype=torch.long)

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Data loaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        return train_loader, val_loader, test_loader

# --- Training Functions ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * Xb.size(0)
        correct += (outputs.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        outputs = model(Xb)
        loss = criterion(outputs, yb)

        running_loss += loss.item() * Xb.size(0)
        correct += (outputs.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total

def main():
    # Set random seed for reproducibility
    import random
    SEED = int(os.environ.get('TRAIN_SEED', 42))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    seed_info = f" (SEED={SEED})" if SEED != 42 else ""
    print("="*70)
    print(f"TRAINING SLEEP-EDF BASELINE MODEL{seed_info}")
    print("="*70)

    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/sleep-edf")
    
    # Allow override via environment variables (for multiple runs)
    models_output_dir = os.environ.get('MODEL_DIR', str(base_dir / "models/sleep-edf/baseline"))
    results_output_dir = os.environ.get('RESULTS_DIR', str(base_dir / "results/sleep-edf/baseline"))

    # Create directories
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # Load processed data
    print("Loading processed Sleep-EDF data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = load_processed_sleep_edf(data_dir)

    print(f"\nDataset info:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")

    # Create data loaders
    data_loader = SleepEDFDataLoader(data_dir, batch_size=64)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(window_size=10)

    # Model configuration
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    num_classes = len(info['class_names'])

    print(f"\nModel configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Num classes: {num_classes}")

    # Configure device for hardware acceleration (CUDA > MPS > CPU)
    device = get_optimal_device()

    # Initialize model
    print_device_info()
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    print(f"\nStarting training on {device}...")
    num_epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_progress = ProgressBar(len(train_loader.dataset), f"Epoch {epoch+1:3d} - Training")

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            train_total += y_batch.size(0)

            train_progress.update(X_batch.size(0))

        train_progress.finish()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if epoch > 10 and val_loss > best_val_loss:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break
        else:
            patience_counter = 0
            best_val_loss = val_loss

    # Evaluate on test set
    print(f"\nEvaluating model...")
    model.eval()
    test_correct = 0
    test_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)

            test_correct += (predictions == y_batch).sum().item()
            test_total += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    test_acc = test_correct / test_total

    # Calculate detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save results
    results = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': info['class_names']
    }

    with open(os.path.join(models_output_dir, 'results_sleep_edf.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print final results
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1-Score: {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  Predicted →")
    print(f"  Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {info['class_names'][i]:8s}: {row}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
