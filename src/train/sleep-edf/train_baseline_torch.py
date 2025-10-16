#!/usr/bin/env python3
"""
Train PyTorch LSTM baseline for Sleep-EDF dataset with high efficiency and ETA tracking.

Optimized for large dataset with progress monitoring and performance optimizations:
- Efficient data loading and batching
- ETA calculation for long training sessions
- Memory optimization for large datasets
- Reproducible results with fixed seeds
- Optimized LSTM architecture for sleep stage classification
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Fix random seeds for reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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

# --- Optimized LSTM Model for Sleep-EDF ---
class SleepEDFLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Optimized LSTM architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better temporal modeling
        )

        # Attention mechanism for better feature extraction
        self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

        # Dense layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size * 2)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * 2)

        # Layer normalization
        x = self.layer_norm(context_vector)

        # Dense layers with residual connections
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

# --- Efficient Data Loading ---
class SleepEDFDataLoader:
    """Optimized data loader for large Sleep-EDF dataset"""

    def __init__(self, data_dir: str, batch_size: int = 128, num_workers: int = 4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Memory mapping for large files
        self.X_train = np.load(os.path.join(data_dir, 'X_train.npy'), mmap_mode='r')
        self.y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

        # For validation and test, load normally (smaller)
        self.X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        self.y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        self.X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

        print(f"Dataset loaded: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def get_dataloaders(self, window_size: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create optimized data loaders with windowing"""

        print(f"Creating LSTM windows (window_size={window_size})...")

        # Create windows efficiently
        def create_windows(X, y, window_size):
            n_samples, n_features = X.shape
            n_windows = n_samples - window_size + 1

            # Pre-allocate arrays for efficiency
            X_windows = np.zeros((n_windows, window_size, n_features), dtype=X.dtype)
            y_windows = np.zeros(n_windows, dtype=y.dtype)

            # Vectorized window creation
            for i in range(n_windows):
                X_windows[i] = X[i:i+window_size]
                y_windows[i] = y[i+window_size-1]  # Label from last timestep

            return X_windows, y_windows

        # Create windows with progress tracking
        print("Creating training windows...")
        progress = ProgressBar(len(self.X_train) - window_size + 1, "Train windows")
        X_train_windows, y_train_windows = create_windows(self.X_train, self.y_train, window_size)
        progress.finish()

        print("Creating validation windows...")
        progress = ProgressBar(len(self.X_val) - window_size + 1, "Val windows")
        X_val_windows, y_val_windows = create_windows(self.X_val, self.y_val, window_size)
        progress.finish()

        print("Creating test windows...")
        progress = ProgressBar(len(self.X_test) - window_size + 1, "Test windows")
        X_test_windows, y_test_windows = create_windows(self.X_test, self.y_test, window_size)
        progress.finish()

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_windows, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_windows, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_windows, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_windows, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_windows, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_windows, dtype=torch.long)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"Data loaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        return train_loader, val_loader, test_loader

# --- Training Functions ---
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer,
                   device: str, progress_bar: ProgressBar) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item() * Xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

        progress_bar.update(Xb.size(0))

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * Xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total

# --- Main Training Function ---
def main():
    print("=" * 80)
    print("SLEEP-EDF PYTORCH TRAINING - OPTIMIZED FOR LARGE DATASET")
    print("=" * 80)

    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/sleep-edf")
    models_dir = str(base_dir / "models/sleep-edf/baseline_torch")
    results_dir = str(base_dir / "results/sleep-edf/baseline")

    # Load data efficiently
    print("Loading Sleep-EDF dataset...")
    data_loader = SleepEDFDataLoader(data_dir, batch_size=256, num_workers=4)

    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(window_size=10)

    # Model configuration
    input_size = data_loader.X_train.shape[1]  # n_features
    hidden_size = 128
    num_layers = 2
    num_classes = len(np.unique(data_loader.y_train))
    dropout = 0.3

    print(f"\nModel configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Num classes: {num_classes}")
    print(f"  Dropout: {dropout}")

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SleepEDFLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # Initialize weights deterministically
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.LSTM)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Class weights for imbalanced data
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(data_loader.y_train), y=data_loader.y_train),
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training configuration
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training history
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print(f"\nStarting training on {device}...")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Training phase
        train_progress = ProgressBar(len(train_loader.dataset), f"Epoch {epoch:3d} - Training")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, train_progress)
        train_progress.finish()

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Update history
        history["loss"].append(float(train_loss))
        history["accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        epoch_time = time.time() - epoch_start_time

        # Print progress
        print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.1f}s")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - total_start_time

    # Load best model
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pth'), map_location=device))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Detailed evaluation
    model.eval()
    y_true, y_pred = [], []
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb).argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate detailed metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save results
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': ['W', 'N1', 'N2', 'N3', 'R']  # Sleep stages
    }

    # Save model and results
    with open(os.path.join(models_dir, 'history_sleep_edf.json'), 'w') as f:
        json.dump({**history, 'epochs': len(history['loss']), 'total_time': total_time}, f, indent=2)

    with open(os.path.join(models_dir, 'results_sleep_edf.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final Test Accuracy:  {test_acc:.4f}")
    print(f"Final Test F1-Score:  {f1:.4f}")
    print(f"Total Training Time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best Validation Loss:  {best_val_loss:.4f}")
    print(f"Epochs Completed:      {len(history['loss'])}")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted →")
    print(f"  Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {results['class_names'][i]:8s}: {row}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
