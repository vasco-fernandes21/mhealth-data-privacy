#!/usr/bin/env python3
"""
Train PyTorch CNN-LSTM baseline for WESAD binary stress classification.

Optimized to match/exceed TensorFlow baseline accuracy:
- Added MaxPooling1D layers for spatial dimension reduction
- Used SpatialDropout1D equivalent (Dropout) at 0.2 rate
- Added L2 regularization via weight_decay
- Improved learning rate scheduling with ReduceLROnPlateau
- Dense layers with dropout for better generalization
- Batch size reduced to 32 for better gradient estimation
- Kernel sizes adjusted to match TensorFlow architecture
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import device utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from device_utils import get_optimal_device, print_device_info
from preprocessing.wesad import load_processed_wesad_temporal

# Fix random seeds for reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configure device-aware seeding and performance
device = get_optimal_device()
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device.type == "mps":
    torch.mps.manual_seed(SEED)
    # MPS doesn't have deterministic/benchmark options like CUDA

# --- Custom CNN-LSTM model with BatchNorm & SpatialDropout equivalent ---
class CNNLSTMWESAD(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        # First Conv block - similar to TensorFlow
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(0.2)  # SpatialDropout equivalent

        # Second Conv block - similar to TensorFlow
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.2)

        # LSTM with regularization
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1,
                            batch_first=True, bidirectional=False,
                            dropout=0.5 if num_classes > 2 else 0.0)
        self.dropout_lstm = nn.Dropout(0.5)

        # Dense layers with L2 regularization effect
        self.fc1 = nn.Linear(32, 32)
        self.dropout_dense = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # LSTM
        x = x.permute(0, 2, 1)  # (batch, timesteps, features) for LSTM
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # last layer
        x = self.dropout_lstm(hn)

        # Dense layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout_dense(x)
        out = self.fc2(x)
        return out

# --- Data augmentation & oversampling ---
def _simple_oversample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())
    max_count = max(counts.values())
    Xb, yb = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        reps = int(np.ceil(max_count / len(idx)))
        idx_rep = np.tile(idx, reps)[:max_count]
        Xb.append(X[idx_rep])
        yb.append(y[idx_rep])
    Xb = np.concatenate(Xb, axis=0)
    yb = np.concatenate(yb, axis=0)
    perm = np.random.permutation(len(yb))
    return Xb[perm], yb[perm]

def _augment_temporal(X: np.ndarray, noise_std: float = 0.01, max_time_shift: int = 8, seed: int = 42) -> np.ndarray:
    """Apply deterministic temporal augmentation with fixed seed."""
    rng = np.random.default_rng(seed)
    X_aug = X.copy()

    # Generate all random numbers at once for complete determinism
    n_samples = X_aug.shape[0]
    noise = rng.normal(0, noise_std, size=X_aug.shape)
    shifts = rng.integers(-max_time_shift, max_time_shift + 1, size=n_samples)

    # Apply noise
    X_aug += noise

    # Apply time shifts
    if max_time_shift > 0:
        for i in range(n_samples):
            shift = shifts[i]
            if shift != 0:
                if shift > 0:
                    X_aug[i, :, shift:] = X_aug[i, :, :-shift]
                else:
                    X_aug[i, :, :shift] = X_aug[i, :, -shift:]

    return X_aug

def build_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      batch_size: int = 64,
                      balance: str = 'class_weight',
                      augment: bool = True,
                      noise_std: float = 0.01,
                      max_time_shift: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if balance == 'oversample':
        X_train, y_train = _simple_oversample(X_train, y_train)
    if augment:
        X_train = _augment_temporal(X_train, noise_std=noise_std, max_time_shift=max_time_shift)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# --- Training & evaluation ---
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str, clip_grad_norm: float = 1.0) -> Tuple[float, float]:
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

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        running_loss += loss.item() * Xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
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

@torch.no_grad()
def evaluate_full_metrics(model: nn.Module, loader: DataLoader, device: str, class_names):
    model.eval()
    y_true, y_pred = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        preds = model(Xb).argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

# --- Main training loop ---
def main():
    print("=" * 70)
    print("WESAD BINARY STRESS CLASSIFICATION - PYTORCH IMPROVED")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_dir = str(base_dir / "models/wesad/baseline_torch")
    results_dir = str(base_dir / "results/wesad/baseline")

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = load_processed_wesad_temporal(data_dir)
    if X_train.shape[1] < X_train.shape[2]:
        pass
    else:
        X_train, X_val, X_test = [np.transpose(x, (0, 2, 1)) for x in (X_train, X_val, X_test)]

    print(f"\nDataset info: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, Classes={info['class_names']}")
    train_loader, val_loader, test_loader = build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                                                              batch_size=32, balance='oversample', augment=True,
                                                              noise_std=0.01, max_time_shift=8)

    # Device is already configured at module level - just print info
    print_device_info()
    model = CNNLSTMWESAD(input_channels=X_train.shape[1], num_classes=len(info['class_names'])).to(device)

    # Initialize weights deterministically for reproducible results
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    model.apply(init_weights)

    # Class weights for imbalanced data
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train),
                                 dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Adam optimizer with L2 regularization (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # More sophisticated learning rate scheduler - similar to ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-6
    )

    # Gradient clipping to prevent exploding gradients
    clip_grad_norm = 1.0

    os.makedirs(models_dir, exist_ok=True)
    best_val_acc, epochs_no_improve = -1.0, 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    start_time = time.time()
    epochs, patience = 100, 8

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, clip_grad_norm)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)

        history["loss"].append(float(train_loss))
        history["accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(f"Epoch {epoch:03d}: loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'model_wesad_binary.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    training_time = time.time() - start_time
    model.load_state_dict(torch.load(os.path.join(models_dir, 'model_wesad_binary.pth'), map_location=device))
    results = evaluate_full_metrics(model, test_loader, device, info['class_names'])

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'history_wesad_binary.json'), 'w') as f:
        json.dump({**history, 'epochs': len(history['loss']), 'training_time_seconds': training_time}, f, indent=2)
    with open(os.path.join(models_dir, 'results_wesad_binary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (PyTorch Improved)!")
    print("=" * 70)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
