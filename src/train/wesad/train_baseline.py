#!/usr/bin/env python3
"""
Train PyTorch baseline for WESAD binary stress classification.

LSTM-only architecture optimized for:
- Good performance with reduced complexity
- Compatibility with Differential Privacy and Federated Learning
- Stable convergence across different environments
- Efficient training and inference

Environment Variables (for multiple runs):
- TRAIN_SEED: Random seed (default: 42)
- MODEL_DIR: Directory to save model (default: models/wesad/baseline)
- RESULTS_DIR: Directory to save results (default: results/wesad/baseline)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import device utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from device_utils import get_optimal_device, print_device_info
from preprocessing.wesad import load_processed_wesad_temporal

# Random seeds for reproducible results
SEED = int(os.environ.get('TRAIN_SEED', 42))
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

# --- Simplified LSTM-only model for DP/FL compatibility ---
class SimpleLSTMWESAD(nn.Module):
    """
    LSTM-only architecture optimized for WESAD binary classification.
    - Simpler than CNN-LSTM for better DP compatibility
    - Maintained performance through LSTM optimization
    - GroupNorm for DP compatibility
    """
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        # Initial projection to reduce dimensionality
        self.input_proj = nn.Linear(input_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)
        self.input_drop = nn.Dropout(0.2)

        # LSTM for temporal modeling (optimized for DP)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # 64*2 for bidirectional
        self.lstm_drop = nn.Dropout(0.3)

        # Dense layers for classification
        self.fc1 = nn.Linear(128, 64)
        self.fc1_norm = nn.GroupNorm(num_groups=8, num_channels=64)
        self.fc1_drop = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        # Convert to (batch, timesteps, channels) for LSTM
        x = x.permute(0, 2, 1)  # (batch, timesteps, channels)

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_drop(x)

        # LSTM processing
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last hidden state from both directions
        x = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, 128)
        x = self.lstm_norm(x.unsqueeze(2)).squeeze(2)
        x = self.lstm_drop(x)

        # Dense classification
        x = self.fc1(x)
        x = self.fc1_norm(x.unsqueeze(2)).squeeze(2)
        x = torch.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc2_drop(x)

        out = self.fc3(x)
        return out

# --- Data loading ---
def build_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    """
    Comprehensive evaluation with all metrics
    """
    model.eval()
    y_true, y_pred = [], []

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        predictions = logits.argmax(dim=1)

        y_true.extend(yb.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

# --- Main training loop ---
def main():
    # Print seed info
    seed_info = f" (SEED={SEED})" if SEED != 42 else ""
    print("=" * 70)
    print(f"WESAD BINARY STRESS CLASSIFICATION - PYTORCH BASELINE{seed_info}")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    
    # Allow override via environment variables (for multiple runs)
    models_dir = os.environ.get('MODEL_DIR', str(base_dir / "models/wesad/baseline"))
    results_dir = os.environ.get('RESULTS_DIR', str(base_dir / "results/wesad/baseline"))

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = load_processed_wesad_temporal(data_dir)
    if X_train.shape[1] < X_train.shape[2]:
        pass
    else:
        X_train, X_val, X_test = [np.transpose(x, (0, 2, 1)) for x in (X_train, X_val, X_test)]

    print(f"\nDataset info: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, Classes={info['class_names']}")

    # Use class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader, val_loader, test_loader = build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                                                              batch_size=64)

    # Device is already configured at module level - just print info
    print_device_info()
    model = SimpleLSTMWESAD(input_channels=X_train.shape[1], num_classes=len(info['class_names'])).to(device)

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

    # Move class weights to device
    class_weights = class_weights.to(device)

    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Adam optimizer with L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-6
    )

    # Training configuration
    clip_grad_norm = 1.0

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_acc, epochs_no_improve = -1.0, 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    start_time = time.time()
    epochs, patience = 100, 8

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {device}...")

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
            torch.save(model.state_dict(), os.path.join(models_dir, 'model_wesad_baseline.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    training_time = time.time() - start_time
    model.load_state_dict(torch.load(os.path.join(models_dir, 'model_wesad_baseline.pth'), map_location=device))
    results = evaluate_full_metrics(model, test_loader, device, info['class_names'])

    # Save results
    results['training_time'] = training_time
    results['epochs_trained'] = epoch
    results['best_val_acc'] = best_val_acc
    results['history'] = history

    with open(os.path.join(models_dir, 'results_wesad_baseline.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print final results
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
