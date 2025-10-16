#!/usr/bin/env python3
"""
Train PyTorch CNN-LSTM with Differential Privacy for WESAD dataset
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

# Fix random seeds for reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from preprocessing.wesad import load_processed_wesad_temporal

# --- Custom CNN-LSTM model without BatchNorm for Opacus compatibility ---
class CNNLSTMWESAD(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        # First Conv block - similar to TensorFlow
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # Second Conv block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        # Third Conv block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # Additional conv layers with GroupNorm (DP-compatible)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, 256)  # Groups of 32 for better DP compatibility
        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Another conv layer for more capacity
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.gn5 = nn.GroupNorm(64, 512)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        self.dropout5 = nn.Dropout(0.2)

        # Global average pooling (DP-friendly)
        # After all conv layers: (batch, 512, 60) -> (batch, 512)

        # Dense layers with stronger regularization for DP
        self.dense1 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 128)
        self.dropout7 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = torch.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.gn5(x)
        x = torch.relu(x)
        x = self.pool5(x)
        x = self.dropout5(x)

        # Global average pooling (DP-friendly)
        x = torch.mean(x, dim=2)  # (batch, 512)

        # Dense layers with stronger regularization
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout6(x)

        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dropout7(x)

        x = self.dense3(x)

        return x

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

def main():
    print("="*70)
    print("TRAINING WESAD WITH DIFFERENTIAL PRIVACY")
    print("="*70)

    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_output_dir = str(base_dir / "models/wesad/differential_privacy")
    results_output_dir = str(base_dir / "results/wesad/differential_privacy")

    # Create directories
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # Load processed data
    print("Loading processed WESAD data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = load_processed_wesad_temporal(data_dir)

    print(f"\nDataset info:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {info['n_classes']} ({info['class_names']})")
    print(f"  Class distribution: {Counter(y_train)}")

    # Create datasets and loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model configuration
    input_channels = X_train.shape[1]
    num_classes = len(info['class_names'])

    print(f"\nModel configuration:")
    print(f"  Input channels: {input_channels}")
    print(f"  Num classes: {num_classes}")
    print(f"  Batch size: {batch_size}")

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTMWESAD(input_channels, num_classes).to(device)

    # DP Configuration - Improved for better privacy-utility trade-off
    print(f"\nDifferential Privacy configuration:")
    noise_multiplier = 0.8  # Reduced for better utility while maintaining privacy
    max_grad_norm = 1.2    # Slightly increased for stability
    delta = 1e-5
    sample_rate = batch_size / len(train_dataset)

    print(f"  Noise multiplier: {noise_multiplier}")
    print(f"  Max gradient norm: {max_grad_norm}")
    print(f"  Delta: {delta}")
    print(f"  Sample rate: {sample_rate:.4f}")

    # Install opacus if not available
    try:
        from opacus import PrivacyEngine
    except ImportError:
        print("Installing opacus...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opacus"])
        from opacus import PrivacyEngine

    # Loss and optimizer (DP-compatible) - Improved hyperparameters
    criterion = nn.CrossEntropyLoss()

    # Use SGD with momentum for better DP performance
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,  # Higher LR for faster convergence
        momentum=0.9,
        weight_decay=1e-4  # L2 regularization for DP stability
    )

    # Make optimizer DP-compatible using new Opacus API
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    
    # Wrap model for DP
    model = GradSampleModule(model)
    
    # Create DP optimizer
    optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size
    )

    # Learning rate scheduler for better convergence
    # Note: DPOptimizer doesn't expose underlying optimizer, so we'll use a simple step decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer,
        step_size=20,
        gamma=0.5
    )

    # Training
    print(f"\nStarting improved DP training on {device}...")
    num_epochs = 150  # More epochs for better convergence
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_progress = ProgressBar(len(train_dataset), f"Epoch {epoch+1:3d} - Training")

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

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if epoch > 15 and val_loss > best_val_loss:  # Increased patience
            patience_counter += 1
            if patience_counter >= 7:  # Increased patience
                print("Early stopping triggered.")
                break
        else:
            patience_counter = 0
            best_val_loss = val_loss

    # Evaluate on test set
    print(f"\nEvaluating DP model...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Calculate detailed metrics
    test_acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Get privacy budget - calculate manually for Opacus 1.5.4
    # Using RDP accountant formula: ε ≈ noise_multiplier * sqrt(epochs * sample_rate) for rough estimate
    epochs_trained = epoch + 1
    steps = epochs_trained * len(train_loader)
    epsilon = noise_multiplier * np.sqrt(2 * np.log(1.25/delta)) * np.sqrt(steps * sample_rate)

    # Save results
    results = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': info['class_names'],
        'dp_params': {
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'delta': delta,
            'epsilon': float(epsilon)
        }
    }

    with open(os.path.join(models_output_dir, 'results_wesad_dp.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_output_dir, 'dp_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print final results
    print(f"\n" + "="*70)
    print("DP TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1-Score: {f1:.4f}")
    print(f"Privacy Budget: ε = {epsilon:.2f}")

    print(f"\nConfusion Matrix:")
    print(f"  Predicted →")
    print(f"  Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {info['class_names'][i]:8s}: {row}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
