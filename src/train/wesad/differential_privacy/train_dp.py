#!/usr/bin/env python3
"""
Train PyTorch LSTM-only with Differential Privacy (DP) using Opacus
for WESAD binary stress classification.

🔒 DP REQUIREMENT: Must use DPLSTM instead of nn.LSTM!
   - nn.LSTM uses internal modules not compatible with Opacus DP hooks
   - DPLSTM is a drop-in replacement with proper gradient sampling
   - Same API, same functionality, but DP-compatible

Based on the optimized LSTM-only baseline for better DP compatibility:
- Uses the same architecture as the new baseline (SimpleLSTMWESAD)
- Maintains all DP optimizations (GroupNorm, reduced complexity)
- Compatible with privacy guarantees and federated learning
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

from opacus import PrivacyEngine
from opacus.layers import DPLSTM

# ❌ NÃO FUNCIONA: nn.LSTM não é compatível com Opacus DP
# ✅ FUNCIONA: DPLSTM é drop-in replacement compatível com DP

# Import device utilities and preprocessing function
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
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
    torch.backends.cudnn.deterministic = False  # Relaxed for speed in DP training
    torch.backends.cudnn.benchmark = True
elif device.type == "mps":
    torch.mps.manual_seed(SEED)
    # MPS doesn't have deterministic/benchmark options like CUDA

# --- LSTM-only model otimizado para DP (MESMA ARQUITETURA DO BASELINE) ---
class SimpleLSTMWESAD(nn.Module):
    """
    LSTM-only architecture optimized for DP compatibility.
    Same architecture as the new baseline for fair comparison.
    """
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        # Initial projection to reduce dimensionality (DP-friendly)
        self.input_proj = nn.Linear(input_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # GroupNorm para DP
        self.input_drop = nn.Dropout(0.2)  # Dropout moderado

        # LSTM layers - using DPLSTM for DP compatibility (REQUIRED!)
        # ❌ nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        # ✅ DPLSTM funciona exatamente igual, mas com hooks DP
        self.lstm = DPLSTM(input_size=128, hidden_size=64, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)

        # Normalization after LSTM (crucial for DP)
        self.lstm_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # 8 groups para 128 channels
        self.lstm_drop = nn.Dropout(0.3)  # Dropout após LSTM

        # Dense layers (simplified for DP)
        self.fc1 = nn.Linear(128, 64)
        self.fc1_norm = nn.GroupNorm(num_groups=8, num_channels=64)
        self.fc1_drop = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Input processing (batch, channels, timesteps) -> (batch, timesteps, channels)
        x = x.permute(0, 2, 1)

        # Initial projection and normalization
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_drop(x)

        # LSTM processing
        lstm_out, (hn, cn) = self.lstm(x)

        # Concatenate final hidden states from both directions
        x = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, 128)

        # Normalization after LSTM
        x = self.lstm_norm(x.unsqueeze(2)).squeeze(2)
        x = self.lstm_drop(x)

        # Dense layers
        x = self.fc1(x)
        x = self.fc1_norm(x.unsqueeze(2)).squeeze(2)
        x = torch.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc2_drop(x)

        out = self.fc3(x)
        return out

# --- Data augmentation & oversampling (mantido igual do baseline) ---
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
                    # Shift right (pad left, truncate right)
                    X_aug[i] = np.pad(X_aug[i], ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
                else:
                    # Shift left (pad right, truncate left)
                    X_aug[i] = np.pad(X_aug[i], ((0, 0), (0, -shift)), mode='edge')[:, -shift:]

    return X_aug

# --- Training and evaluation functions (adaptado para DP) ---
def train_one_epoch_dp(model, train_loader, criterion, optimizer, device, privacy_engine):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    print(f"  🔄 Training on {len(train_loader)} batches...")
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Clip gradients for DP compatibility
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

        # Progress indicator every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)} processed")

    print(f"  ✅ Training epoch completed")
    return train_loss / len(train_loader), correct / total

def evaluate_dp(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    print(f"  🔍 Evaluating on {len(val_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"  ✅ Validation completed")
    return val_loss / len(val_loader), correct / total

def evaluate_full_metrics(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_names': class_names
    }

# --- Main training function ---
def main():
    print("=" * 70)
    print("WESAD BINARY STRESS CLASSIFICATION - DP (OPACUS)")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_dir = str(base_dir / "models/wesad/dp")
    results_dir = str(base_dir / "results/wesad/dp")

    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load and prepare data
    print("📊 Loading processed WESAD data...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = load_processed_wesad_temporal(data_dir)

    print(f"📈 Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"🏷️  Classes: {info['class_names']}")

    # Apply oversampling only (skip augmentation for speed)
    print("🔄 Balancing classes (oversampling only)...")
    X_train_bal, y_train_bal = _simple_oversample(X_train, y_train)

    print(f"✅ After augmentation: Train={X_train_bal.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_bal), y=y_train_bal)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_bal, dtype=torch.float32),
        torch.tensor(y_train_bal, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Optimized batch sizes for DP (try 64 if memory allows)
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
    )

    # Model initialization (device already configured at module level)
    print_device_info()
    model = SimpleLSTMWESAD(input_channels=X_train.shape[1], num_classes=len(info['class_names']))
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {device}...")

    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # DP parameters (balanced default)
    TARGET_EPSILON = 8.0      # Common target for meaningful privacy
    TARGET_DELTA = 1e-5       # Standard delta
    MAX_GRAD_NORM = 1.0       # Typical clipping norm
    EPOCHS = 25               # Reasonable cap; early stopping will cut sooner
    PATIENCE = 6              # Moderate patience

    # Optimizer (Adam with weight decay for DP)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Setup Privacy Engine
    privacy_engine = PrivacyEngine()

    # Make model compatible with DP
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=(1.0 if batch_size == 64 else 0.9),
        max_grad_norm=MAX_GRAD_NORM,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        poisson_sampling=True       # Standard accounting, better privacy bounds
    )

    print(f"🔒 DP Parameters: ε={TARGET_EPSILON}, δ={TARGET_DELTA}, noise_mult=0.9, max_grad_norm={MAX_GRAD_NORM}")
    print(f"📋 Training config: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print("🚀 Starting DP training loop...")

    # Training loop with DP
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epsilon": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n——— Epoch {epoch:03d}/{EPOCHS} ———")
        # Train one epoch with DP
        train_loss, train_acc = train_one_epoch_dp(model, train_loader, criterion, optimizer, device, privacy_engine)

        # Evaluate on validation set
        val_loss, val_acc = evaluate_dp(model, val_loader, criterion, device)

        # Get current privacy budget (epsilon)
        epsilon = privacy_engine.get_epsilon(TARGET_DELTA)

        # Store metrics
        history["loss"].append(float(train_loss))
        history["accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        history["epsilon"].append(epsilon)

        current_lr = optimizer.param_groups[0].get('lr', None)
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}: "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"ε={epsilon:.2f} | "
            + (f"lr={current_lr:.5f} | " if current_lr is not None else "")
            + f"time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model state (DP-compatible)
            torch.save(model.state_dict(), os.path.join(models_dir, 'model_wesad_dp.pth'))
            epochs_no_improve = 0
            print(f"  ⬆️  New best val_acc={best_val_acc:.4f}. Model saved.")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"⚠️  Early stopping triggered after {epoch} epochs (patience={PATIENCE})")
            break

        # Privacy budget warning
        if epsilon >= TARGET_EPSILON * 1.1:
             print(f"⚠️  WARNING: Privacy budget (ε={TARGET_EPSILON:.2f}) exceeded (current: ε={epsilon:.2f})")

    training_time = time.time() - start_time
    epochs_trained = len(history['loss'])

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(models_dir, 'model_wesad_dp.pth'), map_location=device))
    results = evaluate_full_metrics(model, test_loader, device, info['class_names'])

    final_epsilon = history['epsilon'][-1] if history['epsilon'] else 0.0

    # Add DP information to results
    results['dp_params'] = {
        'target_epsilon': TARGET_EPSILON,
        'final_epsilon': final_epsilon,
        'delta': TARGET_DELTA,
        'max_grad_norm': MAX_GRAD_NORM,
        'noise_multiplier': 0.9
    }
    results['training_time_seconds'] = training_time
    results['epochs_trained'] = epochs_trained

    # Save results
    with open(os.path.join(models_dir, 'history_wesad_dp.json'), 'w') as f:
        json.dump({**history, 'epochs': epochs_trained, 'training_time_seconds': training_time, 'final_epsilon': final_epsilon}, f, indent=2)
    with open(os.path.join(models_dir, 'results_wesad_dp.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(results_dir, 'dp_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ DP TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Epochs trained: {epochs_trained}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"\n🔒 PRIVACY:")
    print(f"  Final epsilon (ε): {final_epsilon:.2f} (target: {TARGET_EPSILON})")
    print(f"  Delta (δ): {TARGET_DELTA}")
    print(f"  Max grad norm: {MAX_GRAD_NORM}")
    print(f"\n🎯 PERFORMANCE (Test Set):")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"\n📈 CONFUSION MATRIX:")
    cm = results['confusion_matrix']
    class_names = results['class_names']
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:12s}: {row}")
    print(f"\n💾 Results saved to:")
    print(f"  - {models_dir}/results_wesad_dp.json")
    print(f"  - {results_dir}/dp_results.json")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
