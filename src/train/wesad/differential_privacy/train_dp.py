#!/usr/bin/env python3
"""
Train PyTorch LSTM-only with Differential Privacy (DP) using Opacus
for WESAD binary stress classification.

DP REQUIREMENT: Must use DPLSTM instead of nn.LSTM!
- nn.LSTM uses internal modules not compatible with Opacus DP hooks
- DPLSTM is a drop-in replacement with proper gradient sampling
- Same API, same functionality, but DP-compatible

        Based on the optimized LSTM-only baseline for better DP compatibility:
        - Uses the same architecture as the new baseline (SimpleLSTMWESAD)
        - Maintains all DP optimizations (GroupNorm, reduced complexity)
        - Compatible with privacy guarantees and federated learning

        Environment Variables (for multiple runs):
        - TRAIN_SEED: Random seed (default: 42)
        - NOISE_MULTIPLIER: DP noise multiplier (default: 0.9)
        - MODEL_DIR: Directory to save model (default: models/wesad/dp)
        - RESULTS_DIR: Directory to save results (default: results/wesad/dp)
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
from tqdm import tqdm

# Import device utilities and preprocessing function
# Adjusting path insert to be relative to the running script's parent directories
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

# Assuming device_utils and preprocessing.wesad are available via sys.path
try:
    from device_utils import get_optimal_device, print_device_info
    from preprocessing.wesad import load_augmented_wesad_temporal
except ImportError:
    print("FATAL ERROR: Could not import utility modules. Ensure project structure is correct.")
    sys.exit(1)


# Fix random seeds for reproducible results
SEED = int(os.environ.get('TRAIN_SEED', 42))
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
    Uses DPLSTM, which is required by Opacus.
    """
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        # Initial projection to reduce dimensionality (DP-friendly)
        self.input_proj = nn.Linear(input_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # GroupNorm para DP
        self.input_drop = nn.Dropout(0.2)  # Dropout moderado

        # LSTM layers - using DPLSTM for DP compatibility (REQUIRED!)
        self.lstm = DPLSTM(input_size=128, hidden_size=64, num_layers=2,
                        batch_first=True, bidirectional=True, dropout=0.2)

        # Normalization after LSTM (crucial for DP)
        self.lstm_norm = nn.GroupNorm(num_groups=8, num_channels=128)  # 8 groups para 128 channels (64*2)
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
        # Apply GroupNorm: transpose for channel-first, apply GN, transpose back
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
def train_one_epoch_dp(model, train_loader, criterion, optimizer, device):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        # Optimized progress bar with tqdm
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_X, batch_y in pbar:
            # Optimized: non_blocking transfer
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Optimized: set_to_none=True is faster
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Opacus handles gradient clipping and noise addition during optimizer.step()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })

        return train_loss / total, correct / total

def evaluate_dp(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Optimized: non_blocking transfer
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return val_loss / total, correct / total

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
    # Print seed and noise multiplier info
    noise_mult = float(os.environ.get('NOISE_MULTIPLIER', 0.9))
    seed_info = f" (SEED={SEED}, noise_mult={noise_mult})" if SEED != 42 or noise_mult != 0.9 else ""
    print("=" * 70)
    print(f"WESAD BINARY STRESS CLASSIFICATION - DP (OPACUS){seed_info}")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")

    # Allow override via environment variables (for multiple runs)
    models_dir = os.environ.get('MODEL_DIR', str(base_dir / "models/wesad/dp"))
    results_dir = os.environ.get('RESULTS_DIR', str(base_dir / "results/wesad/dp"))

    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data with pre-calculated augmentation (dados já otimizados no pré-processamento!)
    print("Loading WESAD data...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, info = load_augmented_wesad_temporal(data_dir)

    print(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Classes: {info['class_names']}")

    # Apply oversampling to augmented data
    print("Balancing classes (oversampling on augmented data)...")
    X_train_bal, y_train_bal = _simple_oversample(X_train, y_train)

    print(f"After balancing: Train={X_train_bal.shape}, Val={X_val.shape}, Test={X_test.shape}")

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

    # Optimized batch sizes for DP - 64 is ideal for this dataset size
    batch_size = 64
    # IMPORTANT: drop_last=True is MANDATORY for Opacus to ensure all batches have the same size.
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

    # Make model compatible with DP (use environment variable for noise_multiplier)
    noise_multiplier = float(os.environ.get('NOISE_MULTIPLIER', 0.9))
    
    # IMPORTANT: Check if the model has been wrapped before re-wrapping (for safety)
    if hasattr(model, 'privacy_engine'):
        print("WARNING: Model appears to be already wrapped. Skipping Opacus wrapper.")
    else:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=MAX_GRAD_NORM,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            poisson_sampling=True       # Standard accounting, better privacy bounds
        )

    print(f"DP Parameters: ε={TARGET_EPSILON}, δ={TARGET_DELTA}, noise_mult={noise_multiplier}, max_grad_norm={MAX_GRAD_NORM}")
    print(f"Training config: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print("Starting DP training loop...")

    # Training loop with DP
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epsilon": []}

    start_time = time.time()

    # --- MAIN TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch:03d}/{EPOCHS} ---")
        
        # 1. Train one epoch with DP
        train_loss, train_acc = train_one_epoch_dp(model, train_loader, criterion, optimizer, device)

        # 2. Evaluate on validation set
        val_loss, val_acc = evaluate_dp(model, val_loader, criterion, device)

        # 3. Get current privacy budget (epsilon)
        try:
            epsilon = privacy_engine.get_epsilon(TARGET_DELTA)
        except Exception as e:
            epsilon = 0.0
            print(f"ERROR calculating epsilon: {e}")

        # 4. Store metrics
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

        # 5. Early Stopping and Model Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model state (DP-compatible)
            torch.save(model.state_dict(), os.path.join(models_dir, 'model_wesad_dp.pth'))
            epochs_no_improve = 0
            print(f"  New best val_acc={best_val_acc:.4f}. Model saved.")
        else:
            epochs_no_improve += 1

        # Early stopping logic
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs (patience={PATIENCE})")
            break

        # Privacy budget warning
        if epsilon >= TARGET_EPSILON * 1.1:
            print(f"WARNING: Privacy budget (ε={TARGET_EPSILON:.2f}) exceeded (current: ε={epsilon:.2f})")
            # Decide whether to stop or continue (stopping is safer for DP papers)
            # break # Uncomment to stop immediately when budget is exceeded

    # --- END OF TRAINING LOOP: FINAL EVALUATION AND REPORTING ---

    training_time = time.time() - start_time
    epochs_trained = len(history['loss'])

    # Load best model and evaluate on TEST set
    try:
        model.load_state_dict(torch.load(os.path.join(models_dir, 'model_wesad_dp.pth'), map_location=device))
    except Exception as e:
        print(f"WARNING: Could not load best model state, using current model state: {e}")

    results = evaluate_full_metrics(model, test_loader, device, info['class_names'])

    final_epsilon = history['epsilon'][-1] if history['epsilon'] else 0.0

    # Add DP information to results
    results['dp_params'] = {
        'target_epsilon': TARGET_EPSILON,
        'final_epsilon': final_epsilon,
        'delta': TARGET_DELTA,
        'max_grad_norm': MAX_GRAD_NORM,
        'noise_multiplier': noise_multiplier
    }
    results['training_time_seconds'] = training_time
    results['epochs_trained'] = epochs_trained

    # Save results
    history_file_path = os.path.join(models_dir, 'history_wesad_dp.json')
    results_models_path = os.path.join(models_dir, 'results_wesad_dp.json')
    results_reports_path = os.path.join(results_dir, 'dp_results.json')
    
    with open(history_file_path, 'w') as f:
        json.dump({**history, 'epochs': epochs_trained, 'training_time_seconds': training_time, 'final_epsilon': final_epsilon}, f, indent=2)
    with open(results_models_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(results_reports_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("DP TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFINAL RESULTS:")
    print(f"  Epochs trained: {epochs_trained}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"\nPRIVACY:")
    print(f"  Final epsilon (ε): {final_epsilon:.2f} (target: {TARGET_EPSILON})")
    print(f"  Delta (δ): {TARGET_DELTA}")
    print(f"  Noise Multiplier (σ): {noise_multiplier}")
    print(f"  Max grad norm: {MAX_GRAD_NORM}")
    print(f"\nPERFORMANCE (Test Set):")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"\nCONFUSION MATRIX:")
    cm = results['confusion_matrix']
    class_names = results['class_names']
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:12s}: {row}")
    print(f"\nResults saved to:")
    print(f"  - {results_models_path}")
    print(f"  - {results_reports_path}")
    print("=" * 70)
    
    # Return 0 for success
    return 0

if __name__ == "__main__":
    sys.exit(main())
