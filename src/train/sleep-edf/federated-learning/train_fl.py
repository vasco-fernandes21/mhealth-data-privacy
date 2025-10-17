#!/usr/bin/env python3
"""
Federated Learning training for Sleep-EDF dataset using Flower framework.

Uses the SAME architecture as baseline (SimpleLSTM) to ensure fair comparison.
The only difference is the federated training strategy.

Environment Variables (for multiple runs):
- TRAIN_SEED: Random seed (default: 42)
- NUM_CLIENTS: Number of FL clients (default: 5)
- NUM_ROUNDS: Number of FL rounds (default: 10)
- MODEL_DIR: Directory to save model (default: models/sleep-edf/fl)
- RESULTS_DIR: Directory to save results (default: results/sleep-edf/fl)
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import flwr as fl
from flwr.common import Metrics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from device_utils import get_optimal_device, print_device_info
from preprocessing.sleep_edf import load_processed_sleep_edf

# Fix random seeds
SEED = int(os.environ.get('TRAIN_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- SAME Model as Baseline ---
class SimpleLSTM(nn.Module):
    """EXACT same architecture as baseline for fair comparison"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# --- Data preparation ---
def create_windows(X, y, window_size=10):
    """Create sliding windows (same as baseline)"""
    n_samples, n_features = X.shape
    n_windows = n_samples - window_size + 1
    
    X_windows = np.zeros((n_windows, window_size, n_features), dtype=X.dtype)
    y_windows = np.zeros(n_windows, dtype=y.dtype)
    
    for i in range(n_windows):
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size-1]
    
    return X_windows, y_windows

def partition_data(X_train, y_train, num_clients):
    """Partition data across clients (IID split)"""
    dataset_size = len(X_train)
    client_size = dataset_size // num_clients
    
    # Shuffle indices
    indices = np.random.permutation(dataset_size)
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else dataset_size
        client_indices = indices[start_idx:end_idx]
        
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        client_datasets.append((X_client, y_client))
    
    return client_datasets

# --- Flower Client ---
class SleepEDFClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        # Train for 1 epoch per round
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss += self.criterion(outputs, y_batch).item() * X_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)
        
        return float(loss / total), total, {"accuracy": float(correct / total)}

# --- Flower strategy with metrics aggregation ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    print("="*70)
    print(f"FEDERATED LEARNING - SLEEP-EDF (SEED={SEED})")
    print("="*70)
    
    # Configuration
    num_clients = int(os.environ.get('NUM_CLIENTS', 5))
    num_rounds = int(os.environ.get('NUM_ROUNDS', 10))
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/sleep-edf")
    models_output_dir = os.environ.get('MODEL_DIR', str(base_dir / f"models/sleep-edf/fl/fl_clients{num_clients}"))
    results_output_dir = os.environ.get('RESULTS_DIR', str(base_dir / f"results/sleep-edf/fl/fl_clients{num_clients}"))
    
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading processed Sleep-EDF data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = load_processed_sleep_edf(data_dir)
    
    print(f"\nFL Configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Dataset: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Create windows
    print("\nCreating windows...")
    window_size = 10
    X_train_win, y_train_win = create_windows(X_train, y_train, window_size)
    X_val_win, y_val_win = create_windows(X_val, y_val, window_size)
    X_test_win, y_test_win = create_windows(X_test, y_test, window_size)
    
    # Partition training data across clients
    print(f"Partitioning data across {num_clients} clients...")
    client_datasets = partition_data(X_train_win, y_train_win, num_clients)
    
    # Device
    device = get_optimal_device()
    print_device_info()
    
    # Model configuration (SAME as baseline)
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    num_classes = len(info['class_names'])
    
    # Create client functions
    def client_fn(cid: str):
        """Create a Flower client"""
        # Get client data
        X_client, y_client = client_datasets[int(cid)]
        
        # Create dataloaders
        X_tensor = torch.tensor(X_client, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Use centralized val set for all clients
        X_val_tensor = torch.tensor(X_val_win, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_win, dtype=torch.long)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create model
        model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        
        return SleepEDFClient(model, train_loader, val_loader, device)
    
    # FL Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # All clients participate
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start FL simulation
    print(f"\nStarting Federated Learning with {num_clients} clients for {num_rounds} rounds...")
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    training_time = time.time() - start_time
    
    # Get final global model
    print("\nFL training complete. Evaluating final model...")
    final_model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Evaluate on test set
    X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_win, dtype=torch.long).to(device)
    
    final_model.eval()
    with torch.no_grad():
        outputs = final_model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    y_true = y_test_win
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE!")
    print("="*70)
    print(f"Training time: {training_time:.2f}s")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test F1-Score: {f1:.4f}")
    print(f"Final Test Precision: {precision:.4f}")
    print(f"Final Test Recall: {recall:.4f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': info['class_names'],
        'num_clients': num_clients,
        'rounds': num_rounds,
        'training_time': float(training_time),
        'seed': SEED,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(models_output_dir, 'results_sleep_edf_fl.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save model
    model_file = os.path.join(models_output_dir, 'model_sleep_edf_fl.pth')
    torch.save(final_model.state_dict(), model_file)
    print(f"Model saved to: {model_file}")
    
    return results

if __name__ == "__main__":
    sys.exit(0 if main() else 1)

