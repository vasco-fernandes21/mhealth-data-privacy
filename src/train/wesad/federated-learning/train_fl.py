#!/usr/bin/env python3
"""
Federated Learning training for WESAD dataset using Flower framework.

Uses the SAME architecture as baseline (SimpleLSTMWESAD) to ensure fair comparison.
The only difference is the federated training strategy.

Environment Variables (for multiple runs):
- TRAIN_SEED: Random seed (default: 42)
- NUM_CLIENTS: Number of FL clients (default: 5)
- NUM_ROUNDS: Number of FL rounds (default: 10)
- MODEL_DIR: Directory to save model (default: models/wesad/fl)
- RESULTS_DIR: Directory to save results (default: results/wesad/fl)
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import flwr as fl
from flwr.common import Metrics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from device_utils import get_optimal_device, print_device_info
from preprocessing.wesad import load_processed_wesad_temporal

# Fix random seeds
SEED = int(os.environ.get('TRAIN_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- SAME Model as Baseline ---
class SimpleLSTMWESAD(nn.Module):
    """EXACT same architecture as baseline for fair comparison"""
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        # Initial projection to reduce dimensionality
        self.input_proj = nn.Linear(input_channels, 128)
        self.input_norm = nn.GroupNorm(num_groups=8, num_channels=128)
        self.input_drop = nn.Dropout(0.2)

        # LSTM for temporal modeling
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

# --- Data partitioning ---
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
class WESADClient(fl.client.NumPyClient):
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
            
            # Gradient clipping (same as baseline)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
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
    print(f"FEDERATED LEARNING - WESAD (SEED={SEED})")
    print("="*70)
    
    # Configuration
    num_clients = int(os.environ.get('NUM_CLIENTS', 5))
    num_rounds = int(os.environ.get('NUM_ROUNDS', 10))
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_output_dir = os.environ.get('MODEL_DIR', str(base_dir / f"models/wesad/fl/fl_clients{num_clients}"))
    results_output_dir = os.environ.get('RESULTS_DIR', str(base_dir / f"results/wesad/fl/fl_clients{num_clients}"))
    
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading processed WESAD data...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_processed_wesad_temporal(data_dir)
    
    print(f"\nFL Configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Dataset: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"  Classes: {label_encoder.classes_}")
    
    # Partition training data across clients
    print(f"\nPartitioning data across {num_clients} clients...")
    client_datasets = partition_data(X_train, y_train, num_clients)
    
    # Device
    device = get_optimal_device()
    print_device_info()
    
    # Model configuration (SAME as baseline)
    input_channels = X_train.shape[1]  # 14 channels
    num_classes = len(label_encoder.classes_)  # 2 (non-stress, stress)
    
    # Create client functions
    def client_fn(cid: str):
        """Create a Flower client"""
        # Get client data
        X_client, y_client = client_datasets[int(cid)]
        
        # Create dataloaders
        X_tensor = torch.tensor(X_client, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Use centralized val set for all clients
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = SimpleLSTMWESAD(input_channels, num_classes).to(device)
        
        return WESADClient(model, train_loader, val_loader, device)
    
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
    final_model = SimpleLSTMWESAD(input_channels, num_classes).to(device)
    
    # Evaluate on test set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    final_model.eval()
    with torch.no_grad():
        outputs = final_model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    y_true = y_test
    
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
        'class_names': label_encoder.classes_.tolist(),
        'num_clients': num_clients,
        'rounds': num_rounds,
        'training_time': float(training_time),
        'seed': SEED,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(models_output_dir, 'results_wesad_fl.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save model
    model_file = os.path.join(models_output_dir, 'model_wesad_fl.pth')
    torch.save(final_model.state_dict(), model_file)
    print(f"Model saved to: {model_file}")
    
    return results

if __name__ == "__main__":
    sys.exit(0 if main() else 1)

