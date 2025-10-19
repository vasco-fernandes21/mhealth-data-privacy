#!/usr/bin/env python3
"""
Federated Learning training for WESAD dataset using Flower framework.

Optimizations:
- Real server (no client proxy)
- Dynamic GPU allocation per client
- Batch size 64, learning rate 0.002
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import flwr as fl
from flwr.common import Metrics

# --- Paths and imports ---
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))
from device_utils import get_optimal_device, print_device_info
from preprocessing.wesad import load_augmented_wesad_temporal

# --- Seed ---
SEED = int(os.environ.get('TRAIN_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Model ---
class SimpleLSTMWESAD(nn.Module):
    """Baseline architecture"""
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, 128)
        self.input_norm = nn.GroupNorm(8, 128)
        self.input_drop = nn.Dropout(0.2)

        self.lstm = nn.LSTM(128, 64, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_norm = nn.GroupNorm(8, 128)
        self.lstm_drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 64)
        self.fc1_norm = nn.GroupNorm(8, 64)
        self.fc1_drop = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_drop(x)

        lstm_out, (hn, cn) = self.lstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=1)
        x = self.lstm_norm(x.unsqueeze(2)).squeeze(2)
        x = self.lstm_drop(x)

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
    dataset_size = len(X_train)
    client_size = dataset_size // num_clients
    indices = np.random.permutation(dataset_size)
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else dataset_size
        client_indices = indices[start_idx:end_idx]
        client_datasets.append((X_train[client_indices], y_train[client_indices]))
    return client_datasets

# --- Flower Client ---
class WESADClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss += self.criterion(outputs, y_batch).item() * X_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)
        return float(loss / total), total, {"accuracy": float(correct / total)}

# --- Aggregation ---
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_server_params = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.final_server_params = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

# --- Main ---
def main():
    num_clients = int(os.environ.get('NUM_CLIENTS', 3))
    num_rounds = int(os.environ.get('NUM_ROUNDS', 15))

    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/wesad")
    models_output_dir = os.environ.get('MODEL_DIR', str(base_dir / f"models/wesad/fl/fl_clients{num_clients}"))
    results_output_dir = os.environ.get('RESULTS_DIR', str(base_dir / f"results/wesad/fl/fl_clients{num_clients}"))

    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, _ = load_augmented_wesad_temporal(data_dir)
    client_datasets = partition_data(X_train, y_train, num_clients)

    device = get_optimal_device()
    print_device_info()

    input_channels = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    def client_fn(cid: str):
        X_client, y_client = client_datasets[int(cid)]
        # Optimized DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_client, dtype=torch.float32),
                         torch.tensor(y_client, dtype=torch.long)),
            batch_size=64, 
            shuffle=True,
            num_workers=2,           # Parallel loading
            pin_memory=True,         # Fast GPU transfer
            persistent_workers=True  # Reuse workers
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.long)),
            batch_size=64, 
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        model = SimpleLSTMWESAD(input_channels, num_classes).to(device)
        return WESADClient(model, train_loader, val_loader, device)

    gpu_per_client = 1.0 / num_clients if torch.cuda.is_available() else 0.0
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={"num_cpus": 1, "num_gpus": gpu_per_client},
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    training_time = time.time() - start_time

    # Final evaluation
    final_model = SimpleLSTMWESAD(input_channels, num_classes).to(device)
    if strategy.final_server_params is not None:
        param_names = list(final_model.state_dict().keys())
        server_state_dict = {}
        server_tensors = strategy.final_server_params.tensors
        for name, param_bytes in zip(param_names, server_tensors):
            param_numpy = np.frombuffer(param_bytes, dtype=np.float32)
            server_state_dict[name] = torch.tensor(param_numpy)
        final_model.load_state_dict(server_state_dict)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    final_model.eval()
    with torch.no_grad():
        y_pred = final_model(X_test_tensor).argmax(dim=1).cpu().numpy()
    y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

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
    model_file = os.path.join(models_output_dir, 'model_wesad_fl.pth')
    torch.save(final_model.state_dict(), model_file)

    for key, value in results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")

    return results

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
