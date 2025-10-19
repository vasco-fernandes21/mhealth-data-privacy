#!/usr/bin/env python3
"""
Federated Learning MELHORADO para Sleep-EDF com:
- Particionamento non-IID configurável
- Modelo mais robusto (similar ao WESAD)
- Metrics detalhadas por cliente
- Suporte para múltiplos experimentos
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
from preprocessing.sleep_edf import load_windowed_sleep_edf

# --- Seed ---
SEED = int(os.environ.get('TRAIN_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- MODELO MELHORADO (similar ao WESAD) ---
class ImprovedLSTMSleepEDF(nn.Module):
    """Arquitetura mais robusta com normalização e dropout"""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 128)
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
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        x = self.input_drop(x)
        
        lstm_out, (hn, cn) = self.lstm(x)
        # Concatenar últimos estados das direções forward e backward
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


# --- PARTICIONAMENTO NON-IID ---
def partition_data_non_iid_dirichlet(X_train, y_train, num_clients, alpha=0.5, seed=42):
    """Dirichlet distribution para non-IID realistic"""
    np.random.seed(seed)
    from collections import defaultdict
    
    num_classes = len(np.unique(y_train))
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_train):
        class_indices[label].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        
        start = 0
        for client_id, prop in enumerate(proportions):
            end = start + int(prop * len(indices))
            client_indices[client_id].extend(indices[start:end])
            start = end
    
    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        client_datasets.append((X_train[indices], y_train[indices]))
    
    return client_datasets


def partition_data_iid(X_train, y_train, num_clients, seed=42):
    """IID baseline"""
    np.random.seed(seed)
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
class SleepEDFClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, client_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "client_id": self.client_id
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        correct, total, loss = 0, 0, 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                outputs = self.model(X_batch)
                loss += self.criterion(outputs, y_batch).item() * X_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)
        
        return float(loss / total), total, {
            "accuracy": float(correct / total),
            "client_id": self.client_id
        }


# --- Aggregation ---
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_server_params = None
        self.round_metrics = []
    
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            self.final_server_params = aggregated_parameters
        
        # Log per-client metrics
        client_metrics = {}
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                cid = fit_res.metrics.get('client_id', 'unknown')
                client_metrics[f"client_{cid}"] = {
                    "loss": fit_res.metrics.get('loss'),
                    "accuracy": fit_res.metrics.get('accuracy')
                }
        
        self.round_metrics.append({
            "round": rnd,
            "clients": client_metrics,
            "aggregated": aggregated_metrics
        })
        
        return aggregated_parameters, aggregated_metrics


# --- Main ---
def main():
    num_clients = int(os.environ.get('NUM_CLIENTS', 5))
    num_rounds = int(os.environ.get('NUM_ROUNDS', 15))
    partition_type = os.environ.get('PARTITION_TYPE', 'non_iid_dir_0.5')
    
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = str(base_dir / "data/processed/sleep-edf")
    models_output_dir = os.environ.get('MODEL_DIR', 
                                       str(base_dir / f"models/sleep-edf/fl/fl_clients{num_clients}"))
    results_output_dir = os.environ.get('RESULTS_DIR', 
                                        str(base_dir / f"results/sleep-edf/fl/fl_clients{num_clients}"))
    
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = load_windowed_sleep_edf(data_dir)
    
    # Partition data
    if partition_type.startswith('non_iid_dir'):
        alpha = float(partition_type.split('_')[-1])
        client_datasets = partition_data_non_iid_dirichlet(X_train, y_train, num_clients, alpha=alpha, seed=SEED)
        print(f"Usando particionamento Non-IID Dirichlet (α={alpha})")
    else:
        client_datasets = partition_data_iid(X_train, y_train, num_clients, seed=SEED)
        print("Usando particionamento IID")
    
    device = get_optimal_device()
    print_device_info()
    
    input_size = X_train.shape[2]
    num_classes = len(info['class_names'])
    
    def client_fn(cid: str):
        X_client, y_client = client_datasets[int(cid)]
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_client, dtype=torch.float32),
                         torch.tensor(y_client, dtype=torch.long)),
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
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
        
        model = ImprovedLSTMSleepEDF(input_size, num_classes).to(device)
        return SleepEDFClient(model, train_loader, val_loader, device, int(cid))
    
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
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    training_time = time.time() - start_time
    
    # Final evaluation
    final_model = ImprovedLSTMSleepEDF(input_size, num_classes).to(device)
    if strategy.final_server_params is not None:
        param_names = list(final_model.state_dict().keys())
        server_state_dict = {}
        server_tensors = strategy.final_server_params.tensors
        for name, param_bytes in zip(param_names, server_tensors):
            param_numpy = np.frombuffer(param_bytes, dtype=np.float32)
            param_shape = final_model.state_dict()[name].shape
            server_state_dict[name] = torch.tensor(param_numpy.reshape(param_shape))
        final_model.load_state_dict(server_state_dict)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
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
        'class_names': info['class_names'],
        'num_clients': num_clients,
        'rounds': num_rounds,
        'partition_type': partition_type,
        'training_time': float(training_time),
        'seed': SEED,
        'round_metrics': strategy.round_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(models_output_dir, 'results_sleep_edf_fl.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(final_model.state_dict(), os.path.join(models_output_dir, 'model_sleep_edf_fl.pth'))
    
    print("\n" + "="*60)
    print("RESULTADOS FINAIS:")
    print("="*60)
    for key, value in results.items():
        if key not in ['confusion_matrix', 'round_metrics']:
            print(f"{key}: {value}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)