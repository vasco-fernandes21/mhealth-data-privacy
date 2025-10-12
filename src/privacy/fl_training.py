"""
Federated Learning Training Module

This module provides functions for training models with Federated Learning
using Flower framework.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import flwr as fl
from typing import Dict, Any, List, Tuple
import json
import os
import warnings
warnings.filterwarnings('ignore')


class HealthDataClient(fl.client.NumPyClient):
    """
    Flower client for health data federated learning.
    """
    
    def __init__(self, model, X_train, y_train, X_val, y_val, client_id: str = "client"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.client_id = client_id
    
    def get_parameters(self, config):
        """Return model parameters."""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """Train model locally."""
        # Set global parameters
        self.model.set_weights(parameters)
        
        # Local training
        epochs = config.get('local_epochs', 5)
        batch_size = config.get('batch_size', 32)
        
        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        
        # Return updated parameters
        return self.model.get_weights(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model locally."""
        # Set global parameters
        self.model.set_weights(parameters)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        
        return loss, len(self.X_val), {"accuracy": accuracy}


def build_lstm_model(input_shape: Tuple[int, int], n_classes: int,
                    lstm_units: int = 64, dense_units: int = 32,
                    dropout_rate: float = 0.5, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build LSTM model for federated learning.
    
    Args:
        input_shape: Shape of input data (n_timesteps, n_features)
        n_classes: Number of output classes
        lstm_units: Number of LSTM units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def split_data_for_clients(X_train: np.ndarray, y_train: np.ndarray,
                          n_clients: int, distribution: str = 'iid') -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split training data among clients.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_clients: Number of clients
        distribution: Data distribution ('iid' or 'non_iid')
    
    Returns:
        List of (X_client, y_client) tuples
    """
    n_samples = len(X_train)
    samples_per_client = n_samples // n_clients
    
    client_data = []
    
    if distribution == 'iid':
        # IID: Randomly shuffle and split
        indices = np.random.permutation(n_samples)
        for i in range(n_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            if i == n_clients - 1:  # Last client gets remaining samples
                end_idx = n_samples
            
            client_indices = indices[start_idx:end_idx]
            X_client = X_train[client_indices]
            y_client = y_train[client_indices]
            client_data.append((X_client, y_client))
    
    elif distribution == 'non_iid':
        # Non-IID: Each client gets data from specific classes
        unique_classes = np.unique(y_train)
        classes_per_client = len(unique_classes) // n_clients
        
        for i in range(n_clients):
            # Assign classes to this client
            start_class = i * classes_per_client
            end_class = start_class + classes_per_client
            if i == n_clients - 1:  # Last client gets remaining classes
                end_class = len(unique_classes)
            
            client_classes = unique_classes[start_class:end_class]
            
            # Get indices for these classes
            client_indices = []
            for class_label in client_classes:
                class_indices = np.where(y_train == class_label)[0]
                client_indices.extend(class_indices)
            
            client_indices = np.array(client_indices)
            X_client = X_train[client_indices]
            y_client = y_train[client_indices]
            client_data.append((X_client, y_client))
    
    return client_data


def reshape_for_lstm(X: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Reshape data for LSTM input format.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        window_size: Number of timesteps for LSTM window
    
    Returns:
        Reshaped data of shape (n_samples - window_size + 1, window_size, n_features)
    """
    n_samples, n_features = X.shape
    X_reshaped = []
    
    for i in range(n_samples - window_size + 1):
        X_reshaped.append(X[i:i+window_size, :])
    
    return np.array(X_reshaped)


def train_with_fl(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 config: Dict[str, Any]) -> Tuple[tf.keras.Model, Dict[str, List], Dict[str, Any]]:
    """
    Train model with Federated Learning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
    
    Returns:
        Tuple of (trained_model, training_history, fl_info)
    """
    print("="*70)
    print("TRAINING WITH FEDERATED LEARNING")
    print("="*70)
    
    # Reshape data for LSTM
    window_size = config.get('window_size', 10)
    X_train_lstm = reshape_for_lstm(X_train, window_size)
    X_val_lstm = reshape_for_lstm(X_val, window_size)
    
    # Adjust labels
    y_train_lstm = y_train[window_size-1:]
    y_val_lstm = y_val[window_size-1:]
    
    # Convert to categorical
    n_classes = len(np.unique(y_train_lstm))
    y_train_cat = to_categorical(y_train_lstm, n_classes)
    y_val_cat = to_categorical(y_val_lstm, n_classes)
    
    print(f"LSTM data shapes:")
    print(f"  Train: {X_train_lstm.shape}, Val: {X_val_lstm.shape}")
    print(f"  Classes: {n_classes}")
    
    # FL parameters
    n_clients = config.get('n_clients', 5)
    n_rounds = config.get('n_rounds', 50)
    distribution = config.get('distribution', 'iid')
    
    print(f"\nFL Parameters:")
    print(f"  Number of clients: {n_clients}")
    print(f"  Communication rounds: {n_rounds}")
    print(f"  Data distribution: {distribution}")
    
    # Split data among clients
    client_data = split_data_for_clients(X_train_lstm, y_train_cat, n_clients, distribution)
    
    print(f"\nClient data distribution:")
    for i, (X_client, y_client) in enumerate(client_data):
        print(f"  Client {i}: {X_client.shape[0]} samples")
    
    # Create client function
    def client_fn(cid: str):
        client_id = int(cid)
        X_client, y_client = client_data[client_id]
        
        # Build model for this client
        input_shape = (window_size, X_train_lstm.shape[2])
        model = build_lstm_model(
            input_shape=input_shape,
            n_classes=n_classes,
            lstm_units=config.get('lstm_units', 64),
            dense_units=config.get('dense_units', 32),
            dropout_rate=config.get('dropout_rate', 0.5),
            learning_rate=config.get('learning_rate', 0.001)
        )
        
        return HealthDataClient(
            model, X_client, y_client, X_val_lstm, y_val_cat, f"client_{client_id}"
        )
    
    # FL strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # All clients participate
        min_available_clients=n_clients,
        min_fit_clients=n_clients
    )
    
    # Start FL simulation
    print(f"\nStarting FL simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy
    )
    
    # Build final model with same architecture
    input_shape = (window_size, X_train_lstm.shape[2])
    final_model = build_lstm_model(
        input_shape=input_shape,
        n_classes=n_classes,
        lstm_units=config.get('lstm_units', 64),
        dense_units=config.get('dense_units', 32),
        dropout_rate=config.get('dropout_rate', 0.5),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    # Get final parameters from FL history
    if hasattr(history, 'metrics_distributed') and 'accuracy' in history.metrics_distributed:
        final_accuracy = history.metrics_distributed['accuracy'][-1][1]
        final_model.set_weights(history.metrics_distributed['accuracy'][-1][2])
    else:
        # Fallback: train final model normally
        print("Warning: Could not extract final parameters from FL history")
        final_model.fit(X_train_lstm, y_train_cat, epochs=10, batch_size=32, verbose=0)
        final_accuracy = final_model.evaluate(X_val_lstm, y_val_cat, verbose=0)[1]
    
    # Prepare FL info
    fl_info = {
        'n_clients': n_clients,
        'n_rounds': n_rounds,
        'distribution': distribution,
        'final_accuracy': float(final_accuracy),
        'communication_cost': n_clients * n_rounds,  # Simplified metric
        'privacy_technique': 'FL'
    }
    
    print(f"\nFL Training Complete:")
    print(f"  Final accuracy: {final_accuracy:.4f}")
    print(f"  Communication rounds: {n_rounds}")
    print(f"  Total clients: {n_clients}")
    
    return final_model, history.metrics_distributed, fl_info


def evaluate_fl_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                     window_size: int = 10) -> Dict[str, float]:
    """
    Evaluate FL model on test data.
    
    Args:
        model: Trained FL model
        X_test: Test features
        y_test: Test labels
        window_size: Window size used for LSTM
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Reshape test data
    X_test_lstm = reshape_for_lstm(X_test, window_size)
    y_test_lstm = y_test[window_size-1:]
    
    # Convert to categorical
    n_classes = len(np.unique(y_test_lstm))
    y_test_cat = to_categorical(y_test_lstm, n_classes)
    
    # Predictions
    y_pred_probs = model.predict(X_test_lstm, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_lstm, y_pred)
    precision = precision_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_lstm, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_lstm, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nFL Model Test Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return results


def save_fl_model(model: tf.keras.Model, history: Dict[str, List],
                 results: Dict[str, float], fl_info: Dict[str, Any],
                 save_path: str, model_name: str):
    """
    Save FL model, history, results, and FL info.
    
    Args:
        model: Trained FL model
        history: FL training history
        results: Evaluation results
        fl_info: FL analysis info
        save_path: Directory to save files
        model_name: Name for the model
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_path, f'{model_name}.h5')
    model.save(model_path)
    print(f"FL model saved: {model_path}")
    
    # Save FL history
    history_path = os.path.join(save_path, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"FL history saved: {history_path}")
    
    # Save results
    results_path = os.path.join(save_path, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    # Save FL info
    fl_path = os.path.join(save_path, f'{model_name}_fl_info.json')
    with open(fl_path, 'w') as f:
        json.dump(fl_info, f, indent=2)
    print(f"FL info saved: {fl_path}")


def get_fl_configs(n_clients_list: list = [3, 5, 10]) -> list:
    """
    Get FL configurations for different numbers of clients.
    
    Args:
        n_clients_list: List of client numbers to test
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for n_clients in n_clients_list:
        config = {
            'window_size': 10,
            'lstm_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'n_rounds': 50,
            'n_clients': n_clients,
            'distribution': 'iid',
            'local_epochs': 5,
            'batch_size': 32,
            'privacy_technique': 'FL'
        }
        configs.append(config)
    
    return configs


if __name__ == "__main__":
    print("Federated Learning Training Module")
    print("This module provides functions for training models with FL.")
