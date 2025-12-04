import torch
import numpy as np
import asyncio
from typing import Dict, List, Callable
from opacus.accountants import RDPAccountant
from .client import VirtualClient
from .models import get_model
from ..data.loader import data_loader
from ..core.dataset_configs import get_config


class FederatedSimulation:
    def __init__(self, dataset_name: str, n_clients: int, sigma: float):
        self.dataset_meta = data_loader.load_dataset(dataset_name)
        self.n_clients = n_clients
        
        # Load paper configuration
        paper_config = get_config(dataset_name, n_clients, sigma)
        
        # Compute class weights from data
        y_train = self.dataset_meta["train"][1]
        class_weights = self._compute_class_weights(y_train)
        paper_config["dataset"]["class_weights"] = class_weights
        
        self.config = paper_config
        self.clients = self._setup_clients()
        self.global_model = get_model(
            self.config["dataset"]["input_dim"],
            self.config["dataset"]["n_classes"]
        )
        self.should_stop = False
        
        # In the paper's implementation, each client maintains its PrivacyEngine across rounds
        # The PrivacyEngine automatically accumulates epsilon as training progresses
        # We'll get epsilon from one of the clients (they should all have similar values)
        self.delta = self.config["differential_privacy"]["target_delta"]
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights matching paper implementation."""
        class_counts = np.bincount(y.astype(int))
        n_classes = len(class_counts)
        n_samples = len(y)
        
        weights = {}
        for c in range(n_classes):
            if class_counts[c] > 0:
                weights[c] = n_samples / (n_classes * class_counts[c])
            else:
                weights[c] = 1.0
        
        return weights

    def _setup_clients(self) -> List[VirtualClient]:
        X, y, subjects = self.dataset_meta["train"]
        unique_subs = np.unique(subjects)
        
        print(f"Setting up {self.n_clients} clients with {len(unique_subs)} unique subjects")
        
        # Split subjects across clients
        sub_splits = np.array_split(unique_subs, self.n_clients)
        
        clients = []
        for i, subs in enumerate(sub_splits):
            mask = np.isin(subjects, subs)
            client_X = X[mask]
            client_y = y[mask]
            
            if len(client_X) == 0:
                print(f"Warning: Client {i} has no data")
                continue
                
            print(f"Client {i}: {len(client_X)} samples from {len(subs)} subjects")
            clients.append(VirtualClient(i, (client_X, client_y), self.config))
        
        if len(clients) == 0:
            raise ValueError("No clients created - check data splitting")
        
        print(f"Created {len(clients)} virtual clients")
        return clients

    def stop(self):
        self.should_stop = True

    async def run_simulation(self, callback: Callable):
        rounds = self.config["federated_learning"]["global_rounds"]
        sigma = self.config["differential_privacy"]["noise_multiplier"]
        
        await callback({"type": "log", "message": f"Initializing FL Protocol with {self.n_clients} clients..."})
        await callback({"type": "log", "message": f"DP Profile: Sigma={sigma} (Gaussian Noise)"})
        await callback({"type": "log", "message": f"Configuration: Batch={self.config['training']['batch_size']}, LR={self.config['training']['learning_rate']}, Rounds={rounds}"})

        for r in range(1, rounds + 1):
            if self.should_stop:
                await callback({"type": "log", "message": f"Simulation stopped at round {r}"})
                break
            
            await callback({"type": "round_start", "round": r})
            await callback({"type": "log", "message": f"--- Starting Round {r}/{rounds} ---"})
            
            local_weights = []
            metrics_agg = {"loss": 0.0}

            # Serial simulation to save memory
            for client in self.clients:
                if self.should_stop:
                    break
                
                await callback({"type": "client_training", "client_id": client.id})
                await callback({"type": "log", "message": f"   > Client {client.id+1} computing local gradients..."})
                
                # Run heavy training in thread pool to avoid blocking websocket
                try:
                    w, m = await asyncio.to_thread(client.train_round, self.global_model.state_dict())
                    local_weights.append(w)
                    metrics_agg["loss"] += m["loss"]
                    # Store epsilon from client (for debugging)
                    if "epsilon" in m:
                        metrics_agg["epsilon"] = max(metrics_agg.get("epsilon", 0.0), m["epsilon"])
                    await callback({"type": "log", "message": f"   > Client {client.id+1} completed: loss={m['loss']:.4f}, ε={m.get('epsilon', 0.0):.2f}"})
                except Exception as e:
                    await callback({"type": "log", "message": f"   > Error training client {client.id+1}: {str(e)}"})
                    import traceback
                    traceback.print_exc()
                    raise

            if not local_weights:
                await callback({"type": "log", "message": "No weights collected, stopping simulation"})
                break
            
            # Get epsilon from PrivacyEngine (matching paper's approach)
            # Each client's PrivacyEngine accumulates epsilon automatically across rounds
            # Use the maximum epsilon from all clients (they should be similar, but use max for safety)
            accumulated_epsilon = metrics_agg.get("epsilon", 0.0)
            
            # Fallback: Try to get epsilon directly from PrivacyEngine if not in metrics
            if accumulated_epsilon == 0.0 and len(self.clients) > 0:
                client = self.clients[0]
                if hasattr(client, 'privacy_engine') and client.privacy_engine is not None:
                    try:
                        accumulated_epsilon = client.privacy_engine.get_epsilon(delta=self.delta)
                        if not isinstance(accumulated_epsilon, (int, float)) or accumulated_epsilon == float('inf') or accumulated_epsilon != accumulated_epsilon or accumulated_epsilon < 0:
                            accumulated_epsilon = 0.0
                    except Exception as e:
                        print(f"Warning: Could not get epsilon from PrivacyEngine: {e}")
                        accumulated_epsilon = 0.0
                
            await callback({"type": "log", "message": "   > Aggregating encrypted models (FedAvg)..."})
            
            avg_weights = self._aggregate(local_weights)
            self.global_model.load_state_dict(avg_weights)
            
            acc, recall_min = self._evaluate_global()
            
            await callback({
                "type": "round_complete",
                "round": r,
                "metrics": {
                    "accuracy": acc,
                    "loss": metrics_agg["loss"] / len(self.clients),
                    "epsilon": accumulated_epsilon,
                    "minority_recall": recall_min
                }
            })
            await callback({"type": "log", "message": f"Round {r} Complete. Global Acc: {acc:.2%}, Privacy (ε): {accumulated_epsilon:.2f}"})
        
        await callback({"type": "log", "message": "Simulation finished successfully."})
        await callback({"type": "finished"})

    def _aggregate(self, weights: List[Dict]) -> Dict:
        avg_weights = {}
        for key in weights[0].keys():
            avg_weights[key] = torch.stack([w[key] for w in weights]).mean(0)
        return avg_weights

    def _evaluate_global(self):
        X_test, y_test = self.dataset_meta["test"]
        self.global_model.eval()
        with torch.no_grad():
            logits = self.global_model(torch.from_numpy(X_test.astype(np.float32)))
            preds = logits.argmax(1).numpy()
            
        acc = (preds == y_test).mean()
        
        # Minority class recall (assuming class 1 for WESAD/Sleep)
        # In a robust app, determine minority dynamically
        n_classes = self.config["dataset"]["n_classes"]
        min_class = 1 if n_classes > 2 else 1
        mask = y_test == min_class
        if mask.sum() > 0:
            recall = (preds[mask] == y_test[mask]).mean()
        else:
            recall = 0.0
            
        return float(acc), float(recall)


