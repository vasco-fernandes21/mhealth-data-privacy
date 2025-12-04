from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class FLServer:
    def __init__(self):
        self.round = 0
        self.global_model = None
        self.client_updates: Dict[str, Dict] = {}
        self.history: List[Dict] = []
    
    def init_round(self):
        self.round += 1
        self.client_updates = {}
        if self.global_model is None:
            self.global_model = self._init_model()
    
    def _init_model(self) -> Dict:
        model = nn.Sequential(
            nn.Linear(140, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        return {
            name: param.detach().cpu().numpy().tolist()
            for name, param in model.named_parameters()
        }
    
    def receive_update(self, client_id: str, weights: Dict, epsilon: float, samples: int) -> Dict:
        self.client_updates[client_id] = {
            "weights": weights,
            "epsilon": epsilon,
            "samples": samples
        }
        
        if len(self.client_updates) == 3:
            aggregated = self._aggregate()
            total_epsilon = sum(u["epsilon"] for u in self.client_updates.values())
            
            self.history.append({
                "round": self.round,
                "epsilon": total_epsilon,
                "clients": list(self.client_updates.keys())
            })
            
            return {
                "status": "aggregated",
                "round": self.round,
                "epsilon": total_epsilon
            }
        
        return {
            "status": "waiting",
            "received": len(self.client_updates),
            "expected": 3
        }
    
    def _aggregate(self) -> Dict:
        total_samples = sum(u["samples"] for u in self.client_updates.values())
        aggregated = None
        
        for client_id, update in self.client_updates.items():
            weight_factor = update["samples"] / total_samples
            
            if aggregated is None:
                aggregated = {
                    k: np.array(v) * weight_factor
                    for k, v in update["weights"].items()
                }
            else:
                for k in aggregated:
                    aggregated[k] += np.array(update["weights"][k]) * weight_factor
        
        self.global_model = {k: v.tolist() for k, v in aggregated.items()}
        return self.global_model

server = FLServer()

@app.post("/fl/round/start")
async def start_round():
    server.init_round()
    return {"round": server.round, "status": "started"}

@app.get("/fl/models/download")
async def download():
    if server.global_model is None:
        raise HTTPException(status_code=404, detail="No model available")
    return {"round": server.round, "model": server.global_model}

@app.post("/fl/models/upload/{client_id}")
async def upload(client_id: str, body: Dict):
    weights = body.get("weights", {})
    epsilon = body.get("epsilon", 0.0)
    samples = body.get("samples", 0)
    
    if not weights or samples == 0:
        raise HTTPException(status_code=400, detail="Missing weights or samples")
    
    result = server.receive_update(client_id, weights, epsilon, samples)
    return {**result, "round": server.round}

@app.get("/fl/status")
async def fl_status():
    return {
        "round": server.round,
        "received": len(server.client_updates),
        "expected": 3,
        "clients": list(server.client_updates.keys()),
        "history": server.history
    }