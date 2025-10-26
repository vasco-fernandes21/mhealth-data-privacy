# src/training/fl_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import time

from src.training.base_trainer import BaseTrainer
from src.privacy.fl_client import FLClient
from src.privacy.fl_server import FLServer


class FLTrainer(BaseTrainer):
    """Federated Learning trainer."""
    
    def setup_optimizer_and_loss(self) -> None:
        """Setup loss (clients manage optimizers)."""
        cfg = self.config['training']
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(self.device)
    
    def fit(self, train_loaders: List[DataLoader],
            val_loaders: List[DataLoader],
            client_ids: List[str],
            epochs: int = 100,
            patience: int = 8,
            output_dir: str = None) -> Dict[str, Any]:
        """Train with FL."""
        
        self.setup_optimizer_and_loss()
        self._reset_training_state()
        
        # Setup clients + server
        clients = [
            FLClient(
                client_id=cid,
                model=type(self.model)(
                    self.config, device=str(self.device)
                ),
                train_loader=tl,
                config=self.config,
                device=str(self.device)
            )
            for cid, tl in zip(client_ids, train_loaders)
        ]
        
        server = FLServer(
            model=self.model,
            clients=clients,
            config=self.config,
            device=str(self.device)
        )
        
        start_time = time.time()
        
        for round_num in range(1, epochs + 1):
            # Train round
            train_metrics = server.train_round()
            train_loss = train_metrics['avg_loss']
            train_acc = train_metrics['avg_accuracy']
            
            # Validate
            val_metrics = server.evaluate_on_clients(val_loaders)
            val_loss = val_metrics['avg_loss']
            val_acc = val_metrics['avg_accuracy']
            
            # Store + log
            self._update_history(round_num, train_loss, train_acc,
                               val_loss, val_acc)
            print(
                f"Round {round_num:03d}: "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                if output_dir:
                    from pathlib import Path
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(
                        f"{output_dir}/best_model.pth"
                    )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    break
        
        elapsed = time.time() - start_time
        
        if output_dir:
            from pathlib import Path
            best_path = Path(output_dir) / 'best_model.pth'
            if best_path.exists():
                self.load_checkpoint(str(best_path))
        
        return {
            'total_rounds': round_num,
            'training_time_seconds': elapsed,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'n_clients': len(clients)
        }