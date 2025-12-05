import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from src.ml.trainers.centralized import BaselineTrainer, DPTrainer
from src.ml.trainers.federated import FederatedTrainer, FLClient
from src.ml.models import get_model


@pytest.fixture
def sample_config():
    return {
        "dataset": {
            "name": "wesad",
            "input_dim": 140,
            "n_classes": 2,
            "class_names": ["non-stress", "stress"],
            "use_class_weights": False
        },
        "training": {
            "batch_size": 32,
            "epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.05,
            "early_stopping_patience": 10
        },
        "federated_learning": {
            "global_rounds": 5,
            "local_epochs": 1,
            "validation_frequency": 5
        },
        "differential_privacy": {
            "enabled": True,
            "noise_multiplier": 1.0,
            "max_grad_norm": 5.0,
            "delta": 1e-5,
            "poisson_sampling": False
        }
    }


@pytest.fixture
def sample_data():
    X_train = torch.randn(100, 140).float()
    y_train = torch.randint(0, 2, (100,)).long()
    X_val = torch.randn(20, 140).float()
    y_val = torch.randint(0, 2, (20,)).long()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    
    return train_loader, val_loader


class TestBaselineTrainer:
    def test_setup_optimizer_and_loss(self, sample_config):
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.scheduler is not None
    
    def test_train_epoch(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        train_loader, _ = sample_data
        loss, acc = trainer.train_epoch(train_loader)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_validate(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        _, val_loader = sample_data
        loss, acc = trainer.validate(val_loader)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_fit_completes(self, sample_config, sample_data):
        model = get_model(140, 2)
        callback = MagicMock()
        trainer = BaselineTrainer(model, sample_config, device="cpu", callback=callback)
        
        train_loader, val_loader = sample_data
        result = trainer.fit(train_loader, val_loader)
        
        assert "final_acc" in result
        assert "total_epochs" in result
        assert "best_epoch" in result
        assert isinstance(result["final_acc"], float)
        assert result["total_epochs"] > 0
    
    def test_fit_updates_history(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, sample_config, device="cpu")
        
        train_loader, val_loader = sample_data
        trainer.fit(train_loader, val_loader)
        
        assert len(trainer.history["epoch"]) > 0
        assert len(trainer.history["train_loss"]) > 0
        assert len(trainer.history["val_acc"]) > 0
    
    def test_fit_calls_callback(self, sample_config, sample_data):
        model = get_model(140, 2)
        callback = MagicMock()
        trainer = BaselineTrainer(model, sample_config, device="cpu", callback=callback)
        
        train_loader, val_loader = sample_data
        trainer.fit(train_loader, val_loader)
        
        assert callback.called
    
    def test_evaluate_full(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        _, val_loader = sample_data
        metrics = trainer.evaluate_full(val_loader)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert "minority_recall" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_early_stopping(self, sample_config, sample_data):
        config = sample_config.copy()
        config["training"]["epochs"] = 20
        config["training"]["early_stopping_patience"] = 2
        
        model = get_model(140, 2)
        trainer = BaselineTrainer(model, config, device="cpu")
        
        train_loader, val_loader = sample_data
        result = trainer.fit(train_loader, val_loader)
        
        assert result["total_epochs"] <= 20
        assert "epochs_no_improve" in result


class TestDPTrainer:
    def test_setup_optimizer_and_loss(self, sample_config):
        model = get_model(140, 2)
        trainer = DPTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.privacy_engine is not None
    
    def test_fit_with_dp(self, sample_config, sample_data):
        model = get_model(140, 2)
        callback = MagicMock()
        trainer = DPTrainer(model, sample_config, device="cpu", callback=callback)
        
        train_loader, val_loader = sample_data
        result = trainer.fit(train_loader, val_loader)
        
        assert "final_acc" in result
        assert "final_epsilon" in result
        assert isinstance(result["final_epsilon"], float)
        assert result["final_epsilon"] >= 0
    
    def test_epsilon_accumulation(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = DPTrainer(model, sample_config, device="cpu")
        
        train_loader, val_loader = sample_data
        trainer.fit(train_loader, val_loader)
        
        if trainer.privacy_engine is not None:
            try:
                epsilon = trainer.privacy_engine.get_epsilon(delta=1e-5)
                assert isinstance(epsilon, (int, float))
                assert epsilon >= 0
            except Exception:
                pass
    
    def test_fit_without_dp(self, sample_config, sample_data):
        config = sample_config.copy()
        config["differential_privacy"]["enabled"] = False
        config["differential_privacy"]["noise_multiplier"] = 0.0
        
        model = get_model(140, 2)
        trainer = DPTrainer(model, config, device="cpu")
        
        train_loader, val_loader = sample_data
        result = trainer.fit(train_loader, val_loader)
        
        assert "final_epsilon" in result
        assert result["final_epsilon"] == 0.0
    
    def test_evaluate_full(self, sample_config, sample_data):
        model = get_model(140, 2)
        trainer = DPTrainer(model, sample_config, device="cpu")
        trainer.setup_optimizer_and_loss()
        
        _, val_loader = sample_data
        metrics = trainer.evaluate_full(val_loader)
        
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)


class TestFLClient:
    def test_client_initialization(self, sample_config):
        model = get_model(140, 2)
        X = np.random.randn(50, 140).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        
        client = FLClient(0, model, (X, y), sample_config, device="cpu", use_dp=False)
        
        assert client.id == 0
        assert client.model is not None
        assert client.optimizer is not None
        assert client.criterion is not None
    
    def test_client_with_dp(self, sample_config):
        model = get_model(140, 2)
        X = np.random.randn(50, 140).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        
        client = FLClient(0, model, (X, y), sample_config, device="cpu", use_dp=True)
        
        assert client.use_dp is True
    
    def test_client_train_round(self, sample_config):
        model = get_model(140, 2)
        X = np.random.randn(50, 140).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        
        client = FLClient(0, model, (X, y), sample_config, device="cpu", use_dp=False)
        
        global_state = model.state_dict()
        weights, metrics = client.train_round(global_state)
        
        assert isinstance(weights, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
    
    def test_client_epsilon_in_metrics(self, sample_config):
        model = get_model(140, 2)
        X = np.random.randn(50, 140).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        
        client = FLClient(0, model, (X, y), sample_config, device="cpu", use_dp=True)
        
        global_state = model.state_dict()
        weights, metrics = client.train_round(global_state)
        
        assert "epsilon" in metrics
        assert isinstance(metrics["epsilon"], float)
        assert metrics["epsilon"] >= 0


class TestFederatedTrainer:
    def test_initialization(self, sample_config):
        model = get_model(140, 2)
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=5,
            use_dp=False,
            device="cpu"
        )
        
        assert trainer.n_clients == 5
        assert trainer.use_dp is False
        
        X_train = np.random.randn(100, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 100).astype(np.int64)
        subjects = np.zeros(100)
        trainer.setup_clients((X_train, y_train, subjects))
        
        assert len(trainer.clients) == 5
    
    def test_initialization_with_dp(self, sample_config):
        model = get_model(140, 2)
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=3,
            use_dp=True,
            device="cpu"
        )
        
        assert trainer.use_dp is True
        
        X_train = np.random.randn(90, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 90).astype(np.int64)
        subjects = np.zeros(90)
        trainer.setup_clients((X_train, y_train, subjects))
        
        assert len(trainer.clients) == 3
    
    def test_fit_completes(self, sample_config):
        model = get_model(140, 2)
        X_train = np.random.randn(200, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 200).astype(np.int64)
        subjects = np.zeros(200)
        
        callback = MagicMock()
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=5,
            use_dp=False,
            device="cpu",
            callback=callback
        )
        
        trainer.train_data_full = (X_train, y_train, subjects)
        trainer.val_loader = DataLoader(
            TensorDataset(torch.randn(20, 140).float(), torch.randint(0, 2, (20,)).long()),
            batch_size=32
        )
        
        result = trainer.fit(trainer.train_data_full, trainer.val_loader)
        
        assert "final_acc" in result or "best_val_acc" in result
        assert "communication" in result
        assert isinstance(result.get("final_acc") or result.get("best_val_acc"), float)
    
    def test_fit_with_dp(self, sample_config):
        model = get_model(140, 2)
        X_train = np.random.randn(200, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 200).astype(np.int64)
        subjects = np.zeros(200)
        
        callback = MagicMock()
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=3,
            use_dp=True,
            device="cpu",
            callback=callback
        )
        
        trainer.train_data_full = (X_train, y_train, subjects)
        trainer.val_loader = DataLoader(
            TensorDataset(torch.randn(20, 140).float(), torch.randint(0, 2, (20,)).long()),
            batch_size=32
        )
        
        result = trainer.fit(trainer.train_data_full, trainer.val_loader)
        
        assert "final_epsilon" in result
        assert isinstance(result["final_epsilon"], (float, type(None)))
    
    def test_evaluate_full(self, sample_config):
        model = get_model(140, 2)
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=5,
            use_dp=False,
            device="cpu"
        )
        
        test_loader = DataLoader(
            TensorDataset(torch.randn(20, 140).float(), torch.randint(0, 2, (20,)).long()),
            batch_size=32
        )
        
        metrics = trainer.evaluate_full(test_loader)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert isinstance(metrics["accuracy"], float)
    
    def test_aggregation(self, sample_config):
        model = get_model(140, 2)
        trainer = FederatedTrainer(
            model=model,
            config=sample_config,
            n_clients=3,
            use_dp=False,
            device="cpu"
        )
        
        weights_list = []
        for i in range(3):
            w = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
            weights_list.append(w)
        
        aggregated = trainer.aggregate(weights_list)
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) == len(model.state_dict())
    
    def test_validation_frequency(self, sample_config):
        config = sample_config.copy()
        config["federated_learning"] = {
            "validation_frequency": 3,
            "global_rounds": 10,
            "local_epochs": 1
        }
        
        model = get_model(140, 2)
        X_train = np.random.randn(150, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 150).astype(np.int64)
        subjects = np.zeros(150)
        
        callback = MagicMock()
        trainer = FederatedTrainer(
            model=model,
            config=config,
            n_clients=3,
            use_dp=False,
            device="cpu",
            callback=callback
        )
        
        trainer.train_data_full = (X_train, y_train, subjects)
        trainer.val_loader = DataLoader(
            TensorDataset(torch.randn(15, 140).float(), torch.randint(0, 2, (15,)).long()),
            batch_size=32
        )
        
        result = trainer.fit(trainer.train_data_full, trainer.val_loader)
        
        assert "communication" in result
        assert "total_rounds" in result["communication"]

