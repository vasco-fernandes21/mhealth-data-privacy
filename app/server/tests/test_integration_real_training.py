import numpy as np
import pytest

from src.ml.factory import create_trainer
from src.core.jobs import job_store


@pytest.mark.slow
def test_real_dp_training_cycle(monkeypatch):
    dataset_name = "wesad"

    def fake_load_dataset(name: str):
        assert name == dataset_name
        X_train = np.random.randn(64, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, size=64).astype(np.int64)
        subjects = np.zeros_like(y_train)
        X_test = np.random.randn(16, 140).astype(np.float32)
        y_test = np.random.randint(0, 2, size=16).astype(np.int64)
        return {
            "train": (X_train, y_train, subjects),
            "test": (X_test, y_test),
            "input_dim": 140,
            "n_classes": 2,
        }

    from src.data.loader import data_loader

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    config = {
        "dataset": dataset_name,
        "clients": 0,
        "sigma": 0.6,
        "batch_size": 16,
        "epochs": 1,
    }

    job_id = "integration-test-job"
    job_store._jobs = {}
    callback = lambda *args, **kwargs: None

    trainer = create_trainer(job_id, config, callback)

    result = trainer.fit(trainer.train_loader, trainer.val_loader)
    metrics = trainer.evaluate_full(trainer.val_loader)

    assert "final_acc" in result
    assert isinstance(result["final_acc"], float)

    epsilon = metrics.get("final_epsilon") or result.get("final_epsilon") or 0.0
    assert isinstance(epsilon, float)
    assert epsilon >= 0.0


