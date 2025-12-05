import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app
from src.core.jobs import job_store


@pytest.fixture
def client():
    return TestClient(app, headers={"X-API-Key": "mhealth-secret-2024"})


@pytest.fixture(autouse=True)
def clean_job_store():
    job_store._jobs = {}
    yield


@pytest.fixture
def mock_trainer(mocker):
    mock_create = mocker.patch("src.api.routes.create_trainer")
    mock_instance = MagicMock()
    mock_instance.fit.return_value = {
        "final_acc": 0.95,
        "final_epsilon": 1.2,
        "epochs": 10
    }
    mock_instance.evaluate_full.return_value = {
        "accuracy": 0.96,
        "minority_recall": 0.85
    }
    mock_instance.train_loader = MagicMock()
    mock_instance.val_loader = MagicMock()
    mock_instance.train_data_full = None
    mock_create.return_value = mock_instance
    return mock_create


@pytest.fixture
def sample_config():
    return {
        "dataset": "wesad",
        "clients": 5,
        "sigma": 1.0,
        "batch_size": 128,
        "epochs": 2
    }
