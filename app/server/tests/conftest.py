"""
Pytest configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app
from src.core.jobs import job_store


@pytest.fixture
def client():
    """
    FastAPI Test Client.
    """
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_job_store():
    """
    Automatically clears the in-memory job store before every test.
    Ensures tests are isolated.
    """
    job_store._jobs = {}
    yield


@pytest.fixture
def mock_trainer(mocker):
    """
    Mocks the ML Trainer Factory.
    Instead of starting a heavy PyTorch training, it returns a mock object
    that simulates immediate success.
    """
    # Patch the factory where it is imported in the router/background task
    mock_create = mocker.patch("src.api.routes.create_trainer")
    
    # Create a mock trainer instance
    mock_instance = MagicMock()
    
    # Mock the fit method to return dummy metrics immediately
    mock_instance.fit.return_value = {
        "final_acc": 0.95,
        "final_epsilon": 1.2,
        "epochs": 10
    }
    
    # Mock evaluate_full
    mock_instance.evaluate_full.return_value = {
        "accuracy": 0.96,
        "minority_recall": 0.85
    }
    
    # Ensure train_loader/val_loader attributes exist (required by routes.py)
    mock_instance.train_loader = MagicMock()
    mock_instance.val_loader = MagicMock()
    mock_instance.train_data_full = None  # For FL trainers
    
    mock_create.return_value = mock_instance
    return mock_create


@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return {
        "dataset": "wesad",
        "clients": 5,
        "sigma": 1.0,
        "batch_size": 128,
        "epochs": 2  # Short for testing
    }
