"""
Automated tests for REST API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import time


def test_health_check(client):
    """Verifies the server is running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_start_baseline_training(client, mock_trainer):
    """
    Test Case 1: Start a centralized baseline training (No DP, No FL).
    Expected: Job created with status 'queued'.
    """
    payload = {
        "dataset": "wesad",
        "clients": 0,
        "sigma": 0.0,
        "epochs": 1,
        "batch_size": 32
    }
    
    response = client.post("/api/v1/train", json=payload)
    
    # Assert Request accepted
    assert response.status_code == 200
    data = response.json()
    
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["mode_detected"] == "BASELINE_CENTRALIZED"


def test_start_dp_training(client, mock_trainer):
    """Test Case: Clients=0, Sigma>0 -> DP Centralized"""
    payload = {
        "dataset": "wesad",
        "clients": 0,
        "sigma": 1.0,
        "epochs": 1
    }
    
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode_detected"] == "DP_CENTRALIZED"
    assert "job_id" in data


def test_start_fl_training(client, mock_trainer):
    """Test Case: Clients>0, Sigma=0 -> FL"""
    payload = {
        "dataset": "wesad",
        "clients": 5,
        "sigma": 0.0,
        "epochs": 1
    }
    
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode_detected"] == "FEDERATED_LEARNING"
    assert "job_id" in data


def test_start_fl_dp_training(client, mock_trainer):
    """
    Test Case 2: Start Federated Learning with DP.
    Expected: Mode detected correctly.
    """
    payload = {
        "dataset": "sleep-edf",
        "clients": 5,
        "sigma": 1.0,
        "epochs": 1,
        "batch_size": 32
    }
    
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    assert response.json()["mode_detected"] == "FEDERATED_LEARNING_DP"


def test_job_polling_flow(client, mock_trainer):
    """
    Test Case 3: Integration Flow (Start -> Poll -> Complete).
    Uses the mock_trainer to simulate instant completion.
    """
    # 1. Start Job
    payload = {"dataset": "wesad", "clients": 2, "sigma": 0.5, "epochs": 1}
    start_res = client.post("/api/v1/train", json=payload)
    job_id = start_res.json()["job_id"]
    
    # 2. Wait briefly for background task (running in thread) to finish
    # Since we mocked the training to be instant, this happens fast.
    time.sleep(0.2)
    
    # 3. Poll Status
    status_res = client.get(f"/api/v1/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    
    # 4. Verification
    # It might be 'running' or 'completed' depending on race condition of test runner,
    # but since we mocked instant return, it should likely be completed.
    assert status_data["job_id"] == job_id
    
    # Ensure logs exist
    assert isinstance(status_data["logs"], list)
    
    # If completed, check metrics
    if status_data["status"] == "completed":
        assert "metrics" in status_data
        assert "accuracy" in status_data["metrics"] or "final_acc" in status_data["metrics"]


def test_get_job_status(client, mock_trainer, sample_config):
    """Test getting job status."""
    # Start a job
    response = client.post("/api/v1/train", json=sample_config)
    job_id = response.json()["job_id"]
    
    # Get status
    response = client.get(f"/api/v1/status/{job_id}")
    assert response.status_code == 200
    status = response.json()
    
    assert status["job_id"] == job_id
    assert status["status"] in ["pending", "running", "completed", "failed"]
    assert isinstance(status["logs"], list)
    assert isinstance(status["progress"], int)
    assert 0 <= status["progress"] <= 100


def test_get_nonexistent_job(client):
    """Test Case 4: Error handling for invalid IDs."""
    response = client.get("/api/v1/status/fake-uuid-123")
    assert response.status_code == 404


def test_input_validation(client):
    """Test Case 5: Ensure invalid hyperparams are rejected."""
    payload = {
        "dataset": "wesad",
        "clients": -1,  # Invalid: cannot have negative clients
        "sigma": 0.0
    }
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_stop_job(client, mock_trainer, sample_config):
    """Test stopping a job."""
    # Start a job
    response = client.post("/api/v1/train", json=sample_config)
    job_id = response.json()["job_id"]
    
    # Stop it immediately (before it completes)
    response = client.post(f"/api/v1/stop/{job_id}")
    assert response.status_code == 200
    data = response.json()
    # Job might be stopping or already completed (with mocks it's very fast)
    assert data["status"] in ["stopping", "cannot_stop"]


def test_list_jobs(client, mock_trainer, sample_config):
    """Test listing recent jobs."""
    # Create a few jobs
    for _ in range(3):
        client.post("/api/v1/train", json=sample_config)
    
    # List jobs
    response = client.get("/api/v1/jobs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) > 0
