import pytest
import time


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_start_baseline_training(client, mock_trainer):
    payload = {
        "dataset": "wesad",
        "clients": 0,
        "sigma": 0.0,
        "epochs": 1,
        "batch_size": 32
    }
    
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["mode_detected"] == "BASELINE_CENTRALIZED"


def test_start_dp_training(client, mock_trainer):
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
    payload = {"dataset": "wesad", "clients": 2, "sigma": 0.5, "epochs": 1}
    start_res = client.post("/api/v1/train", json=payload)
    job_id = start_res.json()["job_id"]
    
    time.sleep(0.2)
    status_res = client.get(f"/api/v1/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    
    assert status_data["job_id"] == job_id
    assert isinstance(status_data["logs"], list)
    
    if status_data["status"] == "completed":
        assert "metrics" in status_data
        assert "accuracy" in status_data["metrics"] or "final_acc" in status_data["metrics"]


def test_get_job_status(client, mock_trainer, sample_config):
    response = client.post("/api/v1/train", json=sample_config)
    job_id = response.json()["job_id"]
    response = client.get(f"/api/v1/status/{job_id}")
    assert response.status_code == 200
    status = response.json()
    
    assert status["job_id"] == job_id
    assert status["status"] in ["pending", "running", "completed", "failed"]
    assert isinstance(status["logs"], list)
    assert isinstance(status["progress"], int)
    assert 0 <= status["progress"] <= 100


def test_get_nonexistent_job(client):
    response = client.get("/api/v1/status/fake-uuid-123")
    assert response.status_code == 404


def test_input_validation(client):
    payload = {
        "dataset": "wesad",
        "clients": -1,
        "sigma": 0.0
    }
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 422


def test_stop_job(client, mock_trainer, sample_config):
    response = client.post("/api/v1/train", json=sample_config)
    job_id = response.json()["job_id"]
    
    response = client.post(f"/api/v1/stop/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["stopping", "cannot_stop"]


def test_list_jobs(client, mock_trainer, sample_config):
    for _ in range(3):
        client.post("/api/v1/train", json=sample_config)
    response = client.get("/api/v1/jobs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) > 0
