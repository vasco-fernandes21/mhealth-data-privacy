"""
REST API endpoints for training management.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from .schemas import TrainingConfig, TrainingResponse, JobStatus
from ..core.jobs import job_store
from ..ml.factory import create_trainer, get_trainer_mode
import asyncio

router = APIRouter()


async def background_training_task(job_id: str, config: dict):
    """
    Background task that runs training asynchronously.
    Updates job store with progress via callback.
    Handles different data passing for FL vs Centralized.
    """
    job_store.update_job(job_id, {"status": "running"})
    
    try:
        # Create callback function that updates job store
        def update_callback(progress=None, metrics=None, log=None):
            """Callback to update job progress."""
            updates = {}
            if progress is not None:
                updates["progress"] = progress
            if metrics:
                updates["metrics"] = metrics
                if "round" in metrics:
                    updates["current_round"] = metrics["round"]
                elif "epoch" in metrics:
                    updates["current_round"] = metrics["epoch"]
            if log:
                job_store.append_log(job_id, log)
            if updates:
                job_store.update_job(job_id, updates)
        
        # Create trainer
        trainer = create_trainer(job_id, config, update_callback)
        mode = get_trainer_mode(config["clients"], config["sigma"])
        job_store.append_log(job_id, f"Initialized {mode}")
        
        # Execute Training (Blocking)
        # We handle data passing differently for FL vs Centralized
        if hasattr(trainer, 'train_data_full'):
            # Federated: pass (X, y, subjects) tuple
            result = await asyncio.to_thread(trainer.fit, trainer.train_data_full, trainer.val_loader)
        else:
            # Centralized: pass train_loader and val_loader
            result = await asyncio.to_thread(trainer.fit, trainer.train_loader, trainer.val_loader)
        
        # Final Evaluation on TEST set (matching paper)
        test_loader = getattr(trainer, 'test_loader', trainer.val_loader)  # Fallback to val if test not available
        eval_metrics = trainer.evaluate_full(test_loader)
        result.update(eval_metrics)
        
        # Check if stopped
        job = job_store.get_job(job_id)
        if job and job.get("should_stop"):
            job_store.update_job(job_id, {
                "status": "completed",
                "progress": 100,
                "metrics": result
            })
            job_store.append_log(job_id, "Training stopped by user")
        else:
            job_store.update_job(job_id, {
                "status": "completed",
                "progress": 100,
                "metrics": result
            })
            job_store.append_log(job_id, "Training completed successfully")
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        job_store.update_job(job_id, {
            "status": "failed",
            "error": error_msg
        })
        job_store.append_log(job_id, f"Fatal Error: {error_msg}")
        job_store.append_log(job_id, traceback_str)
        print(f"Training job {job_id} failed: {error_msg}")
        print(traceback_str)


@router.post("/train", response_model=TrainingResponse)
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Start a new training job.
    Returns immediately with job_id for polling.
    """
    config_dict = config.model_dump()
    job_id = job_store.create_job(config_dict)
    mode = get_trainer_mode(config.clients, config.sigma)
    
    # Offload to background task immediately
    background_tasks.add_task(background_training_task, job_id, config_dict)
    
    return TrainingResponse(
        job_id=job_id,
        status="queued",
        mode_detected=mode,
        message=f"Training job queued. Mode: {mode}"
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """
    Get current status of a training job.
    Frontend polls this endpoint every 1-2 seconds.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Map internal dict to Pydantic model
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        current_round=job.get("current_round", 0),
        logs=job["logs"],
        metrics=job.get("metrics"),
        error=job.get("error"),
        mode_detected=get_trainer_mode(job["config"]["clients"], job["config"]["sigma"])
    )


@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    """
    Stop a running training job.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] not in ["pending", "running"]:
        return {"status": "cannot_stop", "message": f"Job is {job['status']}, cannot stop"}
    
    job_store.stop_job(job_id)
    job_store.append_log(job_id, "Stop requested by user")
    
    return {"status": "stopping", "message": "Stop signal sent to training job"}


@router.get("/jobs")
async def list_jobs(limit: int = 10):
    """
    List recent jobs (for debugging/admin).
    """
    jobs = job_store.list_jobs(limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.post("/estimate-privacy")
async def estimate_privacy(request: dict):
    """
    Estimate privacy budget (epsilon) before training.
    Quick calculation using RDP accountant without actual training.
    """
    from ..ml.privacy import estimate_epsilon
    
    dataset_size = request.get("dataset_size", 5000)
    batch_size = request.get("batch_size", 128)
    epochs = request.get("epochs", 40)
    sigma = request.get("sigma", 1.0)
    delta = 1e-5
    
    if sigma == 0:
        return {"epsilon": 0.0, "classification": "No Privacy"}
    
    sample_rate = batch_size / dataset_size if dataset_size > 0 else 0.001
    epsilon = estimate_epsilon(sigma, sample_rate, epochs, delta, is_steps=False)  # epochs, not steps
    
    # Classification
    classification = "Weak"
    if epsilon < 3:
        classification = "Strong"
    elif epsilon < 10:
        classification = "Moderate"
    
    return {
        "epsilon": round(epsilon, 2),
        "classification": classification
    }
