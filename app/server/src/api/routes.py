from typing import List
import json

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .schemas import TrainingConfig, TrainingResponse, JobStatus
from ..core.jobs import job_store
from ..core.security import get_api_key
from ..core.database import ExperimentRun, engine
from ..ml.factory import create_trainer, get_trainer_mode
from sqlmodel import Session, select
import asyncio

router = APIRouter()


async def background_training_task(job_id: str, config: dict):
    job_store.update_job(job_id, {"status": "running"})
    runs = int(config.get("runs", 1))
    job_store.update_job(job_id, {"total_runs": runs})

    try:
        last_result = None
        mode = get_trainer_mode(config["clients"], config["sigma"])
        per_run_results: list[dict] = []

        seed_schedule = [42, 123, 456, 789, 1011]

        for run_idx in range(1, runs + 1):
            if runs == 1:
                current_seed = config.get("seed", 42)
            else:
                if run_idx - 1 < len(seed_schedule):
                    current_seed = seed_schedule[run_idx - 1]
                else:
                    current_seed = seed_schedule[(run_idx - 1) % len(seed_schedule)]
            config["seed"] = current_seed

            def update_callback(progress=None, metrics=None, log=None):
                updates = {}
                if progress is not None:
                    updates["progress"] = progress
                if metrics:
                    m = dict(metrics)
                    m["run_index"] = run_idx
                    updates["metrics"] = m
                    if "round" in m:
                        updates["current_round"] = m["round"]
                    elif "epoch" in m:
                        updates["current_round"] = m["epoch"]
                if log:
                    job_store.append_log(job_id, log)
                if updates:
                    updates["current_run"] = run_idx
                    updates["total_runs"] = runs
                    job_store.update_job(job_id, updates)

            trainer = create_trainer(job_id, config, update_callback)
            if run_idx == 1:
                job_store.append_log(job_id, f"Initialized {mode}")

            if hasattr(trainer, "train_data_full"):
                result = await asyncio.to_thread(
                    trainer.fit, trainer.train_data_full, trainer.val_loader
                )
            else:
                result = await asyncio.to_thread(
                    trainer.fit, trainer.train_loader, trainer.val_loader
                )

            test_loader = getattr(trainer, "test_loader", trainer.val_loader)
            eval_metrics = trainer.evaluate_full(test_loader)
            result.update(eval_metrics)
            last_result = result

            per_run_results.append(
                {
                    "run_index": run_idx,
                    "seed": current_seed,
                    "mode": mode,
                    "metrics": result,
                }
            )

            job_store.update_job(
                job_id,
                {
                    "metrics": {**result, "run_index": run_idx},
                    "current_run": run_idx,
                    "total_runs": runs,
                },
            )
            job_store.append_log(job_id, f"Run {run_idx}/{runs} completed")

        if last_result is None:
            raise RuntimeError("No runs executed")

        acc = float(last_result.get("accuracy", last_result.get("final_acc", 0.0)))
        eps = float(last_result.get("epsilon", last_result.get("final_epsilon", 0.0)))

        runs_summary = None
        if len(per_run_results) > 1:
            from statistics import mean, pstdev

            def collect(metric_key: str, fallback_key: str | None = None):
                values = []
                for r in per_run_results:
                    m = r["metrics"]
                    if metric_key in m:
                        values.append(float(m[metric_key]))
                    elif fallback_key and fallback_key in m:
                        values.append(float(m[fallback_key]))
                return values

            seeds = [r["seed"] for r in per_run_results]
            accuracies = collect("accuracy", "final_acc")
            minority_recalls = collect("minority_recall", None)
            epsilons = collect("final_epsilon", "epsilon")
            best_val_accs = collect("best_val_acc", "final_acc")

            def summarize(values: list[float]) -> dict | None:
                if not values:
                    return None
                if len(values) == 1:
                    return {"values": values, "mean": values[0], "std": 0.0}
                return {
                    "values": values,
                    "mean": mean(values),
                    "std": pstdev(values),
                }

            runs_summary = {
                "n_runs": len(per_run_results),
                "seeds": seeds,
                "accuracy": summarize(accuracies),
                "minority_recall": summarize(minority_recalls),
                "epsilon": summarize(epsilons),
                "best_val_acc": summarize(best_val_accs),
            }

        # Build test_metrics with as much detail as the trainer provides,
        # mirroring the structure of the offline experiment JSONs.
        test_metrics = {}
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "precision_per_class",
            "recall_per_class",
            "f1_per_class",
            "confusion_matrix",
            "class_names",
            "minority_recall",
            "class_imbalance",
        ]:
            if key in last_result:
                test_metrics[key] = last_result[key]

        experiment_like = {
            "experiment_info": {
                "name": f"{mode.lower()}_{config['dataset']}_ui",
                "dataset": config["dataset"],
                "method": "dp"
                if "DP" in mode
                else "baseline"
                if "BASELINE" in mode
                else "fl",
                "seed": config.get("seed"),
            },
            "training_metrics": {
                "total_epochs": int(
                    last_result.get(
                        "total_epochs",
                        last_result.get("epochs", 0),
                    )
                ),
                "best_epoch": int(
                    last_result.get(
                        "best_epoch",
                        last_result.get("epochs", 0),
                    )
                ),
                "epochs_no_improve": int(last_result.get("epochs_no_improve", 0)),
                "best_val_acc": float(
                    last_result.get("best_val_acc", last_result.get("final_acc", acc))
                ),
                "training_time_seconds": float(
                    last_result.get("training_time_seconds", 0.0)
                ),
                "convergence": {
                    "early_stopped": bool(
                        last_result.get("epochs_no_improve", 0) > 0
                    ),
                    "epochs_without_improvement": int(
                        last_result.get("epochs_no_improve", 0)
                    ),
                },
            },
            "privacy_metrics": {
                "decentralized": "FL" in mode,
                "formal_privacy": "DP" in mode,
                "final_epsilon": last_result.get("final_epsilon"),
                "n_clients": int(last_result.get("n_clients", config.get("clients", 0)))
                if "FL" in mode
                else None,
                "total_rounds": int(
                    last_result.get(
                        "total_epochs",
                        last_result.get("rounds", 0),
                    )
                )
                if "FL" in mode
                else None,
                "local_epochs": int(last_result.get("local_epochs", 1))
                if "FL" in mode
                else None,
            },
            "test_metrics": test_metrics,
            "timing": {
                "total_time_seconds": float(
                    last_result.get("training_time_seconds", 0.0)
                )
            },
            "hyperparameters": {
                "batch_size": config.get("batch_size", 128),
                "epochs": config.get("epochs", 40),
                "noise_multiplier": config.get("sigma", 0.0),
                "max_grad_norm": config.get("max_grad_norm", 5.0),
                "n_clients": int(last_result.get("n_clients", config.get("clients", 0)))
                if "FL" in mode
                else None,
                "local_epochs": int(last_result.get("local_epochs", 1))
                if "FL" in mode
                else None,
                "aggregation_method": "fedavg" if "FL" in mode else None,
            },
            "communication": last_result.get("communication", {}),
            "runs": per_run_results if per_run_results else None,
            "runs_summary": runs_summary,
            "model_path": None,
        }

        with Session(engine) as session:
            run = ExperimentRun(
                id=job_id,
                dataset=config["dataset"],
                mode=mode,
                sigma=float(config["sigma"]),
                clients=int(config["clients"]),
                final_accuracy=acc,
                final_epsilon=eps,
                config_json=json.dumps(config),
                result_json=json.dumps(experiment_like),
            )
            session.add(run)
            session.commit()

        job = job_store.get_job(job_id)
        if job and job.get("should_stop"):
            job_store.update_job(
                job_id,
                {
                    "status": "completed",
                    "progress": 100,
                    "metrics": last_result,
                },
            )
            job_store.append_log(job_id, "Training stopped by user")
        else:
            job_store.update_job(
                job_id,
                {
                    "status": "completed",
                    "progress": 100,
                    "metrics": last_result,
                },
            )
            job_store.append_log(job_id, "Training completed successfully")

    except Exception as e:
        import traceback

        error_msg = str(e)
        traceback_str = traceback.format_exc()
        job_store.update_job(
            job_id,
            {
                "status": "failed",
                "error": error_msg,
            },
        )
        job_store.append_log(job_id, f"Fatal Error: {error_msg}")
        job_store.append_log(job_id, traceback_str)
        print(f"Training job {job_id} failed: {error_msg}")
        print(traceback_str)


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    config: TrainingConfig,
    _: str = Depends(get_api_key),
):
    """
    Start a new training job.
    Returns immediately with job_id for polling.
    """
    config_dict = config.model_dump()
    job_id = job_store.create_job(config_dict)
    mode = get_trainer_mode(config.clients, config.sigma)
    
    # Offload to background task immediately using asyncio
    asyncio.create_task(background_training_task(job_id, config_dict))
    
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
async def stop_training(
    job_id: str,
    _: str = Depends(get_api_key),
):
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
async def list_jobs(
    limit: int = 10,
    _: str = Depends(get_api_key),
):
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
    epsilon = estimate_epsilon(sigma, sample_rate, epochs, delta, is_steps=False)
    
    # Classification
    classification = "Weak"
    if epsilon < 3:
        classification = "Strong"
    elif epsilon < 10:
        classification = "Moderate"
    
    return {
        "epsilon": round(epsilon, 2),
        "classification": classification,
    }


@router.get("/history")
async def get_history(
    limit: int = 20,
    _: str = Depends(get_api_key),
):
    with Session(engine) as session:
        statement = select(ExperimentRun).order_by(ExperimentRun.created_at.desc()).limit(limit)
        runs: List[ExperimentRun] = session.exec(statement).all()
        return {"runs": runs, "count": len(runs)}


@router.get("/export/{run_id}")
async def export_run(
    run_id: str,
    _: str = Depends(get_api_key),
):
    with Session(engine) as session:
        run = session.get(ExperimentRun, run_id)
        if not run or not run.result_json:
            raise HTTPException(status_code=404, detail="Run not found")
        return JSONResponse(content=json.loads(run.result_json))

@router.delete("/history/{run_id}")
async def delete_history(
    run_id: str,
    _: str = Depends(get_api_key),
):
    with Session(engine) as session:
        run = session.get(ExperimentRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        session.delete(run)
        session.commit()
    return {"status": "deleted", "id": run_id}
