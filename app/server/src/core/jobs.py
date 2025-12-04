"""
In-memory Job Store for managing training jobs.
Thread-safe job state management.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Lock


class JobStore:
    """Thread-safe in-memory job store. In production, use Redis."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def create_job(self, config: Dict[str, Any]) -> str:
        """Create a new training job and return its ID."""
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "config": config,
                "progress": 0,
                "current_round": 0,
                "logs": [],
                "metrics": {},
                "should_stop": False,
                "error": None
            }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job fields. Returns True if job exists."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)
                return True
            return False
    
    def append_log(self, job_id: str, message: str) -> None:
        """Append log message to job. Keeps last 100 logs."""
        with self._lock:
            if job_id in self._jobs:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self._jobs[job_id]["logs"].append(f"[{timestamp}] {message}")
                # Keep log size manageable
                self._jobs[job_id]["logs"] = self._jobs[job_id]["logs"][-100:]
    
    def stop_job(self, job_id: str) -> bool:
        """Mark job for stopping."""
        return self.update_job(job_id, {"should_stop": True})
    
    def list_jobs(self, limit: int = 50) -> list:
        """List recent jobs (for debugging/admin)."""
        with self._lock:
            jobs = list(self._jobs.values())
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return jobs[:limit]


# Global singleton instance
job_store = JobStore()

