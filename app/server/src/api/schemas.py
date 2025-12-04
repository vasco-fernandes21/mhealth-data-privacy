"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List


class TrainingConfig(BaseModel):
    """Training configuration from frontend."""
    dataset: Literal['wesad', 'sleep-edf'] = Field(..., description="Dataset to use")
    clients: int = Field(..., ge=0, le=10, description="0 = Centralized, >0 = Federated")
    sigma: float = Field(..., ge=0.0, le=5.0, description="0 = No Privacy, >0 = DP noise multiplier")
    batch_size: int = Field(default=128, ge=1, le=512)
    epochs: int = Field(default=40, ge=1, le=200, description="Global rounds for FL, epochs for centralized")


class TrainingResponse(BaseModel):
    """Response when starting a training job."""
    job_id: str
    status: str
    mode_detected: str
    message: str = "Training job queued successfully"


class JobStatus(BaseModel):
    """Current status of a training job."""
    job_id: str
    status: Literal['pending', 'running', 'completed', 'failed']
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_round: int = Field(default=0, ge=0, description="Current round/epoch")
    logs: List[str] = Field(default_factory=list, description="Training logs")
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    mode_detected: Optional[str] = None

