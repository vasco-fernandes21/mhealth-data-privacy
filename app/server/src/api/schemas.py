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
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility (None = no explicit seeding)",
    )
    max_grad_norm: float = Field(
        default=5.0,
        ge=0.1,
        le=10.0,
        description="Gradient clipping threshold C (max grad norm)"
    )
    use_class_weights: bool = Field(
        default=True,
        description="Enable weighted loss to handle class imbalance"
    )
    runs: int = Field(default=1, ge=1, le=5, description="Number of independent runs")
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

