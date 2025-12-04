from fastapi import APIRouter
from pydantic import BaseModel
from ..ml.privacy import estimate_epsilon

router = APIRouter()


class PrivacyEstimateRequest(BaseModel):
    dataset_size: int = 5000  # default approx
    batch_size: int = 32
    epochs: int = 1
    sigma: float


@router.post("/estimate-privacy")
def get_privacy_estimate(req: PrivacyEstimateRequest):
    sample_rate = req.batch_size / req.dataset_size if req.dataset_size > 0 else 0.001
    epsilon = estimate_epsilon(req.sigma, sample_rate, req.epochs)
    
    classification = "Weak"
    if epsilon < 3:
        classification = "Strong"
    elif epsilon < 10:
        classification = "Moderate"

    return {
        "epsilon": round(epsilon, 2),
        "classification": classification
    }

