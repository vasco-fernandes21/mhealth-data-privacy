import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PrivacyHealth API"
    PORT: int = 8000
    ENV: str = "development"
    API_KEY: str = "mhealth-secret-2024"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data" / "processed")))
    
    # ML Defaults
    DEFAULT_BATCH_SIZE: int = 128
    DEFAULT_MAX_GRAD_NORM: float = 5.0
    
    class Config:
        env_file = ".env"
        extra = "ignore"  


settings = Settings()

