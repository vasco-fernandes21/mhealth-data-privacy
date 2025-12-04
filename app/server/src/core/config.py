import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PrivacyHealth API"
    PORT: int = 8000
    ENV: str = "development"
    
    # Data Paths - Defaults to relative path but overridable via env
    # From app/server/src/core/config.py -> app/server -> project root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    # Use symlink at app/server/data/processed or fallback to project root
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data" / "processed")))
    
    # ML Defaults
    DEFAULT_BATCH_SIZE: int = 128
    DEFAULT_MAX_GRAD_NORM: float = 1.0
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields in .env


settings = Settings()

