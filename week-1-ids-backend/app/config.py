
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "ML-Based Intrusion Detection System"
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = (
        "postgresql://ids_user:ids_password@localhost:5432/ids_db"
    )

    # Security
    SECRET_KEY: str = "change-this-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Machine Learning model
    ACTIVE_MODEL_PATH: str = str(
        BASE_DIR / "app" / "ml" / "artifacts" / "best_model.pkl"
    )

    # Data artifacts
    FEATURE_COLUMNS_PATH: str = str(
        BASE_DIR / "app" / "ml" / "artifacts" / "feature_columns.json"
    )
    SCALER_PATH: str = str(
        BASE_DIR / "app" / "ml" / "artifacts" / "scaler.pkl"
    )
    LABEL_ENCODERS_PATH: str = str(
        BASE_DIR / "app" / "ml" / "artifacts" / "label_encoders.pkl"
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
