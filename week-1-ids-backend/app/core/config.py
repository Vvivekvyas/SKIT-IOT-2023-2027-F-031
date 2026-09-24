"""
Centralized app configuration.
Values are read from environment variables / a local .env file.
Never hardcode secrets here — this file only defines *names and defaults*.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General ---
    PROJECT_NAME: str = "ML-Based Intrusion Detection System"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"  # development | production

    # --- Database ---
    # Example: postgresql+psycopg://user:password@localhost:5432/ids_db
    DATABASE_URL: str = "postgresql+psycopg://ids_user:ids_pass@localhost:5432/ids_db"

    # --- Auth ---
    SECRET_KEY: str = "change-me-in-.env"  # override in .env, never commit real value
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # --- ML model serving ---
    # Path to the currently "active" trained model artifact.
    # Whoever finishes model training (Module 2) drops the winning model here,
    # and inference.py loads whatever this path points to — no code changes needed.
    ACTIVE_MODEL_PATH: str = "app/ml/artifacts/best_model.pkl"
    MODEL_FEATURE_LIST_PATH: str = "app/ml/artifacts/feature_columns.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached so we don't re-parse .env on every import."""
    return Settings()


settings = get_settings()
