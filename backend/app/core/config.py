"""
App configuration.

All values are pulled from environment variables (see .env.example).
Nothing secret is ever hardcoded here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # General
    APP_NAME: str = "Hybrid IDS API"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"  # development | staging | production

    # Auth / JWT
    JWT_SECRET_KEY: str  # required, no default - must come from env/secret manager
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = "sqlite:///./ids_dev.db"

    # CORS - locked to known frontend origins, never "*"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]

    # Rate limiting
    LOGIN_RATE_LIMIT: str = "5/minute"
    PREDICT_RATE_LIMIT: str = "60/minute"

    # Dataset ingestion
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 200

    # Data pipeline integration (the ML team's code in the repo's top-level src/data).
    # Leave PIPELINE_ROOT empty to use the repo root automatically.
    PIPELINE_ROOT: str | None = None
    # Validation reads at most this many rows (large CSVs would use too much memory).
    VALIDATION_SAMPLE_ROWS: int = 100_000

    # Prediction API (Week 9)
    # Path to a trained checkpoint (format in docs/api/PREDICT_API.md). Empty = none yet.
    PREDICT_MODEL_PATH: str | None = None
    # Run the real model code with UNTRAINED random weights so the API can be demoed.
    # Ignored when ENV=production. Responses are labelled "demo-untrained".
    PREDICT_DEMO_MODE: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
