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
    JWT_SECRET_KEY: str  # required, no default — must come from env/secret manager
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database (Pranjal owns the actual schema/models; this is just the connection string)
    DATABASE_URL: str = "sqlite:///./ids_dev.db"

    # CORS — locked to known frontend origins, never "*"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]

    # Rate limiting
    LOGIN_RATE_LIMIT: str = "5/minute"
    PREDICT_RATE_LIMIT: str = "60/minute"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
