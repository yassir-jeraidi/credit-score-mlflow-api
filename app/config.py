"""
Application Configuration using Pydantic Settings.

Loads configuration from environment variables.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Application
    app_name: str = Field(default="Credit Score API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="credit-scoring",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    model_name: str = Field(
        default="credit-score-model",
        env="MODEL_NAME"
    )
    model_stage: str = Field(
        default="Production",
        env="MODEL_STAGE"
    )

    # Security
    jwt_secret_key: str = Field(default="secret", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="JWT_EXPIRES_IN_MINUTES")

    # PostgreSQL (for MLflow backend)
    postgres_user: str = Field(default="mlflow", env="POSTGRES_USER")
    postgres_password: str = Field(default="mlflow123", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="mlflow", env="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
