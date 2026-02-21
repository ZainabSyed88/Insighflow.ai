"""Configuration management for AI Data Insights Analyst."""

from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # NVIDIA API
    nvidia_api_key: str = ""
    openai_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # LLM Configuration
    llm_model: str = "qwen/qwen3-coder-480b-a35b-instruct"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    
    # Database
    database_url: str = "sqlite:///./data/insights_analyst.db"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    # Application
    max_upload_size_mb: int = 50
    max_verification_attempts: int = 3
    outlier_iqr_multiplier: float = 1.5
    missing_value_threshold: float = 0.5
    
    # ------------------------------------------------------------------
    # Fast model for per-agent narration (override via LLM_FAST_MODEL)
    # Falls back to the primary model if not set.
    # ------------------------------------------------------------------
    llm_fast_model: str = ""

    # ------------------------------------------------------------------
    # Enterprise-readiness placeholders (NOT active — design hooks only)
    # ------------------------------------------------------------------
    # AUTH_PROVIDER: str = ""          # e.g. "okta", "azure-ad"
    # STORAGE_BACKEND: str = "local"   # e.g. "s3", "gcs", "azure-blob"
    # WEBHOOK_URL: str = ""            # e.g. post-pipeline completion hook

    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def database_dir(self) -> Path:
        """Get or create database directory."""
        db_path = Path(self.database_url.replace("sqlite:///./", ""))
        db_path.parent.mkdir(exist_ok=True, parents=True)
        return db_path.parent
    
    @property
    def log_dir(self) -> Path:
        """Get or create logs directory."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        return log_path.parent


# Global settings instance
settings = Settings()
