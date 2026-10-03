"""Application settings, loaded once from environment variables / `.env`.

Import `settings` (or call `get_settings()` in a dependency) rather than
reading `os.environ` anywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root: app/core/config.py -> parents[2]. Relative paths in settings
# (model files, media dir) resolve against this, not the process CWD, so
# `uvicorn app.main:app` behaves the same from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---------------------------------------------------------------
    PROJECT_NAME: str = "GeoAlert NER API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # --- Database ----------------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://geoalert:geoalert@localhost:5432/geoalert"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE_S: int = 1800
    DB_ECHO: bool = False

    # --- ML models ---------------------------------------------------------
    RISK_MODEL_PATH: Path = Path("ml/models/landslide_risk_model.joblib")
    HAZARD_MODEL_PATH: Path = Path("ml/models/hazard_yolov8_demo.pt")
    HAZARD_CONF_DEFAULT: float = Field(0.25, ge=0.0, le=1.0)
    HAZARD_MAX_DIM: int = 1280

    # --- Uploads -----------------------------------------------------------
    MAX_UPLOAD_MB: float = 15
    MEDIA_DIR: Path = Path("media")
    MEDIA_URL_PREFIX: str = "/media"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        if isinstance(v, str):
            import json

            return json.loads(v)
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def _require_asyncpg(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the async driver: postgresql+asyncpg://...")
        return v

    @field_validator("RISK_MODEL_PATH", "HAZARD_MODEL_PATH", "MEDIA_DIR")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def max_upload_bytes(self) -> int:
        return int(self.MAX_UPLOAD_MB * 1024 * 1024)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
