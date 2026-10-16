"""Application settings, loaded once from environment variables / `.env`.

Import `settings` (or call `get_settings()` in a dependency) rather than
reading `os.environ` anywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root: app/core/config.py -> parents[2]. Relative paths in settings
# (model files, media dir) resolve against this, not the process CWD, so
# `uvicorn app.main:app` behaves the same from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WeatherTarget(BaseModel):
    """A named point whose precipitation is ingested (see services/weather_ingestion.py)."""

    name: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


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
    # Comma-separated or JSON list, e.g. "https://geoalert.vercel.app,http://localhost:3000".
    # "*" allows any origin but is only accepted when ENVIRONMENT=local (see below).
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

    # --- Weather ingestion (OpenWeatherMap) -----------------------------------
    # Off by default. Enable it on exactly ONE process: every API replica with it
    # on would call OWM for every target (inserts are idempotent, quota is not).
    WEATHER_INGESTION_ENABLED: bool = False
    OPENWEATHER_API_KEY: SecretStr | None = None
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    # OWM refreshes current conditions roughly every 10 minutes; faster is waste.
    WEATHER_INGEST_INTERVAL_MINUTES: int = Field(60, ge=10)
    WEATHER_HTTP_TIMEOUT_S: float = Field(10.0, gt=0)
    # JSON list: [{"name": "Shillong", "latitude": 25.5788, "longitude": 91.8933}, ...]
    WEATHER_TARGETS: list[WeatherTarget] = []

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        if isinstance(v, str):
            import json

            return json.loads(v)
        return v

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _normalise_origins(cls, v: list[str]) -> list[str]:
        # Browsers send Origin without a trailing slash, so an origin copied from
        # the address bar ("https://geoalert.vercel.app/") would never match.
        return [o.strip().rstrip("/") for o in v if o.strip()]

    @model_validator(mode="after")
    def _no_wildcard_cors_outside_local(self) -> Settings:
        # "*" lets any website call the API from a visitor's browser. Fine on a
        # laptop, never on a reachable deployment: fail at startup, not silently.
        if self.cors_allow_all and self.ENVIRONMENT != "local":
            raise ValueError(
                f"CORS_ORIGINS='*' is only allowed when ENVIRONMENT=local "
                f"(got {self.ENVIRONMENT!r}). List the exact frontend origins instead."
            )
        return self

    @field_validator("WEATHER_TARGETS")
    @classmethod
    def _unique_target_names(cls, v: list[WeatherTarget]) -> list[WeatherTarget]:
        # The name is part of the dedup key in realtime_weather_logs; two targets
        # sharing one would silently drop each other's rows.
        names = [t.name for t in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"WEATHER_TARGETS names must be unique; duplicated: {dupes}")
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
    def cors_allow_all(self) -> bool:
        return "*" in self.CORS_ORIGINS

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
