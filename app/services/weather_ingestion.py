"""Scheduled weather ingestion from OpenWeatherMap into PostGIS.

Every WEATHER_INGEST_INTERVAL_MINUTES (default 60) this fetches OWM "current
weather" for each WEATHER_TARGETS point and stores rainfall (`rain.1h`),
humidity and temperature in `realtime_weather_logs`. Polling hourly against a
1-hour rain accumulation gives an approximately continuous hourly series. It is
not exact: OWM's `dt` is not aligned to the hour, so consecutive windows can
overlap or leave small gaps.

Snowfall (`snow.1h`) is not stored. It only matters for high-altitude Sikkim /
Arunachal targets; add a column if those need it.

Runs in one of two ways:

- inside the API process (started from `app/main.py` lifespan when
  WEATHER_INGESTION_ENABLED=true), or
- standalone: `python -m app.services.weather_ingestion [--once]`.

Enable it in exactly one process. Inserts are idempotent on
(location_name, recorded_at), but each extra process spends OWM quota.

Missing configuration (disabled, no API key, no targets) is logged and the
scheduler simply doesn't start; it never stops the API from booting.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, WeatherTarget
from app.repositories import weather as weather_repo
from app.schemas.weather import RealtimeWeatherLogCreate

logger = logging.getLogger(__name__)

JOB_ID = "weather_ingestion"
# OWM free tier allows 60 calls/minute; a few in flight keeps a run short
# without bursting through that for large target lists.
MAX_CONCURRENT_REQUESTS = 5


@dataclass(frozen=True)
class IngestResult:
    fetched: int
    inserted: int
    failed: int


def parse_current_weather(
    target: WeatherTarget, payload: dict[str, Any]
) -> RealtimeWeatherLogCreate:
    """Normalise an OWM /weather response (requested with units=metric).
    Raises ValueError on an unexpected shape or an out-of-range value."""
    dt = payload.get("dt")
    if not isinstance(dt, int):
        raise ValueError("response has no integer 'dt'")
    main = payload.get("main")
    if not isinstance(main, dict):
        raise ValueError("response has no 'main' block")
    # pydantic's ValidationError subclasses ValueError, so range failures
    # (e.g. a Kelvin temperature) surface the same way as shape errors.
    return RealtimeWeatherLogCreate(
        location_name=target.name,
        latitude=target.latitude,
        longitude=target.longitude,
        rainfall_mm=_rain_1h_mm(payload),
        humidity=_number(main, "humidity"),
        temperature=_number(main, "temp"),
        recorded_at=datetime.fromtimestamp(dt, tz=UTC),
    )


def _number(block: dict[str, Any], key: str) -> Decimal:
    value = block.get(key)
    # bool is an int subclass; a JSON true is not a measurement.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"'{key}' is missing or not a number")
    return Decimal(str(round(float(value), 2)))


def _rain_1h_mm(payload: dict[str, Any]) -> Decimal:
    # OWM omits `rain` entirely when nothing fell, so absence means 0 mm.
    # A block present *without* "1h" is a different shape (e.g. only "3h");
    # treating that as 0 would under-report rainfall, so reject it instead.
    block = payload.get("rain")
    if block is None:
        return Decimal("0")
    if not isinstance(block, dict) or "1h" not in block:
        raise ValueError("'rain' present without a '1h' value")
    value = _number(block, "1h")
    if value < 0:
        raise ValueError("'rain.1h' is negative")
    return value


def _describe_error(exc: Exception) -> str:
    # Never log str(exc) for httpx errors: it includes the request URL, and the
    # URL carries the API key (`appid=`).
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        hint = " (check OPENWEATHER_API_KEY)" if code == 401 else ""
        return f"HTTP {code}{hint}"
    if isinstance(exc, httpx.HTTPError):
        return type(exc).__name__
    return str(exc)


async def fetch_target(
    client: httpx.AsyncClient, api_key: str, target: WeatherTarget
) -> RealtimeWeatherLogCreate:
    resp = await client.get(
        "/weather",
        params={
            "lat": target.latitude,
            "lon": target.longitude,
            "appid": api_key,
            "units": "metric",
        },
    )
    resp.raise_for_status()
    return parse_current_weather(target, resp.json())


async def ingest_once(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    api_key: str,
    targets: Sequence[WeatherTarget],
) -> IngestResult:
    """Fetch every target and store what succeeded. One bad target (timeout,
    5xx, odd payload) is logged and skipped; it doesn't sink the whole run."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def one(target: WeatherTarget) -> RealtimeWeatherLogCreate | None:
        async with semaphore:
            try:
                return await fetch_target(client, api_key, target)
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning(
                    "OpenWeatherMap fetch failed for %r: %s", target.name, _describe_error(exc)
                )
                return None

    results = await asyncio.gather(*(one(t) for t in targets))
    logs = [r for r in results if r is not None]
    failed = len(targets) - len(logs)

    inserted = 0
    if logs:
        # No endpoint here, so this job owns the transaction (the repository
        # only flushes, per the project convention).
        async with session_factory() as session:
            try:
                inserted = await weather_repo.insert_logs(session, logs)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    return IngestResult(fetched=len(logs), inserted=inserted, failed=failed)


class WeatherIngestionService:
    """Owns the APScheduler instance and the shared HTTP client."""

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._scheduler: AsyncIOScheduler | None = None
        self._client: httpx.AsyncClient | None = None
        self._running_job: asyncio.Task[Any] | None = None
        self._stopping = False
        self.last_result: IngestResult | None = None
        self.last_success_at: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def _api_key(self) -> str:
        key = self._settings.OPENWEATHER_API_KEY
        return key.get_secret_value() if key is not None else ""

    def _config_problem(self, *, require_enabled: bool) -> str | None:
        # `--once` runs ignore the enabled flag: it gates the *scheduler*, and a
        # manual run is an explicit request.
        if require_enabled and not self._settings.WEATHER_INGESTION_ENABLED:
            return "WEATHER_INGESTION_ENABLED is false"
        if not self._api_key():
            return "OPENWEATHER_API_KEY is not set"
        if not self._settings.WEATHER_TARGETS:
            return "WEATHER_TARGETS is empty"
        return None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.OPENWEATHER_BASE_URL,
                timeout=self._settings.WEATHER_HTTP_TIMEOUT_S,
            )
        return self._client

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            # Imported here so that importing this module doesn't build the engine.
            from app.db.session import SessionLocal

            self._session_factory = SessionLocal
        return self._session_factory

    def start(self) -> bool:
        """Start the hourly job (first run immediately). Must be called with an
        event loop running. Returns False, with a log line, if not configured."""
        problem = self._config_problem(require_enabled=True)
        if problem:
            logger.info("Weather ingestion not started: %s", problem)
            return False

        self._ensure_client()
        self._scheduler = AsyncIOScheduler(timezone=UTC)
        self._scheduler.add_job(
            self._run_job,
            IntervalTrigger(minutes=self._settings.WEATHER_INGEST_INTERVAL_MINUTES),
            id=JOB_ID,
            # Run once at startup, so a fresh deploy has data without waiting an hour.
            next_run_time=datetime.now(UTC),
            # A slow run must never overlap the next; after downtime, run once
            # rather than replaying every missed hour (OWM only serves "now").
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )
        self._scheduler.start()
        logger.info(
            "Weather ingestion started: %d target(s) every %d min",
            len(self._settings.WEATHER_TARGETS),
            self._settings.WEATHER_INGEST_INTERVAL_MINUTES,
        )
        return True

    async def run_once(self) -> IngestResult:
        problem = self._config_problem(require_enabled=False)
        if problem:
            raise RuntimeError(f"Weather ingestion is not configured: {problem}")
        result = await ingest_once(
            self._ensure_client(),
            self._get_session_factory(),
            self._api_key(),
            self._settings.WEATHER_TARGETS,
        )
        self.last_result = result
        if result.fetched:
            self.last_success_at = datetime.now(UTC)
        return result

    async def _run_job(self) -> None:
        if self._stopping:
            return
        self._running_job = asyncio.current_task()
        try:
            result = await self.run_once()
            log = logger.warning if result.failed else logger.info
            log(
                "Weather ingestion: %d fetched, %d new, %d failed",
                result.fetched,
                result.inserted,
                result.failed,
            )
        except asyncio.CancelledError:
            if not self._stopping:
                raise
            # Our own shutdown cancelled it; returning keeps APScheduler from
            # logging a routine stop as "Error running job".
            logger.info("Weather ingestion run cancelled by shutdown")
        except Exception:
            # Database down, etc. Log it and let the next interval retry.
            logger.exception("Weather ingestion run failed")
        finally:
            self._running_job = None

    async def shutdown(self) -> None:
        self._stopping = True
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            # AsyncIOScheduler applies shutdown via call_soon_threadsafe, i.e. on
            # the next loop iteration. Yield once so it has actually stopped.
            await asyncio.sleep(0)
        # AsyncIOScheduler doesn't await in-flight coroutines. Cancel a running
        # job so it can't touch the DB engine after the lifespan disposes it.
        job = self._running_job
        if job is not None and not job.done():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def _main(once: bool) -> None:
    from app.core.config import settings
    from app.db.session import engine

    service = WeatherIngestionService(settings)
    try:
        if once:
            result = await service.run_once()
            logger.info(
                "Weather ingestion: %d fetched, %d new, %d failed",
                result.fetched,
                result.inserted,
                result.failed,
            )
            return
        if not service.start():
            return
        await asyncio.Event().wait()  # run until interrupted
    finally:
        await service.shutdown()
        await engine.dispose()


def main() -> None:
    from app.core.config import settings
    from app.core.logging import configure_logging

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="run one ingestion pass and exit")
    args = parser.parse_args()

    configure_logging(settings.LOG_LEVEL)
    try:
        asyncio.run(_main(args.once))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
