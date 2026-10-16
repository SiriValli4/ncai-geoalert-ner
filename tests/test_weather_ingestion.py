"""Weather ingestion tests -- no network, no database."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.core.config import Settings, WeatherTarget
from app.repositories import weather as weather_repo
from app.schemas.weather import RealtimeWeatherLogCreate
from app.services import weather_ingestion as wi

API_KEY = "sekret-owm-key"
SHILLONG = WeatherTarget(name="Shillong", latitude=25.5788, longitude=91.8933)
GANGTOK = WeatherTarget(name="Gangtok", latitude=27.3389, longitude=88.6065)
DT = 1_760_600_000


def _payload(**extra: Any) -> dict[str, Any]:
    return {
        "dt": DT,
        "coord": {"lat": 25.58, "lon": 91.89},
        "main": {"temp": 19.46, "humidity": 88},
        **extra,
    }


def test_parse_rain_humidity_temperature() -> None:
    log = wi.parse_current_weather(SHILLONG, _payload(rain={"1h": 12.347}, snow={"1h": 0.5}))
    assert log.rainfall_mm == Decimal("12.35")
    assert log.humidity == Decimal("88")
    assert log.temperature == Decimal("19.46")
    assert log.recorded_at == datetime.fromtimestamp(DT, tz=UTC)
    assert log.location_name == "Shillong"
    assert (log.latitude, log.longitude) == (SHILLONG.latitude, SHILLONG.longitude)


def test_parse_no_rain_block_means_zero() -> None:
    assert wi.parse_current_weather(SHILLONG, _payload()).rainfall_mm == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"coord": {}, "main": {"temp": 20, "humidity": 50}},  # no dt
        {"dt": DT},  # no main block
        _payload(main={"humidity": 50}),  # no temp
        _payload(main={"temp": 20, "humidity": True}),  # bool is not a measurement
        _payload(main={"temp": 292.6, "humidity": 50}),  # Kelvin: units=metric not honoured
        _payload(main={"temp": 20, "humidity": 140}),
        _payload(rain={"3h": 4.0}),  # different accumulation window: don't record as 0
        _payload(rain={"1h": -1}),
        _payload(rain={"1h": "lots"}),
    ],
)
def test_parse_rejects_unexpected_shapes(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        wi.parse_current_weather(SHILLONG, payload)


async def test_insert_logs_is_idempotent_and_never_writes_location() -> None:
    captured: list[Any] = []

    class _Result:
        def all(self) -> list[int]:
            return [1]

    class _Db:
        async def scalars(self, stmt: Any) -> _Result:
            captured.append(stmt)
            return _Result()

        async def flush(self) -> None:
            return None

    log = wi.parse_current_weather(SHILLONG, _payload())
    assert await weather_repo.insert_logs(_Db(), [log]) == 1  # type: ignore[arg-type]
    sql = str(captured[0].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_realtime_weather_logs_name_recorded DO NOTHING" in sql
    # `location` is a generated column; Postgres rejects any INSERT that supplies it.
    columns = sql.split("(", 1)[1].split(")", 1)[0]
    assert "latitude" in columns and "location," not in columns


async def test_insert_logs_empty_is_a_noop() -> None:
    assert await weather_repo.insert_logs(None, []) == 0  # type: ignore[arg-type]


class _FakeSession:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[RealtimeWeatherLogCreate]:
    rows: list[RealtimeWeatherLogCreate] = []

    async def fake_insert(db: Any, obs: Sequence[RealtimeWeatherLogCreate]) -> int:
        rows.extend(obs)
        return len(obs)

    monkeypatch.setattr(weather_repo, "insert_logs", fake_insert)
    return rows


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://owm.test/data/2.5", transport=httpx.MockTransport(handler)
    )


async def test_ingest_once_stores_successes_and_skips_failures(
    captured: list[RealtimeWeatherLogCreate], caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["lat"] == str(GANGTOK.latitude):
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(200, json=_payload(rain={"1h": 3.2}))

    session = _FakeSession()
    caplog.set_level(logging.DEBUG)
    async with _client(handler) as client:
        result = await wi.ingest_once(client, lambda: session, API_KEY, [SHILLONG, GANGTOK])  # type: ignore[arg-type]

    assert result == wi.IngestResult(fetched=1, inserted=1, failed=1)
    assert [o.location_name for o in captured] == ["Shillong"]
    assert session.committed
    assert seen[0].url.params["appid"] == API_KEY
    assert seen[0].url.params["units"] == "metric"
    assert "HTTP 503" in caplog.text
    # The key travels in the URL; it must never reach the logs.
    assert API_KEY not in caplog.text


async def test_ingest_once_with_no_successes_skips_db(
    captured: list[RealtimeWeatherLogCreate],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Invalid API key"})

    def no_session() -> Any:
        raise AssertionError("DB opened with nothing to write")

    async with _client(handler) as client:
        result = await wi.ingest_once(client, no_session, API_KEY, [SHILLONG])  # type: ignore[arg-type]
    assert result == wi.IngestResult(fetched=0, inserted=0, failed=1)


def _settings(**env: Any) -> Settings:
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "env",
    [
        {},  # disabled by default
        {"WEATHER_INGESTION_ENABLED": True, "WEATHER_TARGETS": [SHILLONG]},  # no key
        {"WEATHER_INGESTION_ENABLED": True, "OPENWEATHER_API_KEY": API_KEY},  # no targets
    ],
)
async def test_scheduler_does_not_start_when_unconfigured(env: dict[str, Any]) -> None:
    service = wi.WeatherIngestionService(_settings(**env))
    assert service.start() is False
    assert not service.is_running
    await service.shutdown()


async def test_scheduler_starts_with_immediate_hourly_job() -> None:
    service = wi.WeatherIngestionService(
        _settings(
            WEATHER_INGESTION_ENABLED=True,
            OPENWEATHER_API_KEY=API_KEY,
            WEATHER_TARGETS=[SHILLONG],
        )
    )
    # Don't let the immediate first run fire during the test.
    service._run_job = _noop  # type: ignore[method-assign]
    assert service.start() is True
    try:
        assert service._scheduler is not None
        job = service._scheduler.get_job(wi.JOB_ID)
        assert job.trigger.interval.total_seconds() == 3600
        assert job.max_instances == 1 and job.coalesce
    finally:
        await service.shutdown()
    assert not service.is_running


async def _noop() -> None:
    return None


async def test_shutdown_cancels_in_flight_run_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import asyncio

    service = wi.WeatherIngestionService(_settings(OPENWEATHER_API_KEY=API_KEY))
    started = asyncio.Event()

    async def slow_run_once() -> wi.IngestResult:
        started.set()
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    service.run_once = slow_run_once  # type: ignore[method-assign]
    task = asyncio.create_task(service._run_job())
    await started.wait()
    await service.shutdown()

    assert task.done() and not task.cancelled()
    assert "cancelled by shutdown" in caplog.text
    assert "failed" not in caplog.text


def test_weather_targets_parse_from_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "WEATHER_TARGETS", '[{"name": "Aizawl", "latitude": 23.7271, "longitude": 92.7176}]'
    )
    assert _settings().WEATHER_TARGETS == [
        WeatherTarget(name="Aizawl", latitude=23.7271, longitude=92.7176)
    ]


def test_weather_target_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _settings(WEATHER_TARGETS=[SHILLONG, SHILLONG])


def test_secret_key_not_in_settings_repr() -> None:
    assert API_KEY not in repr(_settings(OPENWEATHER_API_KEY=API_KEY))
