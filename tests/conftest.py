from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app


class _NoDbSession:
    """Stand-in for tests that must fail validation before touching the DB."""

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        raise AssertionError(f"DB accessed unexpectedly: session.{name}")


async def _no_db() -> AsyncIterator[_NoDbSession]:
    yield _NoDbSession()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """App client with the real lifespan (models load) and no database."""
    app.dependency_overrides[get_db] = _no_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
