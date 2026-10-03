"""Async engine + session factory.

One engine per process, created at import time (it doesn't connect until
first use). Request handlers get a session via `app.api.deps.DbSession`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE_S,
    # Drops connections the DB/network silently closed (common behind cloud LBs).
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a session per request. Commits are explicit in the repository /
    endpoint layer; anything uncommitted is rolled back when the request ends.
    """
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
