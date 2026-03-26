from collections.abc import AsyncGenerator

from pydantic import BaseModel as PBaseModel
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from src.config import settings

async_engine = create_async_engine(
    settings.DATABASE_URL_async,
    pool_size=5,
    max_overflow=10,
)

async_session_maker = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    import src.models  # noqa: F401 — регистрация ORM-моделей в Base.metadata

    async with async_engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
    if not table_names:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


# Модель Pydantic для входных данных


class UserSignUp(PBaseModel):
    lastName: str
    firstName: str
    login: str
    email: str
    password: str


class Credentials(PBaseModel):
    login: str
    password: str


class GetHistoryRequest(PBaseModel):
    user_id: int
