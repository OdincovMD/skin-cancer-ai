from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from redis_client import get_redis
from routers import auth, classification, health
from src.database import async_engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_redis().ping)
    await init_db()
    yield
    await async_engine.dispose()


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(classification.router)
