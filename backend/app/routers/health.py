import asyncio

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from redis_client import get_redis
from src.database import async_session_maker

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    try:
        await asyncio.to_thread(get_redis().ping)
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="unavailable")
    return {"status": "ok"}
