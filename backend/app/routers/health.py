import asyncio

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from core.redis_client import get_redis
from src.config import settings
from src.database import async_session_maker

router = APIRouter(tags=["health"])


_ML_HEALTH_TIMEOUT = httpx.Timeout(2.5, connect=1.0)


async def _check_ml_health() -> dict:
    ml_url = settings.ML_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_ML_HEALTH_TIMEOUT) as client:
            response = await client.get(f"{ml_url}/health")
            response.raise_for_status()
            payload = response.json()
        ml_ok = isinstance(payload, dict) and payload.get("status") == "ok"
        models = payload.get("models") if isinstance(payload, dict) else None
        detail = "доступен"
        if isinstance(models, dict):
            loaded_count = models.get("loaded_count")
            total_count = models.get("total_count")
            if isinstance(loaded_count, int) and isinstance(total_count, int):
                detail = f"готово моделей: {loaded_count}/{total_count}"
        return {
            "ok": ml_ok,
            "label": "ML",
            "detail": detail if ml_ok else "вернул неожиданный ответ",
            "models": models if isinstance(models, dict) else None,
        }
    except Exception:
        return {
            "ok": False,
            "label": "ML",
            "detail": "недоступен",
            "models": None,
        }


@router.get("/health")
async def health():
    services = {
        "backend": {"ok": True, "label": "Backend API", "detail": "доступен"},
        "redis": {"ok": False, "label": "Redis", "detail": "недоступен"},
        "db": {"ok": False, "label": "PostgreSQL", "detail": "недоступен"},
    }
    try:
        await asyncio.to_thread(get_redis().ping)
        services["redis"] = {"ok": True, "label": "Redis", "detail": "доступен"}
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        services["db"] = {"ok": True, "label": "PostgreSQL", "detail": "доступен"}
    except Exception:
        payload = {
            "status": "unavailable",
            "healthy_count": sum(1 for item in services.values() if item["ok"]),
            "total_count": len(services),
            "services": services,
            "detail": "Критичные зависимости backend недоступны.",
        }
        return JSONResponse(status_code=503, content=payload)

    services["ml"] = await _check_ml_health()
    healthy_count = sum(1 for item in services.values() if item["ok"])
    total_count = len(services)
    status = "ok" if healthy_count == total_count else "degraded"
    return {
        "status": status,
        "healthy_count": healthy_count,
        "total_count": total_count,
        "services": services,
    }
