from __future__ import annotations

from typing import Any, Dict

import httpx

from src.config import settings


def description_enabled() -> bool:
    return bool(settings.DESCRIPTION_ENABLED and settings.DESCRIPTION_SERVICE_URL.strip())


def _service_headers() -> Dict[str, str]:
    token = settings.DESCRIPTION_SERVICE_API_TOKEN.strip()
    if not token:
        raise RuntimeError("DESCRIPTION_SERVICE_API_TOKEN is not configured.")
    return {"X-Service-Token": token}


async def create_description_job(
    client: httpx.AsyncClient,
    job_id: str,
    image_name: str,
    image_bytes: bytes,
    image_content_type: str,
    mask_name: str,
    mask_bytes: bytes,
) -> Dict[str, Any]:
    response = await client.post(
        f"{settings.DESCRIPTION_SERVICE_URL.rstrip('/')}/v1/description-jobs",
        headers=_service_headers(),
        data={"job_id": job_id},
        files={
            "image": (image_name, image_bytes, image_content_type),
            "mask": (mask_name, mask_bytes, "image/png"),
        },
    )
    response.raise_for_status()
    return response.json()


async def submit_description_classification(
    client: httpx.AsyncClient,
    job_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    response = await client.post(
        f"{settings.DESCRIPTION_SERVICE_URL.rstrip('/')}/v1/description-jobs/{job_id}/classification",
        headers={
            **_service_headers(),
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    return response.json()
