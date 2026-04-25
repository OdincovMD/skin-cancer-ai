import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db
from src.schemas import DescriptionCallbackBody
from src.queries.orm import Orm

router = APIRouter(prefix="/internal", tags=["internal"])


def require_description_callback_token(
    x_service_token: Optional[str] = Header(default=None, alias="X-Service-Token"),
) -> None:
    expected = settings.DESCRIPTION_CALLBACK_API_TOKEN.strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="DESCRIPTION_CALLBACK_API_TOKEN is not configured.",
        )
    if not x_service_token or not secrets.compare_digest(x_service_token, expected):
        raise HTTPException(status_code=401, detail="Invalid service token.")


@router.post("/description-results/{job_id}")
async def receive_description_result(
    job_id: str,
    payload: DescriptionCallbackBody,
    session: AsyncSession = Depends(get_db),
    _: None = Depends(require_description_callback_token),
):
    updated = await Orm.upsert_description_callback(
        session,
        service_job_id=job_id,
        status=payload.status,
        description=payload.description,
        important_labels=payload.important_labels,
        bucketed_labels=payload.bucketed_labels,
        features_only=payload.features_only,
        description_result=payload.model_dump(mode="json"),
        error=payload.error,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Description job not found.")
    return {"ok": True}
