import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from auth.api_key_deps import get_user_id_from_api_key
from auth.api_rate_limit import (
    enforce_api_v1_rate_limit,
    enforce_api_v1_status_rate_limit,
)
from services.classification import (
    active_job_payload,
    active_integration_job_payload,
    classification_job_payload,
    history_image_stream,
    history_with_image_tokens,
    integration_job_payload,
    perform_integration_upload,
    perform_upload,
)
from services.image_access import verify_image_access_token
from src.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["api_v1"])


async def api_v1_user_id_rate_limited(
    user_id: int = Depends(get_user_id_from_api_key),
) -> int:
    await enforce_api_v1_rate_limit(user_id)
    return user_id


async def api_v1_user_id_status_rate_limited(
    user_id: int = Depends(get_user_id_from_api_key),
) -> int:
    await enforce_api_v1_status_rate_limit(user_id)
    return user_id


@router.post("/uploadfile")
async def api_upload(
    file: UploadFile = File(),
    features_only: bool = Form(False),
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_rate_limited),
):
    return await perform_upload(
        session, user_id, file, features_only=features_only, source="api_v1"
    )


@router.post("/integrations/classifications")
async def api_integration_classification_upload(
    file: UploadFile = File(),
    external_user_id: str = Form(...),
    external_case_id: str = Form(...),
    idempotency_key: str = Form(...),
    features_only: bool = Form(True),
    callback_url: Optional[str] = Form(None),
    callback_token: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_rate_limited),
):
    return await perform_integration_upload(
        session,
        user_id,
        file,
        external_user_id=external_user_id,
        external_case_id=external_case_id,
        idempotency_key=idempotency_key,
        features_only=features_only,
        callback_url=callback_url,
        callback_token=callback_token,
    )


@router.get("/integrations/classifications/active")
async def api_active_integration_classification(
    external_user_id: str,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_status_rate_limited),
):
    payload = await active_integration_job_payload(session, user_id, external_user_id)
    if not payload:
        return Response(status_code=204)
    return payload


@router.get("/integrations/classifications/{job_id}")
async def api_integration_classification_job(
    job_id: int,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_status_rate_limited),
):
    payload = await integration_job_payload(session, user_id, job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return payload


@router.get("/classification-jobs/active")
async def api_active_job(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_rate_limited),
):
    payload = await active_job_payload(session, user_id)
    if not payload:
        return Response(status_code=204)
    return payload


@router.get("/classification-jobs/{job_id}")
async def api_job(
    job_id: int,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_rate_limited),
):
    payload = await classification_job_payload(session, user_id, job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return payload


@router.post("/gethistory")
async def api_history(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(api_v1_user_id_rate_limited),
):
    try:
        return await history_with_image_tokens(session, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка при получении истории (API v1)")
        raise HTTPException(
            status_code=500,
            detail="Не удалось получить историю. Повторите попытку позже.",
        ) from e


@router.get("/history/image")
async def api_history_image(
    token: str,
    session: AsyncSession = Depends(get_db),
):
    user_id, _ = verify_image_access_token(token)
    await enforce_api_v1_rate_limit(user_id)
    return await history_image_stream(session, token)
