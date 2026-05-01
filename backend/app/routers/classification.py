from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import require_verified_email_user_id
from services.classification import (
    active_job_payload,
    classification_job_payload,
    history_image_stream,
    history_with_image_tokens,
    perform_upload,
)
from src.database import get_db

router = APIRouter(tags=["classification"])


@router.post("/uploadfile")
async def handle_upload(
    file: UploadFile = File(),
    features_only: bool = Form(False),
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    return await perform_upload(session, user_id, file, features_only=features_only)


@router.get("/classification-jobs/active")
async def get_active_classification_job(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    payload = await active_job_payload(session, user_id)
    if not payload:
        return Response(status_code=204)
    return payload


@router.get("/classification-jobs/{job_id}")
async def get_classification_job(
    job_id: int,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    payload = await classification_job_payload(session, user_id, job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return payload


@router.post("/gethistory")
async def get_history(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    try:
        return await history_with_image_tokens(session, user_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при получении истории: {str(e)}"
        ) from e


@router.get("/history/image")
async def get_history_image(
    token: str,
    session: AsyncSession = Depends(get_db),
):
    return await history_image_stream(session, token)
