import io
import mimetypes
import os

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth_deps import require_verified_email_user_id
from image_access_token import create_image_access_token, verify_image_access_token
from minio_client import (
    BUCKET_NAME,
    create_bucket_if_not_exists,
    download_file_bytes,
    get_minio_client,
    is_file_in_minio,
    object_key_for_stored_filename,
    upload_file_to_minio,
)
from src.database import get_db
from src.queries.orm import Orm
from workers.tasks import run_classification

router = APIRouter(tags=["classification"])


@router.post("/uploadfile")
async def handle_upload(
    file: UploadFile = File(),
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    if not await Orm.user_exists(session, user_id):
        raise HTTPException(
            status_code=404,
            detail="Пользователь не найден в базе. Выйдите из аккаунта и войдите снова.",
        )

    if await Orm.count_user_active_classifications(session, user_id) > 0:
        raise HTTPException(
            status_code=429,
            detail="Уже выполняется классификация. Дождитесь завершения или обновите статус задания.",
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file_name = file.filename
    file_content = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    s3_client = get_minio_client()
    create_bucket_if_not_exists(s3_client, BUCKET_NAME)
    await Orm.insert_file_record(session, file_name=file_name, bucket_name=BUCKET_NAME)
    if not is_file_in_minio(s3_client, BUCKET_NAME, file_path):
        upload_file_to_minio(s3_client, BUCKET_NAME, file_path)

    try:
        file_id = await Orm.get_file_id_by_name(session, file_name=file_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    db_request = await Orm.create_classification_request(
        session,
        user_id=user_id,
        file_id=file_id,
        status="pending",
        result=None,
    )
    run_classification.delay(db_request.id)
    return {"job_id": db_request.id, "status": "pending"}


@router.get("/classification-jobs/active")
async def get_active_classification_job(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    payload = await Orm.get_user_active_classification_job(session, user_id)
    if not payload:
        return Response(status_code=204)
    fn = payload.get("file_name")
    if fn:
        try:
            payload = {
                **payload,
                "image_token": create_image_access_token(user_id, fn),
            }
        except RuntimeError:
            payload = {**payload, "image_token": None}
    return payload


@router.get("/classification-jobs/{job_id}")
async def get_classification_job(
    job_id: int,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    payload = await Orm.get_classification_job(session, job_id, user_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    meta = await Orm.get_classification_file_meta(session, job_id)
    fn = meta.get("file_name") if meta else None
    if fn:
        try:
            payload["image_token"] = create_image_access_token(user_id, fn)
        except RuntimeError:
            payload["image_token"] = None
    else:
        payload["image_token"] = None
    return payload


@router.post("/gethistory")
async def get_history(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    try:
        history = await Orm.get_classification_requests(session, user_id)
        out = []
        for row in history:
            item = dict(row)
            fn = item.get("file_name")
            if fn:
                try:
                    item["image_token"] = create_image_access_token(
                        user_id, fn
                    )
                except RuntimeError:
                    item["image_token"] = None
            else:
                item["image_token"] = None
            out.append(item)
        return out
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при получении истории: {str(e)}"
        )


@router.get("/history/image")
async def get_history_image(
    token: str,
    session: AsyncSession = Depends(get_db),
):
    """Отдаёт изображение из MinIO по подписанному токену (без перебора user_id/file_name)."""
    user_id, file_name = verify_image_access_token(token)
    bucket = await Orm.get_bucket_for_user_file(session, user_id, file_name)
    if not bucket:
        raise HTTPException(status_code=404, detail="Файл не найден или доступ запрещён")
    key = object_key_for_stored_filename(file_name)
    try:
        s3 = get_minio_client()
        body = download_file_bytes(s3, bucket, key)
    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code") or ""
        if code in ("NoSuchKey", "404", "NotFound"):
            raise HTTPException(
                status_code=404, detail="Объект в хранилище не найден"
            ) from e
        raise HTTPException(
            status_code=502, detail="Ошибка чтения из хранилища"
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    media = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    return StreamingResponse(io.BytesIO(body), media_type=media)
