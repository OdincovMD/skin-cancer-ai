import asyncio
import io
import mimetypes
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from core.minio_client import (
    BUCKET_NAME,
    create_bucket_if_not_exists,
    delete_object,
    download_file_bytes,
    get_minio_client,
    object_key_for_stored_filename,
    unique_object_key_for_user,
    upload_bytes_to_minio,
)
from services.image_access import (
    create_image_access_token,
    verify_image_access_token,
)
from src.queries.orm import Orm
from workers.tasks import run_classification


async def perform_upload(
    session, user_id: int, file: UploadFile, features_only: bool = False
) -> Dict[str, Any]:
    if not await Orm.user_exists(session, user_id):
        raise HTTPException(
            status_code=404,
            detail="Пользователь не найден в базе.",
        )

    if await Orm.count_user_active_classifications(session, user_id) > 0:
        raise HTTPException(
            status_code=429,
            detail="Уже выполняется классификация. Дождитесь завершения или обновите статус задания.",
        )

    file_content = await file.read()
    object_key = unique_object_key_for_user(user_id, file.filename)
    ctype = (file.content_type or "").split(";")[0].strip() or "application/octet-stream"

    try:
        s3_client = get_minio_client()

        def _ensure_bucket_and_put() -> None:
            create_bucket_if_not_exists(s3_client, BUCKET_NAME)
            upload_bytes_to_minio(
                s3_client, BUCKET_NAME, object_key, file_content, ctype
            )

        await asyncio.to_thread(_ensure_bucket_and_put)
    except ClientError as e:
        raise HTTPException(
            status_code=502,
            detail="Не удалось сохранить файл в хранилище. Повторите попытку позже.",
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        file_id = await Orm.insert_file_record(
            session, file_name=object_key, bucket_name=BUCKET_NAME
        )
    except Exception:
        try:
            s3 = get_minio_client()
            await asyncio.to_thread(delete_object, s3, BUCKET_NAME, object_key)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Не удалось сохранить метаданные файла. Повторите попытку.",
        )

    try:
        db_request = await Orm.create_classification_request(
            session,
            user_id=user_id,
            file_id=file_id,
            status="pending",
            result=None,
        )
    except Exception:
        try:
            await Orm.delete_file_record_by_id(session, file_id)
        except Exception:
            pass
        try:
            s3 = get_minio_client()
            await asyncio.to_thread(delete_object, s3, BUCKET_NAME, object_key)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Не удалось создать задание классификации. Повторите попытку.",
        )

    run_classification.delay(db_request.id, features_only)
    return {"job_id": db_request.id, "status": "pending"}


async def active_job_payload(session, user_id: int) -> Optional[Dict[str, Any]]:
    payload = await Orm.get_user_active_classification_job(session, user_id)
    if not payload:
        return None
    fn = payload.get("file_name")
    if fn:
        try:
            return {
                **payload,
                "image_token": create_image_access_token(user_id, fn),
            }
        except RuntimeError:
            return {**payload, "image_token": None}
    return payload


async def classification_job_payload(
    session, user_id: int, job_id: int
) -> Optional[Dict[str, Any]]:
    payload = await Orm.get_classification_job(session, job_id, user_id)
    if not payload:
        return None
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


async def history_with_image_tokens(session, user_id: int) -> List[Dict[str, Any]]:
    history = await Orm.get_classification_requests(session, user_id)
    out: List[Dict[str, Any]] = []
    for row in history:
        item = dict(row)
        fn = item.get("file_name")
        if fn:
            try:
                item["image_token"] = create_image_access_token(user_id, fn)
            except RuntimeError:
                item["image_token"] = None
        else:
            item["image_token"] = None
        out.append(item)
    return out


async def history_image_stream(session, token: str) -> StreamingResponse:
    user_id, file_name = verify_image_access_token(token)
    bucket = await Orm.get_bucket_for_user_file(session, user_id, file_name)
    if not bucket:
        raise HTTPException(status_code=404, detail="Файл не найден или доступ запрещён")
    key = object_key_for_stored_filename(file_name)
    try:
        s3 = get_minio_client()
        body = await asyncio.to_thread(download_file_bytes, s3, bucket, key)
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
