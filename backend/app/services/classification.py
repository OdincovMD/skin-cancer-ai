import asyncio
import io
import mimetypes
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import IntegrityError

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
    create_artifact_access_token,
    create_image_access_token,
    verify_artifact_access_token,
    verify_image_access_token,
)
from src.queries.orm import Orm
from workers.tasks import run_classification


PROCESSING_MODE_CLASSIFICATION = "classification"
PROCESSING_MODE_MASK = "mask"
ALLOWED_PROCESSING_MODES = {PROCESSING_MODE_CLASSIFICATION, PROCESSING_MODE_MASK}

ARTIFACT_FILENAMES = {
    "mask": "mask.png",
    "masked_image": "masked_image.png",
    "archive": "mask_results.zip",
}


def normalize_processing_mode(value: Optional[str]) -> str:
    mode = (value or PROCESSING_MODE_CLASSIFICATION).strip().lower()
    if mode not in ALLOWED_PROCESSING_MODES:
        raise HTTPException(
            status_code=400,
            detail="processing_mode должен быть classification или mask",
        )
    return mode


def _artifact_response_item(user_id: int, artifact: Dict[str, Any]) -> Dict[str, Any]:
    artifact_type = str(artifact["artifact_type"])
    token = None
    try:
        token = create_artifact_access_token(user_id, int(artifact["id"]))
    except RuntimeError:
        token = None
    return {
        "token": token,
        "filename": ARTIFACT_FILENAMES.get(
            artifact_type,
            artifact["file_name"].split("/")[-1],
        ),
        "content_type": artifact["content_type"],
        "size_bytes": artifact["size_bytes"],
        "checksum_sha256": artifact["checksum_sha256"],
    }


async def _mask_result_with_artifact_tokens(
    session,
    user_id: int,
    job_id: int,
) -> Dict[str, Any]:
    artifacts = await Orm.list_classification_artifacts(session, job_id)
    return {
        "mode": PROCESSING_MODE_MASK,
        "artifacts": {
            str(item["artifact_type"]): _artifact_response_item(user_id, item)
            for item in artifacts
        },
    }


async def _enrich_mask_payload(session, user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    if (
        payload.get("processing_mode") == PROCESSING_MODE_MASK
        and payload.get("status") == "completed"
    ):
        payload["result"] = await _mask_result_with_artifact_tokens(
            session,
            user_id,
            int(payload["job_id"]),
        )
    return payload


async def _store_classification_upload(
    session,
    user_id: int,
    file: UploadFile,
    *,
    source: str,
    processing_mode: str = PROCESSING_MODE_CLASSIFICATION,
    external_user_id: Optional[str] = None,
    external_case_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    callback_url: Optional[str] = None,
    callback_token: Optional[str] = None,
) -> Any:
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
            processing_mode=processing_mode,
            source=source,
            external_user_id=external_user_id,
            external_case_id=external_case_id,
            idempotency_key=idempotency_key,
            callback_url=callback_url,
            callback_token=callback_token,
        )
    except IntegrityError:
        try:
            await Orm.delete_file_record_by_id(session, file_id)
        except Exception:
            pass
        try:
            s3 = get_minio_client()
            await asyncio.to_thread(delete_object, s3, BUCKET_NAME, object_key)
        except Exception:
            pass
        raise
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

    return db_request


def _required_integration_value(value: str, field: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field} обязателен")
    if len(cleaned) > 255:
        raise HTTPException(status_code=400, detail=f"{field} слишком длинный")
    return cleaned


def _optional_callback_url(value: Optional[str]) -> Optional[str]:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if len(cleaned) > 2048:
        raise HTTPException(status_code=400, detail="callback_url слишком длинный")
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="callback_url должен начинаться с http:// или https://",
        )
    return cleaned


def _optional_callback_token(value: Optional[str]) -> Optional[str]:
    cleaned = (value or "").strip()
    return cleaned or None


def _integration_creation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": payload["job_id"],
        "status": payload["status"],
        "processing_mode": payload.get("processing_mode", PROCESSING_MODE_CLASSIFICATION),
        "external_user_id": payload["external_user_id"],
        "external_case_id": payload["external_case_id"],
        "idempotency_key": payload["idempotency_key"],
    }


async def perform_upload(
    session,
    user_id: int,
    file: UploadFile,
    features_only: bool = False,
    source: str = "web",
    processing_mode: str = PROCESSING_MODE_CLASSIFICATION,
) -> Dict[str, Any]:
    processing_mode = normalize_processing_mode(processing_mode)
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

    db_request = await _store_classification_upload(
        session,
        user_id,
        file,
        source=source,
        processing_mode=processing_mode,
    )
    run_classification.delay(db_request.id, features_only)
    return {
        "job_id": db_request.id,
        "status": "pending",
        "processing_mode": processing_mode,
    }


async def perform_integration_upload(
    session,
    user_id: int,
    file: UploadFile,
    external_user_id: str,
    external_case_id: str,
    idempotency_key: str,
    features_only: bool = True,
    callback_url: Optional[str] = None,
    callback_token: Optional[str] = None,
    processing_mode: str = PROCESSING_MODE_CLASSIFICATION,
) -> Dict[str, Any]:
    processing_mode = normalize_processing_mode(processing_mode)
    if not await Orm.user_exists(session, user_id):
        raise HTTPException(
            status_code=404,
            detail="Пользователь не найден в базе.",
        )

    external_user_id = _required_integration_value(
        external_user_id, "external_user_id"
    )
    external_case_id = _required_integration_value(
        external_case_id, "external_case_id"
    )
    idempotency_key = _required_integration_value(
        idempotency_key, "idempotency_key"
    )
    callback_url = _optional_callback_url(callback_url)
    callback_token = _optional_callback_token(callback_token)

    existing = await Orm.get_integration_job_by_idempotency_key(
        session, user_id, idempotency_key
    )
    if existing:
        return _integration_creation_payload(existing)

    if await Orm.get_active_integration_classification_job(
        session, user_id, external_user_id
    ):
        raise HTTPException(
            status_code=429,
            detail=(
                "Для external_user_id уже выполняется классификация. "
                "Дождитесь завершения или запросите статус активного задания."
            ),
        )

    try:
        db_request = await _store_classification_upload(
            session,
            user_id,
            file,
            source="integration",
            external_user_id=external_user_id,
            external_case_id=external_case_id,
            idempotency_key=idempotency_key,
            callback_url=callback_url,
            callback_token=callback_token,
            processing_mode=processing_mode,
        )
    except IntegrityError:
        existing = await Orm.get_integration_job_by_idempotency_key(
            session, user_id, idempotency_key
        )
        if existing:
            return _integration_creation_payload(existing)
        raise

    run_classification.apply_async(
        args=(db_request.id, features_only),
        queue="classification_external",
    )
    return {
        "job_id": db_request.id,
        "status": "pending",
        "processing_mode": processing_mode,
        "external_user_id": external_user_id,
        "external_case_id": external_case_id,
        "idempotency_key": idempotency_key,
    }


async def active_job_payload(session, user_id: int) -> Optional[Dict[str, Any]]:
    payload = await Orm.get_user_active_classification_job(session, user_id)
    if not payload:
        return None
    payload = await _enrich_mask_payload(session, user_id, payload)
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
    payload = await _enrich_mask_payload(session, user_id, payload)
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


async def active_integration_job_payload(
    session, user_id: int, external_user_id: str
) -> Optional[Dict[str, Any]]:
    external_user_id = _required_integration_value(
        external_user_id, "external_user_id"
    )
    payload = await Orm.get_active_integration_classification_job(
        session, user_id, external_user_id
    )
    if payload:
        payload = await _enrich_mask_payload(session, user_id, payload)
    return payload


async def integration_job_payload(
    session, user_id: int, job_id: int
) -> Optional[Dict[str, Any]]:
    payload = await Orm.get_integration_classification_job(session, job_id, user_id)
    if payload:
        payload = await _enrich_mask_payload(session, user_id, payload)
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
        if (
            item.get("processing_mode") == PROCESSING_MODE_MASK
            and item.get("status") == "completed"
        ):
            item["result"] = await _mask_result_with_artifact_tokens(
                session,
                user_id,
                int(item.get("job_id") or 0),
            )
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


async def artifact_file_stream(session, token: str) -> StreamingResponse:
    user_id, artifact_id = verify_artifact_access_token(token)
    artifact = await Orm.get_artifact_for_user(session, user_id, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Артефакт не найден или доступ запрещён")
    object_key = str(artifact["file_name"]).lstrip("/")
    try:
        s3 = get_minio_client()
        body = await asyncio.to_thread(
            download_file_bytes,
            s3,
            artifact["bucket_name"],
            object_key,
        )
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

    return StreamingResponse(
        io.BytesIO(body),
        media_type=artifact["content_type"],
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ARTIFACT_FILENAMES.get(artifact["artifact_type"], "artifact")}"'
            )
        },
    )
