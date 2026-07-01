import asyncio
import hashlib
import io
import json
import mimetypes
import os
import secrets
import time
import zipfile
from typing import Optional

import httpx
from PIL import Image

from core.redis_client import get_redis
from core.minio_client import (
    BUCKET_NAME,
    create_bucket_if_not_exists,
    download_file_bytes,
    get_minio_client,
    object_key_for_stored_filename,
    upload_bytes_to_minio,
)
from services.description_service import (
    create_description_job,
    description_enabled,
    submit_description_classification,
)
from services.image_access import create_artifact_access_token
from src.config import settings
from src.database import async_engine, async_session_maker
from src.queries.orm import Orm
from workers.app import celery_app


PROCESSING_MODE_MASK = "mask"
ARTIFACT_FILENAMES = {
    "mask": "mask.png",
    "masked_image": "masked_image.png",
    "archive": "mask_results.zip",
}


@celery_app.task(name="workers.tasks.run_classification")
def run_classification(classification_id: int, features_only: bool = False) -> None:
    asyncio.run(_run_classification_async(classification_id, features_only))


def _error_payload(detail: object) -> str:
    return json.dumps({"detail": detail}, ensure_ascii=True)


def _stage_payload(stage: str, title: str, description: str) -> str:
    return json.dumps(
        {
            "stage": stage,
            "title": title,
            "description": description,
        },
        ensure_ascii=True,
    )


def _unexpected_processing_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    if "minio" in message or "s3" in message:
        return "Не удалось получить изображение из хранилища. Попробуйте повторить попытку позже."
    if "timeout" in message:
        return "Обработка изображения заняла слишком много времени. Попробуйте повторить попытку позже."
    return "Не удалось завершить обработку изображения. Попробуйте повторить попытку позже."


def _ml_service_error_message(exc: httpx.HTTPError) -> str:
    request = getattr(exc, "request", None)
    url = str(request.url) if request else ""
    text = str(exc).lower()
    if any(token in text for token in ["4 channels", "alpha channel", "cannot identify image file"]):
        return (
            "Не удалось обработать изображение. "
            "Загрузите файл в формате JPEG или PNG без прозрачности."
        )
    if url.endswith("/mask"):
        return (
            "Не удалось построить маску для изображения. "
            "Попробуйте другое изображение или повторите попытку позже."
        )
    if url.endswith("/classify"):
        return (
            "Не удалось выполнить классификацию изображения. "
            "Попробуйте повторить попытку позже."
        )
    return "Сервис обработки изображения временно недоступен. Попробуйте позже."


def _description_service_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    if any(
        token in message
        for token in [
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "failed to resolve",
            "getaddrinfo",
            "connection refused",
            "all connection attempts failed",
            "connecterror",
        ]
    ):
        return "Сервис клинического описания временно недоступен. Попробуйте позже."
    if "timeout" in message:
        return (
            "Сервис клинического описания не ответил вовремя. "
            "Попробуйте повторить попытку позже."
        )
    return "Не удалось получить клиническое описание. Попробуйте позже."


def _description_response_fields(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}

    fields = {"description_result": payload}

    description = payload.get("description")
    if isinstance(description, str):
        fields["description"] = description

    important_labels = payload.get("important_labels")
    if isinstance(important_labels, list):
        fields["important_labels"] = important_labels

    bucketed_labels = payload.get("bucketed_labels")
    if isinstance(bucketed_labels, list):
        fields["bucketed_labels"] = bucketed_labels

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        fields["error"] = error

    features_only = payload.get("features_only")
    if isinstance(features_only, bool):
        fields["features_only"] = features_only

    return fields


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _artifact_key(user_id: int, job_id: int, artifact_type: str) -> str:
    ext = "zip" if artifact_type == "archive" else "png"
    return f"artifacts/{user_id}/{job_id}/{artifact_type}.{ext}"


def _artifact_result_item(user_id: int, artifact: dict) -> dict:
    token = None
    try:
        token = create_artifact_access_token(user_id, int(artifact["id"]))
    except RuntimeError:
        token = None
    artifact_type = str(artifact["artifact_type"])
    return {
        "token": token,
        "filename": ARTIFACT_FILENAMES.get(artifact_type, artifact_type),
        "content_type": artifact["content_type"],
        "size_bytes": artifact["size_bytes"],
        "checksum_sha256": artifact["checksum_sha256"],
    }


def _mask_artifact_result(user_id: int, artifacts: list[dict]) -> dict:
    return {
        "mode": PROCESSING_MODE_MASK,
        "artifacts": {
            str(item["artifact_type"]): _artifact_result_item(user_id, item)
            for item in artifacts
        },
    }


def _build_mask_artifacts(
    user_id: int,
    job_id: int,
    source_file_name: str,
    image_bytes: bytes,
    mask_bytes: bytes,
) -> list[dict]:
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask.size != source.size:
        raise ValueError("Размер маски не совпадает с размером исходного изображения")

    binary_mask = mask.point(lambda px: 255 if px > 0 else 0)
    black = Image.new("RGB", source.size, (0, 0, 0))
    masked_image = Image.composite(source, black, binary_mask)

    mask_png = _png_bytes(binary_mask)
    masked_png = _png_bytes(masked_image)

    artifact_blobs = {
        "mask": {
            "filename": ARTIFACT_FILENAMES["mask"],
            "content_type": "image/png",
            "data": mask_png,
        },
        "masked_image": {
            "filename": ARTIFACT_FILENAMES["masked_image"],
            "content_type": "image/png",
            "data": masked_png,
        },
    }
    manifest = {
        "job_id": job_id,
        "source_filename": source_file_name,
        "artifacts": [
            {
                "artifact_type": artifact_type,
                "filename": item["filename"],
                "content_type": item["content_type"],
                "size_bytes": len(item["data"]),
                "checksum_sha256": _sha256_hex(item["data"]),
            }
            for artifact_type, item in artifact_blobs.items()
        ],
    }

    archive_out = io.BytesIO()
    with zipfile.ZipFile(archive_out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(ARTIFACT_FILENAMES["mask"], mask_png)
        zf.writestr(ARTIFACT_FILENAMES["masked_image"], masked_png)
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=True, indent=2),
        )
    artifact_blobs["archive"] = {
        "filename": ARTIFACT_FILENAMES["archive"],
        "content_type": "application/zip",
        "data": archive_out.getvalue(),
    }

    return [
        {
            "artifact_type": artifact_type,
            "file_name": _artifact_key(user_id, job_id, artifact_type),
            "bucket_name": BUCKET_NAME,
            "content_type": item["content_type"],
            "size_bytes": len(item["data"]),
            "checksum_sha256": _sha256_hex(item["data"]),
            "data": item["data"],
        }
        for artifact_type, item in artifact_blobs.items()
    ]


def _upload_mask_artifacts(s3_client, artifacts: list[dict]) -> None:
    create_bucket_if_not_exists(s3_client, BUCKET_NAME)
    for artifact in artifacts:
        upload_bytes_to_minio(
            s3_client,
            artifact["bucket_name"],
            artifact["file_name"],
            artifact["data"],
            artifact["content_type"],
        )


_CLASSIFICATION_THROTTLE_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, member)
  redis.call('PEXPIRE', key, window)
  return 0
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if oldest[2] == nil then
  return 1000
end
return math.max(1, tonumber(oldest[2]) + window - now)
"""


def _classification_rate_limit_wait_ms(
    limit: Optional[int] = None,
    key: str = "classification:global:start_rl",
) -> int:
    limit = max(
        1,
        int(
            limit
            if limit is not None
            else settings.CLASSIFICATION_GLOBAL_RATE_LIMIT_PER_MINUTE
        ),
    )
    now_ms = int(time.time() * 1000)
    window_ms = 60_000
    member = f"{now_ms}:{secrets.token_urlsafe(8)}"
    wait_ms = get_redis().eval(
        _CLASSIFICATION_THROTTLE_LUA,
        1,
        key,
        now_ms,
        window_ms,
        limit,
        member,
    )
    return int(wait_ms or 0)


async def _wait_for_global_classification_slot(
    processing_mode: str = "classification",
) -> None:
    if processing_mode == PROCESSING_MODE_MASK:
        limit = int(settings.MASK_GLOBAL_RATE_LIMIT_PER_MINUTE)
        key = "classification:global:mask_start_rl"
    else:
        limit = int(settings.CLASSIFICATION_GLOBAL_RATE_LIMIT_PER_MINUTE)
        key = "classification:global:start_rl"

    while True:
        wait_ms = await asyncio.to_thread(
            _classification_rate_limit_wait_ms,
            limit,
            key,
        )
        if wait_ms <= 0:
            return
        await asyncio.sleep(max(wait_ms / 1000.0, 0.1))


def _callback_error_from_result(result: object) -> object:
    if isinstance(result, dict) and "detail" in result:
        return result["detail"]
    return result


async def _send_integration_callback_if_configured(
    session,
    classification_id: int,
) -> None:
    payload = await Orm.get_integration_callback_payload(session, classification_id)
    if not payload or not payload.get("callback_url"):
        return
    if payload.get("status") not in {"completed", "error"}:
        return

    result = payload.get("result")
    body = {
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "processing_mode": payload.get("processing_mode") or "classification",
        "external_user_id": payload.get("external_user_id"),
        "external_case_id": payload.get("external_case_id"),
        "idempotency_key": payload.get("idempotency_key"),
        "result": result if payload.get("status") == "completed" else None,
        "error": None
        if payload.get("status") == "completed"
        else _callback_error_from_result(result),
    }

    headers = {}
    callback_token = payload.get("callback_token")
    if callback_token:
        headers["X-Callback-Token"] = str(callback_token)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.post(
                str(payload["callback_url"]),
                json=body,
                headers=headers,
            )
            response.raise_for_status()
        await Orm.update_classification_callback_status(session, classification_id, "sent")
    except Exception as exc:
        await Orm.update_classification_callback_status(
            session,
            classification_id,
            "failed",
            callback_last_error=str(exc)[:2000],
        )


async def _update_terminal_classification_status(
    session,
    classification_id: int,
    status: str,
    result: str,
) -> None:
    await Orm.update_classification_status(
        session,
        classification_id,
        status,
        result=result,
    )
    await _send_integration_callback_if_configured(session, classification_id)


async def _request_mask(
    client: httpx.AsyncClient,
    file_name: str,
    body: bytes,
    content_type: str,
) -> bytes:
    response = await client.post(
        f"{settings.ML_URL.rstrip('/')}/mask",
        files={"file": (file_name, body, content_type)},
    )
    response.raise_for_status()
    return response.content


async def _request_classification(
    client: httpx.AsyncClient,
    file_name: str,
    body: bytes,
    content_type: str,
    mask_bytes: bytes,
) -> dict:
    response = await client.post(
        f"{settings.ML_URL.rstrip('/')}/classify",
        files={
            "file": (file_name, body, content_type),
            "mask": (f"{os.path.splitext(file_name)[0]}_mask.png", mask_bytes, "image/png"),
        },
    )
    response.raise_for_status()
    return response.json()


def _http_error_detail(exc: httpx.HTTPStatusError) -> object:
    if exc.response.status_code >= 500:
        return _ml_service_error_message(exc)
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail
        return body
    except Exception:
        return f"HTTP {exc.response.status_code}"


async def _run_classification_async(
    classification_id: int, features_only: bool = False
) -> None:
    try:
        async with async_session_maker() as session:
            await Orm.update_classification_status(
                session,
                classification_id,
                "processing",
                result=_stage_payload(
                    "preparing",
                    "Подготовка изображения",
                    "Проверяем файл и подготавливаем данные для анализа.",
                ),
            )

            meta = await Orm.get_classification_file_meta(session, classification_id)
            if not meta:
                await _update_terminal_classification_status(
                    session,
                    classification_id,
                    "error",
                    _error_payload("Запись классификации не найдена"),
                )
                return

            file_name = meta["file_name"]
            bucket = meta["bucket_name"]
            object_key = object_key_for_stored_filename(file_name)

            try:
                s3 = get_minio_client()
                image_bytes = await asyncio.to_thread(
                    download_file_bytes, s3, bucket, object_key
                )
            except Exception as exc:
                await _update_terminal_classification_status(
                    session,
                    classification_id,
                    "error",
                    _error_payload(
                        "Не удалось получить изображение из хранилища. Повторите попытку позже."
                    ),
                )
                return

            content_type = (
                mimetypes.guess_type(os.path.basename(file_name))[0]
                or "application/octet-stream"
            )
            image_part_name = os.path.basename(file_name)
            processing_mode = str(meta.get("processing_mode") or "classification")
            user_id = int(meta["user_id"])
            description_job_id = str(classification_id)
            description_registered = False

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                    await _wait_for_global_classification_slot(processing_mode)
                    await Orm.update_classification_status(
                        session,
                        classification_id,
                        "processing",
                        result=_stage_payload(
                            "mask",
                            "Построение маски",
                            "Выделяем область новообразования на изображении.",
                        ),
                    )
                    try:
                        mask_bytes = await _request_mask(
                            client,
                            image_part_name,
                            image_bytes,
                            content_type,
                        )
                    except httpx.HTTPStatusError as exc:
                        await _update_terminal_classification_status(
                            session,
                            classification_id,
                            "error",
                            _error_payload(_http_error_detail(exc)),
                        )
                        return
                    except httpx.RequestError as exc:
                        await _update_terminal_classification_status(
                            session,
                            classification_id,
                            "error",
                            _error_payload(_ml_service_error_message(exc)),
                        )
                        return

                    if processing_mode == PROCESSING_MODE_MASK:
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "processing",
                            result=_stage_payload(
                                "finalizing",
                                "Формирование результата",
                                "Готовим маску, маскированное изображение и архив.",
                            ),
                        )
                        try:
                            artifact_blobs = await asyncio.to_thread(
                                _build_mask_artifacts,
                                user_id,
                                classification_id,
                                image_part_name,
                                image_bytes,
                                mask_bytes,
                            )
                            s3 = get_minio_client()
                            await asyncio.to_thread(
                                _upload_mask_artifacts,
                                s3,
                                artifact_blobs,
                            )
                            artifact_rows = await Orm.upsert_classification_artifacts(
                                session,
                                classification_id,
                                [
                                    {
                                        key: value
                                        for key, value in artifact.items()
                                        if key != "data"
                                    }
                                    for artifact in artifact_blobs
                                ],
                            )
                            await _update_terminal_classification_status(
                                session,
                                classification_id,
                                "completed",
                                json.dumps(
                                    _mask_artifact_result(user_id, artifact_rows),
                                    ensure_ascii=True,
                                ),
                            )
                        except Exception as exc:
                            await _update_terminal_classification_status(
                                session,
                                classification_id,
                                "error",
                                _error_payload(
                                    "Не удалось сформировать артефакты маски. "
                                    "Попробуйте повторить обработку позже."
                                ),
                            )
                        return

                    if description_enabled() and not features_only:
                        await Orm.upsert_description_job(
                            session,
                            classification_result_id=classification_id,
                            service_job_id=description_job_id,
                            status="pending",
                            features_only=features_only,
                            callback_sent=False,
                        )
                        try:
                            description_response = await create_description_job(
                                client,
                                job_id=description_job_id,
                                image_name=image_part_name,
                                image_bytes=image_bytes,
                                image_content_type=content_type,
                                mask_name=f"{os.path.splitext(image_part_name)[0]}_mask.png",
                                mask_bytes=mask_bytes,
                                features_only=features_only,
                            )
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status=str(description_response.get("status") or "received"),
                                callback_sent=False,
                                **_description_response_fields(description_response),
                                features_only=features_only,
                            )
                            description_registered = True
                        except httpx.HTTPStatusError as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=_description_service_error_message(exc),
                                features_only=features_only,
                                callback_sent=False,
                            )
                        except Exception as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=_description_service_error_message(exc),
                                features_only=features_only,
                                callback_sent=False,
                            )

                    await Orm.update_classification_status(
                        session,
                        classification_id,
                        "processing",
                        result=_stage_payload(
                            "classification",
                            "Анализ признаков",
                            "Определяем визуальные признаки и строим классификацию.",
                        ),
                    )
                    try:
                        result = await _request_classification(
                            client,
                            image_part_name,
                            image_bytes,
                            content_type,
                            mask_bytes,
                        )
                    except httpx.HTTPStatusError as exc:
                        await _update_terminal_classification_status(
                            session,
                            classification_id,
                            "error",
                            _error_payload(_http_error_detail(exc)),
                        )
                        return
                    except httpx.RequestError as exc:
                        await _update_terminal_classification_status(
                            session,
                            classification_id,
                            "error",
                            _error_payload(_ml_service_error_message(exc)),
                        )
                        return

                    await Orm.update_classification_status(
                        session,
                        classification_id,
                        "processing",
                        result=_stage_payload(
                            "finalizing",
                            "Формирование результата",
                            "Собираем итог анализа и подготавливаем ответ.",
                        ),
                    )
                    await _update_terminal_classification_status(
                        session,
                        classification_id,
                        "completed",
                        json.dumps(result, ensure_ascii=True),
                    )

                    if description_enabled() and description_registered:
                        try:
                            description_response = await submit_description_classification(
                                client,
                                job_id=description_job_id,
                                payload=result,
                            )
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status=str(
                                    description_response.get("status") or "classification_ready"
                                ),
                                **_description_response_fields(description_response),
                                callback_sent=False,
                            )
                        except httpx.HTTPStatusError as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=_description_service_error_message(exc),
                                callback_sent=False,
                            )
                        except Exception as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=_description_service_error_message(exc),
                                callback_sent=False,
                            )
            except Exception as exc:
                await _update_terminal_classification_status(
                    session,
                    classification_id,
                    "error",
                    _error_payload(_unexpected_processing_error_message(exc)),
                )
    finally:
        await async_engine.dispose()
