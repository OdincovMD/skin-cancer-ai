import asyncio
import json
import mimetypes
import os
import secrets
import time

import httpx

from core.redis_client import get_redis
from core.minio_client import (
    download_file_bytes,
    get_minio_client,
    object_key_for_stored_filename,
)
from services.description_service import (
    create_description_job,
    description_enabled,
    submit_description_classification,
)
from src.config import settings
from src.database import async_engine, async_session_maker
from src.queries.orm import Orm
from workers.app import celery_app


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


def _classification_rate_limit_wait_ms() -> int:
    limit = max(1, int(settings.CLASSIFICATION_GLOBAL_RATE_LIMIT_PER_MINUTE))
    now_ms = int(time.time() * 1000)
    window_ms = 60_000
    member = f"{now_ms}:{secrets.token_urlsafe(8)}"
    wait_ms = get_redis().eval(
        _CLASSIFICATION_THROTTLE_LUA,
        1,
        "classification:global:start_rl",
        now_ms,
        window_ms,
        limit,
        member,
    )
    return int(wait_ms or 0)


async def _wait_for_global_classification_slot() -> None:
    while True:
        wait_ms = await asyncio.to_thread(_classification_rate_limit_wait_ms)
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
            description_job_id = str(classification_id)
            description_registered = False

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                    await _wait_for_global_classification_slot()
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
