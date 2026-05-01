import asyncio
import json
import mimetypes
import os

import httpx

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
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=_error_payload("Запись классификации не найдена"),
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
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=_error_payload(
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
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "error",
                            result=_error_payload(_http_error_detail(exc)),
                        )
                        return
                    except httpx.RequestError as exc:
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "error",
                            result=_error_payload(_ml_service_error_message(exc)),
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
                                description_result=description_response,
                                features_only=features_only,
                                callback_sent=False,
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
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "error",
                            result=_error_payload(_http_error_detail(exc)),
                        )
                        return
                    except httpx.RequestError as exc:
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "error",
                            result=_error_payload(_ml_service_error_message(exc)),
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
                    await Orm.update_classification_status(
                        session,
                        classification_id,
                        "completed",
                        result=json.dumps(result, ensure_ascii=True),
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
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=_error_payload(_unexpected_processing_error_message(exc)),
                )
    finally:
        await async_engine.dispose()
