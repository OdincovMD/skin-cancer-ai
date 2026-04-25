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
    try:
        return exc.response.json()
    except Exception:
        return f"HTTP {exc.response.status_code}"


async def _run_classification_async(
    classification_id: int, features_only: bool = False
) -> None:
    try:
        async with async_session_maker() as session:
            await Orm.update_classification_status(
                session, classification_id, "processing"
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
                    result=_error_payload(f"Ошибка чтения из MinIO: {exc}"),
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
                    mask_bytes = await _request_mask(
                        client,
                        image_part_name,
                        image_bytes,
                        content_type,
                    )

                    if features_only and not description_enabled():
                        await Orm.update_classification_status(
                            session,
                            classification_id,
                            "error",
                            result=_error_payload(
                                "Description service is not configured."
                            ),
                        )
                        return

                    if description_enabled():
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
                            if features_only:
                                return
                        except httpx.HTTPStatusError as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=f"Description service register failed: {_http_error_detail(exc)}",
                                features_only=features_only,
                                callback_sent=False,
                            )
                            if features_only:
                                await Orm.update_classification_status(
                                    session,
                                    classification_id,
                                    "error",
                                    result=_error_payload(_http_error_detail(exc)),
                                )
                                return
                        except Exception as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=f"Description service register failed: {exc}",
                                features_only=features_only,
                                callback_sent=False,
                            )
                            if features_only:
                                await Orm.update_classification_status(
                                    session,
                                    classification_id,
                                    "error",
                                    result=_error_payload(str(exc)),
                                )
                                return

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
                                error=f"Description classification submit failed: {_http_error_detail(exc)}",
                                callback_sent=False,
                            )
                        except Exception as exc:
                            await Orm.upsert_description_job(
                                session,
                                classification_result_id=classification_id,
                                service_job_id=description_job_id,
                                status="error",
                                error=f"Description classification submit failed: {exc}",
                                callback_sent=False,
                            )
            except Exception as exc:
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=_error_payload(str(exc)),
                )
    finally:
        await async_engine.dispose()
