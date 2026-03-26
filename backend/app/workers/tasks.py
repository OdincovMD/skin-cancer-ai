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
from src.config import settings
from src.database import async_engine, async_session_maker
from src.queries.orm import Orm

from workers.app import celery_app

@celery_app.task(name="workers.tasks.run_classification")
def run_classification(classification_id: int) -> None:
    asyncio.run(_run_classification_async(classification_id))


async def _run_classification_async(classification_id: int) -> None:
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
                    result=json.dumps(
                        {"detail": "Запись классификации не найдена"},
                        ensure_ascii=True,
                    ),
                )
                return

            file_name = meta["file_name"]
            bucket = meta["bucket_name"]
            object_key = object_key_for_stored_filename(file_name)

            try:
                s3 = get_minio_client()
                body = await asyncio.to_thread(
                    download_file_bytes, s3, bucket, object_key
                )
            except Exception as e:
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=json.dumps(
                        {"detail": f"Ошибка чтения из MinIO: {e}"},
                        ensure_ascii=True,
                    ),
                )
                return

            content_type = (
                mimetypes.guess_type(os.path.basename(file_name))[0]
                or "application/octet-stream"
            )
            ml_url = f"{settings.ML_URL.rstrip('/')}/uploadfile"
            ml_part_name = os.path.basename(file_name)

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                    response = await client.post(
                        ml_url,
                        files={"file": (ml_part_name, body, content_type)},
                    )
                    response.raise_for_status()
                    result = response.json()
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "completed",
                    result=json.dumps(result, ensure_ascii=True),
                )
            except httpx.HTTPStatusError as e:
                detail = f"ML HTTP {e.response.status_code}"
                try:
                    detail = e.response.json()
                except Exception:
                    pass
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=json.dumps({"detail": detail}, ensure_ascii=True),
                )
            except Exception as e:
                await Orm.update_classification_status(
                    session,
                    classification_id,
                    "error",
                    result=json.dumps({"detail": str(e)}, ensure_ascii=True),
                )
    finally:
        await async_engine.dispose()
