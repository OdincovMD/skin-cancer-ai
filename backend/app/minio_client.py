# minio_client.py
import os
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

MINIO_URL = (os.getenv("MINIO_URL") or "").rstrip("/")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER") or ""
MINIO_SECRET_KEY = os.getenv("MINIO_PASSWORD") or ""

BUCKET_NAME = "bucket"
# Ключ в S3 совпадает с относительным путём при загрузке (см. uploadfile): uploads/<имя_файла>
OBJECT_KEY_PREFIX = "uploads"


def _minio_client_config() -> Config:
    return Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    )


@lru_cache(maxsize=1)
def get_minio_client():
    if not MINIO_URL:
        raise RuntimeError("MINIO_URL не задан в окружении")
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError("MINIO_USER и MINIO_PASSWORD должны быть заданы в окружении")
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_minio_client_config(),
        )
        s3_client.list_buckets()
        print("Соединение с MinIO установлено.")
        return s3_client
    except ClientError as e:
        raise RuntimeError(f"Ошибка MinIO (S3 API): {e}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка подключения к MinIO: {e}") from e


def is_file_in_minio(s3_client, bucket_name, file_name):
    """Проверяет наличие файла в MinIO."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except ClientError:
        return False


def create_bucket_if_not_exists(s3_client, bucket_name):
    try:
        response = s3_client.list_buckets()
        existing_buckets = [
            bucket["Name"] for bucket in response.get("Buckets", [])
        ]

        if bucket_name in existing_buckets:
            print(f"Bucket {bucket_name} already exists")
        else:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} created successfully")
    except NoCredentialsError:
        print("Credentials not available")


def upload_file_to_minio(s3_client, bucket_name, file_name):
    """Загружает файл в MinIO."""
    try:
        s3_client.upload_file(
            Filename=file_name, Bucket=bucket_name, Key=file_name
        )
        print(f"Файл {file_name} успешно загружен в MinIO.")
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_name} в MinIO: {e}")


def download_file_bytes(s3_client, bucket_name: str, object_key: str) -> bytes:
    """Скачивает объект из MinIO/S3 по ключу."""
    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    return obj["Body"].read()


def upload_bytes_to_minio(
    s3_client,
    bucket_name: str,
    object_key: str,
    data: bytes,
    content_type: str,
) -> None:
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=data,
        ContentType=content_type,
    )


def delete_object(s3_client, bucket_name: str, object_key: str) -> None:
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
    except ClientError:
        pass


def object_key_for_stored_filename(stored_file_name: str) -> str:
    """Ключ объекта в бакете для имени файла из БД (без ведущего uploads/ в имени)."""
    name = (stored_file_name or "").lstrip("/")
    if name.startswith(f"{OBJECT_KEY_PREFIX}/"):
        return name
    return f"{OBJECT_KEY_PREFIX}/{name}"
