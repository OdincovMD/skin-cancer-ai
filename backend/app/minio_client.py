# minio_client.py
import boto3
import os
from botocore.exceptions import ClientError

MINIO_URL = os.getenv("MINIO_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASSWORD")

# Подключение к MinIO
def get_minio_client():
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        s3_client.list_buckets()  # Попытка выполнить запрос
        print("Соединение с MinIO установлено.")
        return s3_client
    except Exception as e:
        print(f"Ошибка подключения к MinIO: {e}")
        exit(1)

def is_file_in_minio(s3_client, bucket_name, file_name):
    """Проверяет наличие файла в MinIO."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except ClientError:
        return False

def create_bucket_if_not_exists(s3_client, bucket_name):
    # Create a bucket in MinIO
    try:
        s3 = get_minio_client()

        # Check if the bucket already exists
        response = s3.list_buckets()
        existing_buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]

        if bucket_name in existing_buckets:
            print(f"Bucket {bucket_name} already exists")
        else:
            s3.create_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} created successfully")
    except NoCredentialsError:
        print("Credentials not available")

def upload_file_to_minio(s3_client, bucket_name, file_name):
    """Загружает файл в MinIO."""
    try:
        s3_client.upload_file(Filename=file_name, Bucket=bucket_name, Key=file_name)
        print(f"Файл {file_name} успешно загружен в MinIO.")
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_name} в MinIO: {e}")



