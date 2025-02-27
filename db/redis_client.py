import os
import redis
from minio_client import get_minio_client, upload_file_to_minio

# Подключение к Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_TTL = 60  # Время жизни ключа в секундах (10 минут)

try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()  # Проверка подключения
    print("Соединение с Redis установлено.")
except Exception as e:
    print(f"Ошибка подключения к Redis: {e}")
    exit(1)

def upload_and_cache_file(file_name, bucket_name):
    """Загружает файл в MinIO, кэширует в Redis и удаляет из локальной директории."""
    s3_client = get_minio_client()

    try:
        # Загрузка файла в MinIO
        upload_file_to_minio(s3_client, bucket_name, file_name)

        # Кэширование файла в Redis
        with open(file_name, "rb") as file_data:
            redis_client.setex(file_name, REDIS_TTL, file_data.read())
        print(f"Файл {file_name} добавлен в Redis с временем хранения = {REDIS_TTL} секунд.")

        # Удаление локального файла
        os.remove(file_name)
        print(f"Файл {file_name} удалён из локальной директории.")

    except Exception as e:
        print(f"Ошибка при обработке файла {file_name}: {e}")

def get_file_from_cache(file_name):
    """Пытается получить файл из Redis."""
    try:
        file_data = redis_client.get(file_name)
        if file_data:
            print(f"Файл {file_name} найден в Redis.")
            return file_data
        else:
            print(f"Файл {file_name} не найден в Redis.")
            return None
    except Exception as e:
        print(f"Ошибка при попытке получить файл {file_name} из Redis: {e}")
        return None
