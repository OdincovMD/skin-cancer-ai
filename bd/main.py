# main.py
from minio_client import get_minio_client, is_file_in_minio, upload_file_to_minio
from postgres_client import get_postgres_session, is_file_in_db, insert_file_record, create_files_table
from redis_client import upload_and_cache_file, get_file_from_cache

BUCKET_NAME = "example-bucket"
file_name = "uploads/test.txt"  # Убедитесь, что файл с этим именем существует в рабочей директории

# Получаем подключения
s3_client = get_minio_client()
session, engine = get_postgres_session()

# Создаем таблицу (если её нет)
files_table = create_files_table(session, engine)

# Проверка наличия файла в MinIO
if is_file_in_minio(s3_client, BUCKET_NAME, file_name):
    print(f"Файл {file_name} уже существует в MinIO.")
else:
    # Загружаем файл в MinIO
    # upload_file_to_minio(s3_client, BUCKET_NAME, file_name)
    upload_and_cache_file(file_name, BUCKET_NAME)

# Проверка наличия записи в базе данных
if is_file_in_db(session, files_table, file_name):
    print(f"Файл {file_name} уже записан в базе данных.")
else:
    # Добавляем запись в базу данных
    insert_file_record(session, files_table, file_name, BUCKET_NAME)

# Проверка кэша через 5 секунд
from time import sleep
# from clear_all import clear_postgres_table, clear_all_minio_buckets

sleep(5)
get_file_from_cache(file_name)

sleep(60)
get_file_from_cache(file_name)