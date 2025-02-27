import boto3
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

# Конфигурация MinIO
MINIO_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "MINIO_ROOT_USER"
MINIO_SECRET_KEY = "MINIO_ROOT_PASSWORD"
BUCKET_NAME = "example-bucket"  # Укажите имя бакета для очистки. Если хотите очистить все бакеты, используйте цикл ниже.

# Конфигурация PostgreSQL
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_DB = "postgres"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"

# Подключение к MinIO
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    # print("Соединение с MinIO установлено.")
except Exception as e:
    print(f"Ошибка подключения к MinIO: {e}")
    exit(1)

# Подключение к PostgreSQL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    # print("Соединение с PostgreSQL установлено.")
    connection.close()
except OperationalError as e:
    print(f"Ошибка подключения к PostgreSQL: {e}")
    exit(1)

# def reset_sequence(engine):
#     try:
#         with engine.connect() as connection:
#             # Проверка имени последовательности
#             result = connection.execute(
#                 text("SELECT pg_get_serial_sequence('files', 'id')"))
#             sequence_name = result.scalar()  # Получаем имя последовательности
#
#             if sequence_name:
#                 # Сброс последовательности
#                 connection.execute(text(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1"))
#                 print(f"Счетчик последовательности {sequence_name} успешно сброшен.")
#             else:
#                 print("Последовательность для столбца 'id' не найдена.")
#     except Exception as e:
#         print(f"Ошибка при сбросе последовательности: {e}")

# Удаление всех данных из PostgreSQL
def clear_postgres_table():
    """Удаляет все записи из таблицы files."""
    metadata = MetaData()
    files_table = Table("files", metadata, autoload_with=engine)
    try:
        with engine.connect() as conn:
            conn.execute(files_table.delete())
            conn.commit()
        print("Все данные из таблицы 'files' успешно удалены.")
        # reset_sequence(engine)
    except Exception as e:
        print(f"Ошибка при удалении данных из PostgreSQL: {e}")

# Удаление всех объектов из MinIO
def clear_minio_bucket(bucket_name):
    """Удаляет все объекты из указанного бакета MinIO."""
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in objects:
            for obj in objects["Contents"]:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
            print(f"Все файлы из бакета '{bucket_name}' успешно удалены.")
        else:
            print(f"Бакет '{bucket_name}' уже пуст.")
    except Exception as e:
        print(f"Ошибка при удалении файлов из MinIO бакета '{bucket_name}': {e}")

def clear_all_minio_buckets():
    """Удаляет файлы из всех бакетов MinIO."""
    try:
        buckets = s3_client.list_buckets()
        for bucket in buckets["Buckets"]:
            clear_minio_bucket(bucket["Name"])
    except Exception as e:
        print(f"Ошибка при очистке бакетов MinIO: {e}")


clear_postgres_table()
clear_all_minio_buckets()