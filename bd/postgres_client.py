# postgres_client.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_DB = "postgres"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"

# Подключение к PostgreSQL
def get_postgres_session():
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    try:
        engine = create_engine(DATABASE_URL)
        connection = engine.connect()
        print("Соединение с PostgreSQL установлено.")
        connection.close()
        Session = sessionmaker(bind=engine)
        return Session(), engine  # Возвращаем и сессию, и движок для использования в других местах
    except OperationalError as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        exit(1)

# Определение таблицы
def create_files_table(session, engine):
    metadata = MetaData()
    files_table = Table(
        "files",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("file_name", String(255), nullable=False),
        Column("bucket_name", String(255), nullable=False),
        Column("file_path", Text, nullable=False),
    )
    metadata.create_all(engine)
    return files_table  # Возвращаем таблицу

def is_file_in_db(session, files_table, file_name):
    """Проверяет наличие записи о файле в базе данных."""
    result = session.execute(files_table.select().where(files_table.c.file_name == file_name)).fetchone()
    return result is not None

def insert_file_record(session, files_table, file_name, bucket_name):
    """Добавляет запись о файле в базу данных."""
    try:
        session.execute(
            files_table.insert().values(
                file_name=file_name,
                bucket_name=bucket_name,
                file_path=f"s3://{bucket_name}/{file_name}"
            )
        )
        session.commit()
        print(f"Информация о файле {file_name} успешно добавлена в базу данных.")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при добавлении записи о файле {file_name}: {e}")
