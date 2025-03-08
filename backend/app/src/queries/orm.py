from src.models import Base, File, ClassificationResults, User 
from src.database import session_factory, sync_engine
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from typing import Union, Dict


class SyncOrm:
    @staticmethod
    def create_tables():
        """Создание и сброс всех таблиц в базе данных."""
        Base.metadata.drop_all(sync_engine)
        Base.metadata.create_all(sync_engine)

    @staticmethod
    def insert_file_record(file_name: str, bucket_name: str):
        """Добавляет запись о файле в базу данных."""
        with session_factory() as session:
            try:
                file = File(
                    file_name=file_name,
                    bucket_name=bucket_name,
                    file_path=f"s3://{bucket_name}/{file_name}"
                )
                session.add(file)
                session.commit()
                print(f"Информация о файле {file_name} успешно добавлена в базу данных.")
            except IntegrityError:
                session.rollback()
                print(f"Ошибка при добавлении записи о файле {file_name}: файл уже существует.")
            except Exception as e:
                session.rollback()
                print(f"Ошибка при добавлении записи о файле {file_name}: {e}")

    @staticmethod
    def create_classification_request(user_id: int, file_id: int, status: str = "completed", result: str = None):
        """
        Создает запись о запросе на классификацию.
        """
        with session_factory() as session:
            try:
                db_request = ClassificationResults(
                    user_id=user_id,
                    file_id=file_id,
                    status=status,
                    result=result
                )
                session.add(db_request)
                session.commit()
                session.refresh(db_request)
                return db_request
            except Exception as e:
                session.rollback()
                print(f"Ошибка при создании запроса на классификацию: {e}")
                raise

    @staticmethod
    def get_classification_requests(user_id: int, limit: int = 3):
        """
        Возвращает последние запросы на классификацию для пользователя.
        """
        with session_factory() as session:
            query = (
                select(ClassificationResults)
                .where(ClassificationResults.user_id == user_id)
                .order_by(ClassificationResults.request_date.desc())
                .limit(limit)
            )
            result = session.execute(query)
            return result.scalars().all()

    @staticmethod
    def get_file_id_by_name(file_name: str) -> int:
        """
        Возвращает ID файла по его имени.
        """
        with session_factory() as session:
            query = select(File.file_id).where(File.file_name == file_name)
            result = session.execute(query).scalar()
            if not result:
                raise ValueError(f"Файл {file_name} не найден в базе данных")
            return result
        
    @staticmethod
    def register_user(lastName: str, firstName: str, login: str, email: str, password: str) -> Union[str, Dict[str, Union[int, str]]]:
        """
        Регистрирует нового пользователя в базе данных.

        Возвращает:
        - В случае успеха: словарь с данными пользователя (id, lastName, firstName, login, email).
        - В случае ошибки: строку с описанием ошибки.
        """
        with session_factory() as session:
            try:
                user = User(
                    lastName=lastName,
                    firstName=firstName,
                    login=login,
                    email=email,
                    password=password
                )
                session.add(user)
                session.commit()

                return {
                    "id": user.id,
                    "lastName": user.lastName,
                    "firstName": user.firstName,
                    "login": user.login,
                    "email": user.email,
                }
            except IntegrityError as e:
                session.rollback()
                if "login" in str(e):
                    return f"Ошибка: Пользователь с логином {login} уже существует."
                elif "email" in str(e):
                    return f"Ошибка: Пользователь с email {email} уже существует."
                else:
                    return f"Ошибка при регистрации пользователя: {e}"
            except Exception as e:
                session.rollback()
                return f"Ошибка при регистрации пользователя: {e}"