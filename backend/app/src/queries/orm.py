import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ClassificationResults, File, User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Orm:
    @staticmethod
    async def insert_file_record(
        session: AsyncSession, file_name: str, bucket_name: str
    ) -> None:
        try:
            file = File(
                file_name=file_name,
                bucket_name=bucket_name,
                file_path=f"s3://{bucket_name}/{file_name}",
            )
            session.add(file)
            await session.commit()
            print(f"Информация о файле {file_name} успешно добавлена в базу данных.")
        except IntegrityError:
            await session.rollback()
            print(
                f"Ошибка при добавлении записи о файле {file_name}: файл уже существует."
            )
        except Exception as e:
            await session.rollback()
            print(f"Ошибка при добавлении записи о файле {file_name}: {e}")

    @staticmethod
    async def create_classification_request(
        session: AsyncSession,
        user_id: int,
        file_id: int,
        status: str = "completed",
        result: str = None,
    ) -> ClassificationResults:
        try:
            db_request = ClassificationResults(
                user_id=user_id,
                file_id=file_id,
                status=status,
                result=result,
            )
            session.add(db_request)
            await session.commit()
            await session.refresh(db_request)
            return db_request
        except Exception as e:
            await session.rollback()
            print(f"Ошибка при создании запроса на классификацию: {e}")
            raise

    @staticmethod
    async def update_classification_status(
        session: AsyncSession,
        job_id: int,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        try:
            row = await session.get(ClassificationResults, job_id)
            if not row:
                return
            row.status = status
            if result is not None:
                row.result = result
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"Ошибка при обновлении статуса классификации {job_id}: {e}")
            raise

    @staticmethod
    async def get_classification_file_meta(
        session: AsyncSession, job_id: int
    ) -> Optional[Dict[str, str]]:
        stmt = (
            select(ClassificationResults, File)
            .join(File, ClassificationResults.file_id == File.file_id)
            .where(ClassificationResults.id == job_id)
        )
        result = await session.execute(stmt)
        row = result.first()
        if not row:
            return None
        _cr, f = row
        return {"file_name": f.file_name, "bucket_name": f.bucket_name}

    @staticmethod
    async def count_user_active_classifications(
        session: AsyncSession, user_id: int
    ) -> int:
        stmt = select(func.count(ClassificationResults.id)).where(
            ClassificationResults.user_id == user_id,
            ClassificationResults.status.in_(["pending", "processing"]),
        )
        result = await session.execute(stmt)
        n = result.scalar_one()
        return int(n or 0)

    @staticmethod
    async def get_user_active_classification_job(
        session: AsyncSession, user_id: int
    ) -> Optional[Dict[str, Any]]:
        stmt = (
            select(ClassificationResults, File)
            .join(File, ClassificationResults.file_id == File.file_id)
            .where(
                ClassificationResults.user_id == user_id,
                ClassificationResults.status.in_(["pending", "processing"]),
            )
            .order_by(ClassificationResults.request_date.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.first()
        if not row:
            return None
        cr, f = row
        return {
            "job_id": cr.id,
            "status": cr.status,
            "file_name": f.file_name,
        }

    @staticmethod
    async def get_classification_job(
        session: AsyncSession, job_id: int, user_id: int
    ) -> Optional[Dict[str, Any]]:
        stmt = select(ClassificationResults).where(
            ClassificationResults.id == job_id,
            ClassificationResults.user_id == user_id,
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if not row:
            return None
        parsed: Any = None
        if row.result:
            try:
                parsed = json.loads(row.result)
            except json.JSONDecodeError:
                parsed = {"raw": row.result}
        return {
            "job_id": row.id,
            "status": row.status,
            "result": parsed,
        }

    @staticmethod
    async def get_classification_requests(
        session: AsyncSession, user_id: int, limit: int = 3
    ) -> List[Dict[str, Union[str, datetime]]]:
        stmt = (
            select(
                ClassificationResults.request_date,
                File.file_name,
                File.bucket_name,
                ClassificationResults.status,
                ClassificationResults.result,
            )
            .join(File, ClassificationResults.file_id == File.file_id)
            .where(ClassificationResults.user_id == user_id)
            .order_by(ClassificationResults.request_date.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [
            {
                "request_date": row.request_date,
                "file_name": row.file_name,
                "bucket_name": row.bucket_name,
                "status": row.status,
                "result": row.result,
            }
            for row in result.all()
        ]

    @staticmethod
    async def get_bucket_for_user_file(
        session: AsyncSession, user_id: int, stored_file_name: str
    ) -> Optional[str]:
        """Бакет MinIO, если у пользователя есть запись классификации с этим file_name."""
        stmt = (
            select(File.bucket_name)
            .join(ClassificationResults, ClassificationResults.file_id == File.file_id)
            .where(
                ClassificationResults.user_id == user_id,
                File.file_name == stored_file_name,
            )
            .limit(1)
        )
        row = (await session.execute(stmt)).first()
        return row[0] if row else None

    @staticmethod
    async def user_exists(session: AsyncSession, user_id: int) -> bool:
        stmt = select(User.id).where(User.id == user_id).limit(1)
        return (await session.execute(stmt)).first() is not None

    @staticmethod
    async def get_file_id_by_name(session: AsyncSession, file_name: str) -> int:
        stmt = select(File.file_id).where(File.file_name == file_name)
        result = await session.execute(stmt)
        fid = result.scalar_one_or_none()
        if not fid:
            raise ValueError(f"Файл {file_name} не найден в базе данных")
        return fid

    @staticmethod
    async def register_user(
        session: AsyncSession,
        lastName: str,
        firstName: str,
        login: str,
        email: str,
        password: str,
    ) -> Union[str, Dict[str, Union[int, str]]]:
        try:
            hashed_password = await asyncio.to_thread(pwd_context.hash, password)

            user = User(
                lastName=lastName,
                firstName=firstName,
                login=login,
                email=email,
                password=hashed_password,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

            return {
                "id": user.id,
                "lastName": user.lastName,
                "firstName": user.firstName,
                "login": user.login,
                "email": user.email,
            }
        except IntegrityError as e:
            await session.rollback()
            err_txt = str(e.orig) if getattr(e, "orig", None) else str(e)
            if "ix_users_login" in err_txt or "login" in err_txt.lower():
                return "Ошибка: Пользователь с таким логином уже зарегистрирован."
            if "ix_users_email" in err_txt or "email" in err_txt.lower():
                return "Ошибка: Пользователь с такой почтой уже зарегистрирован."
            return f"Ошибка при регистрации пользователя: {e}"
        except Exception as e:
            await session.rollback()
            return f"Ошибка при регистрации пользователя: {e}"

    @staticmethod
    async def signin_user(
        session: AsyncSession, login: str, password: str
    ) -> Union[Dict[str, Union[int, str]], str]:
        try:
            stmt = select(User).where(User.login == login)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                return "Пользователь не зарегистрирован."

            ok = await asyncio.to_thread(pwd_context.verify, password, user.password)
            if not ok:
                return "Неверный пароль."

            return {
                "id": user.id,
                "lastName": user.lastName,
                "firstName": user.firstName,
                "email": user.email,
            }
        except Exception as e:
            return f"Ошибка при входе: {e}"
