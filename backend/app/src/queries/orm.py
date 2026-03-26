import asyncio
import hashlib
import json
import math
import secrets
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from mail_smtp import send_verification_email
from src.config import settings
from src.models import ClassificationResults, File, User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _verification_resend_remaining_seconds(user: User) -> int:
    if user.email_verified:
        return 0
    last = user.verification_email_last_sent_at
    if last is None:
        return 0
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    elapsed = (now - last).total_seconds()
    cooldown = max(1, int(settings.VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC))
    remaining = cooldown - elapsed
    if remaining <= 0:
        return 0
    return int(math.ceil(remaining))


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
    ) -> Union[str, Dict[str, Union[int, str, bool]]]:
        try:
            hashed_password = await asyncio.to_thread(pwd_context.hash, password)

            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=settings.EMAIL_VERIFICATION_TOKEN_TTL_HOURS
            )

            user = User(
                lastName=lastName,
                firstName=firstName,
                login=login,
                email=email,
                password=hashed_password,
                email_verified=False,
                email_verification_token=token_hash,
                email_verification_expires_at=expires_at,
            )
            session.add(user)
            await session.flush()

            try:
                await send_verification_email(user.email, raw_token)
            except Exception:
                await session.rollback()
                return (
                    "Не удалось отправить письмо подтверждения. "
                    "Проверьте настройки SMTP или попробуйте позже."
                )

            user.verification_email_last_sent_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(user)

            return {
                "id": user.id,
                "lastName": user.lastName,
                "firstName": user.firstName,
                "login": user.login,
                "email": user.email,
                "email_verified": False,
                "requires_email_verification": True,
                "verification_resend_after_seconds": _verification_resend_remaining_seconds(
                    user
                ),
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
                "email_verified": bool(user.email_verified),
                "verification_resend_after_seconds": _verification_resend_remaining_seconds(
                    user
                ),
            }
        except Exception as e:
            return f"Ошибка при входе: {e}"

    @staticmethod
    async def verify_email_by_token(
        session: AsyncSession, raw_token: str
    ) -> Optional[str]:
        if not raw_token or not raw_token.strip():
            return "Токен не указан."
        token_hash = hashlib.sha256(raw_token.strip().encode()).hexdigest()
        stmt = select(User).where(User.email_verification_token == token_hash)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return "Ссылка недействительна или уже использована."
        now = datetime.now(timezone.utc)
        exp = user.email_verification_expires_at
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp is not None and exp < now:
            return "Срок действия ссылки истёк. Зарегистрируйтесь снова или обратитесь в поддержку."
        user.email_verified = True
        user.email_verification_token = None
        user.email_verification_expires_at = None
        await session.commit()
        return None

    @staticmethod
    async def resend_verification_email(
        session: AsyncSession, user_id: int
    ) -> Union[None, str, Dict[str, Union[str, int]]]:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return "Пользователь не найден."
        if user.email_verified:
            return "Этот адрес уже подтверждён."
        remaining = _verification_resend_remaining_seconds(user)
        if remaining > 0:
            return {
                "error": f"Повторная отправка возможна через {remaining} с.",
                "retry_after_seconds": remaining,
            }
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        user.email_verification_token = token_hash
        user.email_verification_expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.EMAIL_VERIFICATION_TOKEN_TTL_HOURS
        )
        await session.flush()
        try:
            await send_verification_email(user.email, raw_token)
        except Exception:
            await session.rollback()
            return (
                "Не удалось отправить письмо. Проверьте настройки SMTP или попробуйте позже."
            )
        user.verification_email_last_sent_at = datetime.now(timezone.utc)
        await session.commit()
        return None

    @staticmethod
    async def get_user_profile_fields(
        session: AsyncSession, user_id: int
    ) -> Optional[Dict[str, Union[int, str, bool]]]:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return None
        return {
            "id": user.id,
            "firstName": user.firstName,
            "lastName": user.lastName,
            "email": user.email,
            "email_verified": bool(user.email_verified),
            "verification_resend_after_seconds": _verification_resend_remaining_seconds(
                user
            ),
        }

    @staticmethod
    async def change_password(
        session: AsyncSession,
        user_id: int,
        current_password: str,
        new_password: str,
    ) -> Optional[str]:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return "Пользователь не найден."
        ok = await asyncio.to_thread(
            pwd_context.verify, current_password, user.password
        )
        if not ok:
            return "Неверный текущий пароль."
        if current_password == new_password:
            return "Новый пароль должен отличаться от текущего."
        user.password = await asyncio.to_thread(pwd_context.hash, new_password)
        await session.commit()
        return None

    @staticmethod
    async def update_user_profile(
        session: AsyncSession,
        user_id: int,
        firstName: Optional[str],
        lastName: Optional[str],
    ) -> Optional[Dict[str, str]]:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return None
        if firstName is not None:
            user.firstName = firstName
        if lastName is not None:
            user.lastName = lastName
        await session.commit()
        await session.refresh(user)
        return {"firstName": user.firstName, "lastName": user.lastName}

    @staticmethod
    async def get_profile_avatar_key(
        session: AsyncSession, user_id: int
    ) -> Optional[str]:
        stmt = select(User.profile_avatar_key).where(User.id == user_id)
        row = (await session.execute(stmt)).first()
        if not row:
            return None
        key = row[0]
        return str(key) if key else None

    @staticmethod
    async def set_profile_avatar_key(
        session: AsyncSession, user_id: int, object_key: str
    ) -> bool:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            return False
        user.profile_avatar_key = object_key
        await session.commit()
        return True
