"""Зависимости FastAPI для проверки состояния аккаунта."""
from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jwt_auth import get_current_user_id
from src.database import get_db
from src.models import User


async def require_verified_email_user_id(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
) -> int:
    """Доступ к классификатору только после подтверждения email."""
    stmt = select(User.email_verified).where(User.id == user_id)
    verified = (await session.execute(stmt)).scalar_one_or_none()
    if verified is None:
        raise HTTPException(
            status_code=404,
            detail="Пользователь не найден. Выйдите из аккаунта и войдите снова.",
        )
    if not verified:
        raise HTTPException(
            status_code=403,
            detail=(
                "Подтвердите адрес email, чтобы загружать изображения в классификатор. "
                "Письмо можно отправить повторно в личном кабинете."
            ),
        )
    return user_id
