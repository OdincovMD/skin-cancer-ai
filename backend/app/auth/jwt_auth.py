from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import settings

security = HTTPBearer(auto_error=False)


def create_access_token(
    user_id: int, expires_minutes: Optional[int] = None
) -> Tuple[str, str]:
    secret = (settings.JWT_SECRET or "").strip()
    if len(secret) < 16:
        raise RuntimeError(
            "JWT_SECRET должен быть задан в .env (не короче 16 символов)"
        )
    ttl_minutes = (
        max(1, int(expires_minutes))
        if expires_minutes is not None
        else max(1, int(settings.JWT_EXPIRE_MINUTES))
    )
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ttl_minutes
    )
    payload = {"sub": str(user_id), "exp": int(expire.timestamp())}
    token = jwt.encode(payload, secret, algorithm=settings.JWT_ALGORITHM)
    return token, expire.isoformat()


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    secret = (settings.JWT_SECRET or "").strip()
    if len(secret) < 16:
        raise HTTPException(
            status_code=503,
            detail="Сервер не настроен для выдачи JWT (JWT_SECRET)",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Требуется заголовок Authorization: Bearer <token>",
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            secret,
            algorithms=[settings.JWT_ALGORITHM],
        )
        sub = payload.get("sub")
        if sub is None:
            raise ValueError("missing sub")
        return int(sub)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Недействительный или просроченный токен",
        ) from None
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Недействительный токен",
        ) from None
