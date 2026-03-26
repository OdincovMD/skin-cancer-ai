from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from auth.api_token import hash_api_token
from src.database import get_db
from src.queries.orm import Orm


async def get_user_id_from_api_key(
    session: AsyncSession = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> int:
    if not x_api_key or not str(x_api_key).strip():
        raise HTTPException(
            status_code=401,
            detail="Требуется заголовок X-API-Key",
        )
    h = hash_api_token(str(x_api_key))
    uid = await Orm.get_user_id_by_api_token_hash(session, h)
    if uid is None:
        raise HTTPException(
            status_code=401,
            detail="Недействительный или отозванный API-ключ",
        )
    return int(uid)
