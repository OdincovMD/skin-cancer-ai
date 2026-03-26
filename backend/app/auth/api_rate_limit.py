import asyncio
from typing import Optional

from fastapi import HTTPException

from core.redis_client import get_redis
from src.config import settings

# Окно «в минуту» (секунды TTL ключа счётчика).
_API_V1_RATE_WINDOW_SEC = 60

# Атомарно: INCR и выставление EXPIRE (и починка ключа без TTL после сбоев).
_API_V1_RL_LUA = """
local c = redis.call('INCR', KEYS[1])
local window = tonumber(ARGV[1])
if c == 1 then
  redis.call('EXPIRE', KEYS[1], window)
else
  local t = redis.call('TTL', KEYS[1])
  if t == -1 then
    redis.call('EXPIRE', KEYS[1], window)
  end
end
return c
"""


def _enforce_api_v1_rate_limit(user_id: int) -> Optional[str]:
    limit = max(1, int(settings.API_V1_RATE_LIMIT_PER_MINUTE))
    r = get_redis()
    key = f"api_v1:rl:{user_id}"
    n = int(
        r.eval(
            _API_V1_RL_LUA,
            1,
            key,
            _API_V1_RATE_WINDOW_SEC,
        )
    )
    if n > limit:
        return (
            f"Превышен лимит API: не более {limit} запросов в минуту. "
            "Повторите позже."
        )
    return None


async def enforce_api_v1_rate_limit(user_id: int) -> None:
    detail = await asyncio.to_thread(_enforce_api_v1_rate_limit, user_id)
    if detail:
        raise HTTPException(status_code=429, detail=detail)
