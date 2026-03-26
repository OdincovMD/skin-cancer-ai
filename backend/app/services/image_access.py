import base64
import hashlib
import hmac
import json
import time
from typing import Tuple

from fastapi import HTTPException

from src.config import settings


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def create_image_access_token(user_id: int, file_name: str) -> str:
    secret = (settings.IMAGE_ACCESS_SIGNING_SECRET or "").strip()
    if len(secret) < 16:
        raise RuntimeError(
            "IMAGE_ACCESS_SIGNING_SECRET должен быть задан в .env (не короче 16 символов)"
        )
    ttl = max(60, int(settings.IMAGE_ACCESS_TOKEN_TTL_SEC))
    payload = {
        "u": user_id,
        "f": file_name,
        "e": int(time.time()) + ttl,
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body_b64 = _b64url_encode(body)
    sig = hmac.new(secret.encode("utf-8"), body_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{body_b64}.{sig_b64}"


def verify_image_access_token(token: str) -> Tuple[int, str]:
    secret = (settings.IMAGE_ACCESS_SIGNING_SECRET or "").strip()
    if len(secret) < 16:
        raise HTTPException(
            status_code=503,
            detail="Сервер не настроен для выдачи изображений по ссылке",
        )
    try:
        body_b64, sig_b64 = token.strip().split(".", 1)
        sig = _b64url_decode(sig_b64)
        expected = hmac.new(
            secret.encode("utf-8"), body_b64.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("signature")
        payload = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        if int(time.time()) > int(payload["e"]):
            raise ValueError("expired")
        return int(payload["u"]), str(payload["f"])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=403,
            detail="Недействительная или просроченная ссылка",
        ) from None
