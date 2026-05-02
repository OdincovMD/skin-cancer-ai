from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from jose import JWTError, jwt

from src.config import settings


VK_PROVIDER = "vk"


class VkIdError(Exception):
    pass


def vk_id_enabled() -> bool:
    return bool(settings.VK_ID_APP_ID.strip() and settings.VK_ID_REDIRECT_URI.strip())


def _vk_timeout() -> httpx.Timeout:
    timeout = max(5, int(settings.VK_ID_TIMEOUT_SEC))
    return httpx.Timeout(timeout)


def _vk_base_url() -> str:
    return settings.VK_ID_BASE_URL.rstrip("/")


def _link_secret() -> str:
    secret = (settings.JWT_SECRET or "").strip()
    if len(secret) < 16:
        raise RuntimeError("JWT_SECRET должен быть задан для VK linking flow.")
    return secret


async def exchange_vk_code(
    *,
    code: str,
    code_verifier: str,
    device_id: str,
    redirect_uri: str,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    if not vk_id_enabled():
        raise VkIdError("VK ID не настроен на сервере.")

    form: Dict[str, str] = {
        "grant_type": "authorization_code",
        "client_id": settings.VK_ID_APP_ID.strip(),
        "code": code,
        "code_verifier": code_verifier,
        "device_id": device_id,
        "redirect_uri": redirect_uri,
    }
    if state:
        form["state"] = state

    async with httpx.AsyncClient(timeout=_vk_timeout()) as client:
        response = await client.post(
            f"{_vk_base_url()}/oauth2/auth",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if response.status_code >= 400:
        raise VkIdError("Invalid VK code / exchange failed")
    payload = response.json()
    if payload.get("error"):
        raise VkIdError("Invalid VK code / exchange failed")
    return payload


async def fetch_vk_user_info(*, access_token: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=_vk_timeout()) as client:
        response = await client.post(
            f"{_vk_base_url()}/oauth2/user_info",
            data={
                "client_id": settings.VK_ID_APP_ID.strip(),
                "access_token": access_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if response.status_code >= 400:
        raise VkIdError("Не удалось получить профиль VK ID.")
    payload = response.json()
    user = payload.get("user")
    if not isinstance(user, dict):
        raise VkIdError("Не удалось получить профиль VK ID.")
    return user


def _clean_optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def extract_vk_id_token_claims(id_token: Optional[str]) -> Dict[str, Any]:
    token = _clean_optional_string(id_token)
    if not token:
        return {}
    try:
        claims = jwt.get_unverified_claims(token)
    except JWTError:
        return {}
    return claims if isinstance(claims, dict) else {}


def normalize_vk_user(
    raw_user: Dict[str, Any],
    *,
    token_payload: Optional[Dict[str, Any]] = None,
    id_token_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    token_payload = token_payload or {}
    id_token_claims = id_token_claims or {}

    provider_user_id = (
        raw_user.get("user_id")
        or token_payload.get("user_id")
        or id_token_claims.get("user_id")
        or id_token_claims.get("sub")
    )
    email = (
        raw_user.get("email")
        or token_payload.get("email")
        or id_token_claims.get("email")
    )
    first_name = (
        raw_user.get("first_name")
        or id_token_claims.get("first_name")
        or id_token_claims.get("given_name")
    )
    last_name = (
        raw_user.get("last_name")
        or id_token_claims.get("last_name")
        or id_token_claims.get("family_name")
    )
    avatar_url = (
        raw_user.get("avatar")
        or raw_user.get("avatar_url")
        or id_token_claims.get("picture")
    )

    return {
        "provider_user_id": str(provider_user_id) if provider_user_id is not None else None,
        "email": _clean_optional_string(email).lower() if _clean_optional_string(email) else None,
        "first_name": _clean_optional_string(first_name),
        "last_name": _clean_optional_string(last_name),
        "avatar_url": _clean_optional_string(avatar_url),
    }


def create_vk_link_token(payload: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "purpose": "vk_link",
        "exp": int((now + timedelta(minutes=15)).timestamp()),
        **payload,
    }
    return jwt.encode(claims, _link_secret(), algorithm=settings.JWT_ALGORITHM)


def decode_vk_link_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            _link_secret(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise VkIdError("VK linking session expired or invalid") from exc
    if payload.get("purpose") != "vk_link":
        raise VkIdError("VK linking session expired or invalid")
    return payload
