import asyncio
import mimetypes
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from auth.api_token import generate_api_token, hash_api_token
from auth.deps import require_verified_email_user_id
from auth.jwt_auth import create_access_token, get_current_user_id
from core.minio_client import (
    BUCKET_NAME,
    create_bucket_if_not_exists,
    delete_object,
    download_file_bytes,
    get_minio_client,
    upload_bytes_to_minio,
)
from src.config import settings
from src.database import get_db
from src.schemas import (
    ChangePasswordBody,
    Credentials,
    ForgotPasswordBody,
    ResetPasswordBody,
    UpdateProfileBody,
    UserSignUp,
    VkExchangeBody,
    VkLinkConfirmBody,
)
from src.queries.orm import Orm
from services.vk_id import (
    VK_PROVIDER,
    VkIdError,
    create_vk_link_token,
    decode_vk_link_token,
    exchange_vk_code,
    extract_vk_id_token_claims,
    fetch_vk_user_info,
    normalize_vk_user,
)

router = APIRouter(tags=["auth"])


def _empty_user_data() -> Dict[str, Any]:
    return {
        "id": None,
        "firstName": None,
        "lastName": None,
        "email": None,
        "has_password": False,
        "email_verified": None,
    }


def _auth_success_response(
    result: Dict[str, Any],
    *,
    expires_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    token, token_expires_at = create_access_token(
        result["id"], expires_minutes=expires_minutes
    )
    return {
        "userData": {
            "id": result["id"],
            "firstName": result["firstName"],
            "lastName": result["lastName"],
            "email": result["email"],
            "has_password": result.get("has_password", False),
            "email_verified": result["email_verified"],
        },
        "error": None,
        "requires_email_verification": bool(
            result.get("requires_email_verification")
        ),
        "access_token": token,
        "access_token_expires_at": token_expires_at,
        "token_type": "bearer",
        "verification_resend_after_seconds": result.get(
            "verification_resend_after_seconds",
            settings.VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC,
        ),
    }


@router.post("/signup")
async def signup(
    user_data: UserSignUp, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.register_user(
            session,
            email=user_data.email,
            password=user_data.password,
        )

        if isinstance(result, str):
            return {
                "userData": _empty_user_data(),
                "error": result,
            }

        try:
            return _auth_success_response(result)
        except RuntimeError as e:
            return {
                "userData": _empty_user_data(),
                "error": str(e),
            }

    except Exception as e:
        return {
            "userData": _empty_user_data(),
            "error": f"Ошибка при регистрации пользователя: {str(e)}",
        }


@router.get("/me")
async def me(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    profile = await Orm.get_user_profile_fields(session, user_id)
    if not profile:
        return {
            "userData": None,
            "error": "Пользователь не найден.",
        }
    return {"userData": profile, "error": None}


@router.get("/me/api-token")
async def get_my_api_token_status(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    st = await Orm.get_api_token_status(session, user_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден.")
    created = st.get("created_at")
    return {
        "has_token": st["has_token"],
        "created_at": created.isoformat() if created else None,
        "display_label": "scai_••••••••" if st["has_token"] else None,
    }


@router.post("/me/api-token")
async def issue_my_api_token(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    raw = generate_api_token()
    h = hash_api_token(raw)
    now = datetime.now(timezone.utc)
    code = await Orm.issue_api_token(session, user_id, h, now)
    if code == "not_found":
        raise HTTPException(status_code=404, detail="Пользователь не найден.")
    if code == "not_verified":
        raise HTTPException(
            status_code=403,
            detail="Подтвердите email, чтобы выпустить API-ключ.",
        )
    if code == "already_exists":
        raise HTTPException(
            status_code=409,
            detail="Ключ уже выпущен. Используйте перевыпуск или отзовите ключ.",
        )
    return {"token": raw, "created_at": now.isoformat()}


@router.post("/me/api-token/rotate")
async def rotate_my_api_token(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    raw = generate_api_token()
    h = hash_api_token(raw)
    now = datetime.now(timezone.utc)
    code = await Orm.rotate_api_token(session, user_id, h, now)
    if code == "not_found":
        raise HTTPException(status_code=404, detail="Пользователь не найден.")
    if code == "not_verified":
        raise HTTPException(
            status_code=403,
            detail="Подтвердите email, чтобы управлять API-ключом.",
        )
    if code == "no_token":
        raise HTTPException(
            status_code=409,
            detail="Сначала выпустите ключ или он уже отозван.",
        )
    return {"token": raw, "created_at": now.isoformat()}


@router.delete("/me/api-token")
async def revoke_my_api_token(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(require_verified_email_user_id),
):
    code = await Orm.revoke_api_token(session, user_id)
    if code == "not_found":
        raise HTTPException(status_code=404, detail="Пользователь не найден.")
    return {"ok": True}


_AVATAR_MAX_BYTES = 5 * 1024 * 1024
_AVATAR_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


@router.patch("/me/profile")
async def patch_my_profile(
    body: UpdateProfileBody,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    updated = await Orm.update_user_profile(
        session,
        user_id,
        body.firstName,
        body.lastName,
    )
    if not updated:
        return {"error": "Пользователь не найден.", "userData": None}
    return {"error": None, "userData": updated}


@router.get("/me/avatar")
async def get_my_avatar(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    key = await Orm.get_profile_avatar_key(session, user_id)
    if not key:
        raise HTTPException(status_code=404, detail="Аватар не загружен")
    try:
        s3 = get_minio_client()
        data = await asyncio.to_thread(
            download_file_bytes, s3, BUCKET_NAME, key
        )
    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code") or ""
        if code in ("NoSuchKey", "404", "NotFound"):
            raise HTTPException(
                status_code=404, detail="Файл аватара не найден в хранилище"
            ) from e
        raise HTTPException(
            status_code=502, detail="Ошибка чтения из хранилища"
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    media = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(
        content=data,
        media_type=media,
        headers={"Cache-Control": "private, no-store"},
    )


@router.post("/me/avatar")
async def upload_my_avatar(
    file: UploadFile = File(),
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    if not await Orm.user_exists(session, user_id):
        return {"error": "Пользователь не найден."}
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in _AVATAR_TYPES:
        return {"error": "Допустимы изображения JPEG, PNG или WebP."}
    raw = await file.read()
    if len(raw) > _AVATAR_MAX_BYTES:
        return {"error": "Файл слишком большой (максимум 5 МБ)."}
    new_key = f"avatars/{user_id}/{secrets.token_hex(16)}.{_AVATAR_TYPES[ctype]}"
    s3 = get_minio_client()
    create_bucket_if_not_exists(s3, BUCKET_NAME)
    old_key = await Orm.get_profile_avatar_key(session, user_id)
    try:
        await asyncio.to_thread(
            upload_bytes_to_minio, s3, BUCKET_NAME, new_key, raw, ctype
        )
    except ClientError as e:
        return {"error": f"Не удалось сохранить файл: {e}"}
    ok = await Orm.set_profile_avatar_key(session, user_id, new_key)
    if not ok:
        await asyncio.to_thread(delete_object, s3, BUCKET_NAME, new_key)
        return {"error": "Пользователь не найден."}
    if old_key and old_key != new_key:
        await asyncio.to_thread(delete_object, s3, BUCKET_NAME, old_key)
    return {"error": None}


@router.post("/change-password")
async def change_password(
    body: ChangePasswordBody,
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    err = await Orm.change_password(
        session,
        user_id,
        body.current_password,
        body.new_password,
    )
    if err:
        return {"error": err}
    return {"error": None}


@router.post("/resend-verification-email")
async def resend_verification_email(
    session: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    out = await Orm.resend_verification_email(session, user_id)
    if out is None:
        return {
            "error": None,
            "verification_resend_after_seconds": settings.VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC,
        }
    if isinstance(out, dict):
        return {
            "error": out["error"],
            "retry_after_seconds": out["retry_after_seconds"],
        }
    return {"error": out}


@router.get("/verify-email")
async def verify_email(
    token: str = Query(..., description="Токен из письма"),
    session: AsyncSession = Depends(get_db),
):
    err = await Orm.verify_email_by_token(session, token)
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "error": None}


@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordBody,
    session: AsyncSession = Depends(get_db),
):
    err = await Orm.request_password_reset(session, body.email)
    if err:
        return {"error": err}
    return {"error": None}


@router.post("/reset-password")
async def reset_password(
    body: ResetPasswordBody,
    session: AsyncSession = Depends(get_db),
):
    err = await Orm.reset_password_by_token(session, body.token, body.new_password)
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "error": None}


@router.post("/vk/exchange")
async def vk_exchange(
    body: VkExchangeBody,
    session: AsyncSession = Depends(get_db),
):
    try:
        token_payload = await exchange_vk_code(
            code=body.code,
            code_verifier=body.code_verifier,
            device_id=body.device_id,
            redirect_uri=body.redirect_uri,
            state=body.state,
        )
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise HTTPException(status_code=401, detail="Invalid VK code / exchange failed")

        raw_user = await fetch_vk_user_info(access_token=access_token)
        id_token_claims = extract_vk_id_token_claims(token_payload.get("id_token"))
        vk_user = normalize_vk_user(
            raw_user,
            token_payload=token_payload,
            id_token_claims=id_token_claims,
        )
        provider_user_id = vk_user.get("provider_user_id")
        email = vk_user.get("email")
        if not provider_user_id:
            raise HTTPException(status_code=401, detail="Invalid VK code / exchange failed")

        identity = await Orm.get_user_identity(session, VK_PROVIDER, provider_user_id)
        if identity:
            user = await Orm.get_user_profile_fields(session, identity.user_id)
            if not user:
                raise HTTPException(status_code=500, detail="VK account already linked")
            expires_minutes = (
                settings.JWT_REMEMBER_ME_EXPIRE_MINUTES
                if body.remember_me
                else settings.JWT_EXPIRE_MINUTES
            )
            return _auth_success_response(user, expires_minutes=expires_minutes)

        existing_user = await Orm.get_user_by_email(session, email) if email else None
        if existing_user:
            link_token = create_vk_link_token(
                {
                    "provider": VK_PROVIDER,
                    "provider_user_id": provider_user_id,
                    "provider_email": email,
                    "first_name": vk_user.get("first_name"),
                    "last_name": vk_user.get("last_name"),
                    "avatar_url": vk_user.get("avatar_url"),
                }
            )
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "Existing email requires linking",
                    "requires_vk_link": True,
                    "vk_link_token": link_token,
                    "email": email,
                },
            )

        payload = await Orm.create_vk_user(
            session,
            email=email,
            first_name=vk_user.get("first_name"),
            last_name=vk_user.get("last_name"),
            provider=VK_PROVIDER,
            provider_user_id=provider_user_id,
        )
        expires_minutes = (
            settings.JWT_REMEMBER_ME_EXPIRE_MINUTES
            if body.remember_me
            else settings.JWT_EXPIRE_MINUTES
        )
        return _auth_success_response(payload, expires_minutes=expires_minutes)
    except VkIdError as exc:
        detail = str(exc)
        status = 422 if detail == "VK email not provided" else 401
        raise HTTPException(status_code=status, detail=detail) from exc


@router.post("/vk/link/confirm")
async def vk_link_confirm(
    body: VkLinkConfirmBody,
    session: AsyncSession = Depends(get_db),
):
    try:
        payload = decode_vk_link_token(body.vk_link_token)
    except VkIdError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    verified = await Orm.verify_user_password_for_linking(
        session,
        body.email,
        body.password,
    )
    if isinstance(verified, str):
        status = 401 if verified == "Password confirmation failed" else 422
        raise HTTPException(status_code=status, detail=verified)

    try:
        auth_payload = await Orm.link_identity_to_existing_user(
            session,
            user=verified,
            provider=str(payload.get("provider") or VK_PROVIDER),
            provider_user_id=str(payload.get("provider_user_id") or ""),
            provider_email=payload.get("provider_email"),
            first_name=payload.get("first_name"),
            last_name=payload.get("last_name"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        expires_minutes = (
            settings.JWT_REMEMBER_ME_EXPIRE_MINUTES
            if body.remember_me
            else settings.JWT_EXPIRE_MINUTES
        )
        return _auth_success_response(auth_payload, expires_minutes=expires_minutes)
    except RuntimeError as exc:
        return {"userData": _empty_user_data(), "error": str(exc)}


@router.post("/signin")
async def signin_user(
    credentials: Credentials, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.signin_user(
            session,
            email=credentials.email,
            password=credentials.password,
        )

        if isinstance(result, str):
            return {
                "userData": _empty_user_data(),
                "error": result,
            }

        try:
            expires_minutes = (
                settings.JWT_REMEMBER_ME_EXPIRE_MINUTES
                if credentials.remember_me
                else settings.JWT_EXPIRE_MINUTES
            )
            return _auth_success_response(result, expires_minutes=expires_minutes)
        except RuntimeError as e:
            return {
                "userData": _empty_user_data(),
                "error": str(e),
            }

    except Exception as e:
        return {
            "userData": _empty_user_data(),
            "error": f"Ошибка при входе: {str(e)}",
        }
