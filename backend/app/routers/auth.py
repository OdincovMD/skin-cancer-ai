import asyncio
import mimetypes
import secrets
from datetime import datetime, timezone

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
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
)
from src.queries.orm import Orm

router = APIRouter(tags=["auth"])


@router.post("/signup")
async def signup(
    user_data: UserSignUp, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.register_user(
            session,
            firstName=user_data.firstName,
            lastName=user_data.lastName,
            login=user_data.login,
            email=user_data.email,
            password=user_data.password,
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                    "email_verified": None,
                },
                "error": result,
            }

        try:
            token = create_access_token(result["id"])
        except RuntimeError as e:
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                    "email_verified": None,
                },
                "error": str(e),
            }

        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
                "email_verified": result["email_verified"],
            },
            "error": None,
            "requires_email_verification": True,
            "access_token": token,
            "token_type": "bearer",
            "verification_resend_after_seconds": result.get(
                "verification_resend_after_seconds",
                settings.VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC,
            ),
        }

    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
                "email_verified": None,
            },
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


@router.post("/signin")
async def signin_user(
    credentials: Credentials, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.signin_user(
            session,
            login=credentials.login,
            password=credentials.password,
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                    "email_verified": None,
                },
                "error": result,
            }

        try:
            token = create_access_token(result["id"])
        except RuntimeError as e:
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                    "email_verified": None,
                },
                "error": str(e),
            }

        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
                "email_verified": result["email_verified"],
            },
            "error": None,
            "access_token": token,
            "token_type": "bearer",
            "verification_resend_after_seconds": result.get(
                "verification_resend_after_seconds", 0
            ),
        }

    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
                "email_verified": None,
            },
            "error": f"Ошибка при входе: {str(e)}",
        }
