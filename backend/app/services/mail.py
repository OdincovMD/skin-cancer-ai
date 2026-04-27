import asyncio
import html
import smtplib
from functools import lru_cache
from pathlib import Path
from email.message import EmailMessage
from email.utils import formataddr

from src.config import settings

# Шаблоны лежат в корне приложения: app/templates/
_TEMPLATE_FILE = (
    Path(__file__).resolve().parents[1] / "templates" / "verification_email.html"
)
_RESET_TEMPLATE_FILE = (
    Path(__file__).resolve().parents[1] / "templates" / "password_reset_email.html"
)


@lru_cache(maxsize=1)
def _verification_email_template() -> str:
    return _TEMPLATE_FILE.read_text(encoding="utf-8")


def _render_verification_email_html(
    app_name: str, verification_url: str, ttl_hours: int
) -> str:
    safe_name = html.escape(app_name.strip() or "Skin Cancer AI")
    safe_url = html.escape(verification_url, quote=True)
    h = str(max(1, int(ttl_hours)))
    return (
        _verification_email_template()
        .replace("__APP_NAME__", safe_name)
        .replace("__VERIFICATION_URL__", safe_url)
        .replace("__TTL_HOURS__", html.escape(h))
    )


@lru_cache(maxsize=1)
def _password_reset_email_template() -> str:
    return _RESET_TEMPLATE_FILE.read_text(encoding="utf-8")


def _render_password_reset_email_html(
    app_name: str, reset_url: str, ttl_hours: int
) -> str:
    safe_name = html.escape(app_name.strip() or "Skin Cancer AI")
    safe_url = html.escape(reset_url, quote=True)
    h = str(max(1, int(ttl_hours)))
    return (
        _password_reset_email_template()
        .replace("__APP_NAME__", safe_name)
        .replace("__RESET_URL__", safe_url)
        .replace("__TTL_HOURS__", html.escape(h))
    )


async def send_password_reset_email(to_addr: str, reset_token: str) -> None:
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        raise RuntimeError("SMTP не настроен (SMTP_HOST / SMTP_USER).")

    base = settings.FRONTEND_PUBLIC_URL.rstrip("/")
    link = f"{base}/reset-password?token={reset_token}"
    app_name = (settings.MAIL_FROM_NAME or "").strip() or "Skin Cancer AI"
    ttl = max(1, int(settings.PASSWORD_RESET_TOKEN_TTL_HOURS))

    text_body = (
        f"Здравствуйте!\n\n"
        f"Мы получили запрос на сброс пароля для вашего аккаунта в «{app_name}».\n\n"
        f"Перейдите по ссылке, чтобы задать новый пароль:\n\n"
        f"{link}\n\n"
        f"Срок действия ссылки — около {ttl} ч.\n\n"
        f"Если вы не запрашивали сброс пароля, просто проигнорируйте это письмо.\n\n"
        f"— {app_name}"
    )
    html_body = _render_password_reset_email_html(app_name, link, ttl)

    from_addr = settings.MAIL_FROM or settings.SMTP_USER
    from_header = formataddr((settings.MAIL_FROM_NAME, from_addr))

    def _send_sync() -> None:
        msg = EmailMessage()
        msg["Subject"] = "Сброс пароля"
        msg["From"] = from_header
        msg["To"] = to_addr
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")

        if settings.SMTP_USE_SSL:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
                smtp.ehlo()
                if settings.SMTP_USE_TLS:
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)

    await asyncio.to_thread(_send_sync)


async def send_verification_email(to_addr: str, verification_token: str) -> None:
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        raise RuntimeError("SMTP не настроен (SMTP_HOST / SMTP_USER).")

    base = settings.FRONTEND_PUBLIC_URL.rstrip("/")
    link = f"{base}/verify-email?token={verification_token}"
    app_name = (settings.MAIL_FROM_NAME or "").strip() or "Skin Cancer AI"
    ttl = max(1, int(settings.EMAIL_VERIFICATION_TOKEN_TTL_HOURS))

    text_body = (
        f"Здравствуйте!\n\n"
        f"Подтвердите регистрацию в «{app_name}», перейдя по ссылке:\n\n"
        f"{link}\n\n"
        f"Срок действия ссылки — около {ttl} ч. После истечения можно запросить новое письмо в личном кабинете.\n\n"
        f"Если вы не регистрировались, проигнорируйте это письмо.\n\n"
        f"— {app_name}"
    )
    html_body = _render_verification_email_html(app_name, link, ttl)

    from_addr = settings.MAIL_FROM or settings.SMTP_USER
    from_header = formataddr((settings.MAIL_FROM_NAME, from_addr))

    def _send_sync() -> None:
        msg = EmailMessage()
        msg["Subject"] = "Подтверждение email"
        msg["From"] = from_header
        msg["To"] = to_addr
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")

        if settings.SMTP_USE_SSL:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
                smtp.ehlo()
                if settings.SMTP_USE_TLS:
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)

    await asyncio.to_thread(_send_sync)
