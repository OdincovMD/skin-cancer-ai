import asyncio
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from src.config import settings


async def send_verification_email(to_addr: str, verification_token: str) -> None:
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        raise RuntimeError("SMTP не настроен (SMTP_HOST / SMTP_USER).")

    base = settings.FRONTEND_PUBLIC_URL.rstrip("/")
    link = f"{base}/verify-email?token={verification_token}"
    text_body = (
        "Подтвердите регистрацию, перейдя по ссылке:\n\n"
        f"{link}\n\n"
        "Если вы не регистрировались, проигнорируйте это письмо."
    )
    html_body = (
        f'<p>Подтвердите регистрацию:</p>'
        f'<p><a href="{link}">{link}</a></p>'
        f"<p>Если вы не регистрировались, проигнорируйте это письмо.</p>"
    )

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
