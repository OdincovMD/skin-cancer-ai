from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str

    REDIS_HOST: str = Field(default="redis")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)

    ML_URL: str = Field(default="http://ml:8000")

    DESCRIPTION_ENABLED: bool = Field(default=False)
    DESCRIPTION_SERVICE_URL: str = Field(default="http://description_service:8000")
    DESCRIPTION_SERVICE_API_TOKEN: str = Field(default="")
    DESCRIPTION_CALLBACK_API_TOKEN: str = Field(default="")

    # Подпись ссылок GET /history/image?token=... (защита от перебора user_id/file_name)
    IMAGE_ACCESS_SIGNING_SECRET: str = Field(default="")
    IMAGE_ACCESS_TOKEN_TTL_SEC: int = Field(default=3600)

    # Лимит запросов к /api/v1 на пользователя (окно 60 с, Redis)
    API_V1_RATE_LIMIT_PER_MINUTE: int = Field(default=5)

    # JWT (Authorization: Bearer) для API классификации и истории
    JWT_SECRET: str = Field(default="")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=240)
    JWT_REMEMBER_ME_EXPIRE_MINUTES: int = Field(default=240)

    # SMTP (верификация email); пустой SMTP_HOST — отправка отключена
    SMTP_HOST: str = Field(default="")
    SMTP_PORT: int = Field(default=587)
    SMTP_USE_TLS: bool = Field(default=True)
    SMTP_USE_SSL: bool = Field(default=False)
    SMTP_USER: str = Field(default="")
    SMTP_PASSWORD: str = Field(default="")
    MAIL_FROM: str = Field(default="")
    MAIL_FROM_NAME: str = Field(default="Skin Cancer AI")
    EMAIL_VERIFICATION_TOKEN_TTL_HOURS: int = Field(default=24)
    # Минимальный интервал между письмами верификации (повторная отправка)
    VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC: int = Field(default=120)
    # Сброс пароля
    PASSWORD_RESET_TOKEN_TTL_HOURS: int = Field(default=1)
    PASSWORD_RESET_COOLDOWN_SEC: int = Field(default=120)
    # Публичный URL фронта (как в браузере). В Docker через NGINX обычно :90, не :3000.
    FRONTEND_PUBLIC_URL: str = Field(default="http://localhost:90")

    @property
    def DATABASE_URL_async(self) -> str:
        user = quote_plus(self.DB_USER)
        password = quote_plus(self.DB_PASS)
        return (
            f"postgresql+asyncpg://{user}:{password}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/1"
    
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
