# Развёртывание Skin Cancer AI

Инструкция для оператора, который поднимает и обслуживает экземпляр системы.

## Требования

- Docker Engine 24+ и Docker Compose V2 (плагин `docker compose`).
- Минимум 4 ГБ ОЗУ на хосте (ML-сервис при загрузке весов занимает ~2 ГБ).
- Открытый порт **90** (NGINX) или тот, что вы укажете в `docker-compose.yml`.

## Быстрый запуск

```bash
cp .env.example .env
# Откройте .env и заполните обязательные секреты (см. ниже)
docker compose up --build -d
```

Compose-файл задаёт `name: skin`, поэтому контейнеры будут называться `skin-backend-1`, `skin-nginx-1` и т.д. Внутренние имена сервисов (`postgres`, `redis`, `minio`, `ml`, `backend`, `frontend`) остаются короткими.

### Порядок запуска

Docker Compose поднимает сервисы с учётом `depends_on` и `healthcheck`:

```
postgres, redis, minio  →  ml  →  backend, celery_worker  →  frontend  →  nginx
```

ML-сервис стартует дольше всех (`start_period: 600s`) из-за загрузки весов моделей. Не пугайтесь, если первые 2–3 минуты `docker compose ps` показывает `health: starting` для `ml`.

## Переменные окружения

Все переменные перечислены в `.env.example`. Ниже — группировка по назначению.

### База данных (PostgreSQL)

| Переменная | Назначение | Пример |
|------------|------------|--------|
| `DB_HOST` | Хост БД (внутри Compose — имя сервиса) | `postgres` |
| `DB_PORT` | Порт PostgreSQL | `5432` |
| `DB_USER` | Пользователь | `postgres` |
| `DB_PASS` | Пароль | задайте надёжный |
| `DB_NAME` | Имя базы | `postgres` |

Схема таблиц создаётся автоматически при первом запуске (`init_db`).

### Redis

| Переменная | Назначение | Пример |
|------------|------------|--------|
| `REDIS_HOST` | Хост Redis | `redis` |
| `REDIS_PORT` | Порт | `6379` |
| `REDIS_DB` | Номер БД | `0` |

Используется как брокер Celery, бекенд результатов и для rate limiting API v1.

### MinIO (S3-хранилище)

| Переменная | Назначение | Пример |
|------------|------------|--------|
| `MINIO_USER` | Логин root-пользователя MinIO | задайте |
| `MINIO_PASSWORD` | Пароль root-пользователя | задайте (мин. 8 символов) |
| `MINIO_URL` | Внутренний URL MinIO | `http://minio:9000` |

Консоль MinIO доступна на порту **1112** хоста (`http://localhost:1112`).

### Приложение

| Переменная | Назначение | Пример |
|------------|------------|--------|
| `BACKEND_URL` | Префикс API за NGINX | `/backend` |
| `ML_URL` | Внутренний URL ML-сервиса | `http://ml:8000` |
| `DOMAIN` | Домен (если используется) | `skin-cancer-ai.ru` |
| `HOST` | Передаётся в NGINX (опционально) | — |

### ML memory management

| Переменная | По умолчанию в Compose | Описание |
|------------|-------------------------|----------|
| `MODEL_CACHE_MEMORY_LIMIT_MB` | `4608` | Порог RSS ML-процесса, после которого начинается eviction не-core lazy-моделей |
| `MODEL_CACHE_IDLE_TTL_SECONDS` | `600` | Через сколько секунд простоя lazy-модель считается кандидатом на выгрузку |
| `ISOLATED_RARE_MODEL_WORKERS` | `1` | Число процессов для изолированных rare-path моделей; `1` даёт минимальный RAM footprint |

Для текущего профиля сервиса это разумные стартовые значения: cold start около `2.5 GiB`, обычный запрос около `3.1 GiB`, поэтому лимит `4.5 GiB` оставляет запас под всплески без возврата к `5-6 GiB` steady-state.
Эти значения удобно переопределять через `.env`; в `docker-compose.yml` оставлены только fallback-дефолты.

Подробное описание расследования по памяти, архитектурных изменений и фактических замеров вынесено в [docs/ML_MEMORY_PROFILE.md](ML_MEMORY_PROFILE.md).

### Description service (`img2txt`)

В проекте подразумевается интеграция с внешним сервисом описания изображений
[`OdincovMD/img2txt`](https://github.com/OdincovMD/img2txt).

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `DESCRIPTION_ENABLED` | `false` | Включает интеграцию с внешним сервисом генерации описаний |
| `DESCRIPTION_SERVICE_URL` | `http://description_service:8000` | Внутренний URL description service |
| `DESCRIPTION_SERVICE_API_TOKEN` | пусто | Токен, который backend/celery отправляет во внешний description service |
| `DESCRIPTION_CALLBACK_API_TOKEN` | пусто | Токен, который backend ожидает на callback-маршруте `/internal/description-results/{job_id}` |

Если `DESCRIPTION_ENABLED=false`, основная классификация продолжит работать без внешнего сервиса описаний. Режим `features_only` при этом будет недоступен.

Для рабочей интеграции с внешним description service нужно обеспечить двустороннюю связность:

- `backend`/`celery_worker` должны иметь сетевой доступ к `DESCRIPTION_SERVICE_URL`
- внешний description service должен иметь доступ к backend-маршруту `/internal/description-results/{job_id}`
- токен `DESCRIPTION_SERVICE_API_TOKEN` используется для исходящих запросов из Skin Cancer AI
- токен `DESCRIPTION_CALLBACK_API_TOKEN` требуется во входящем callback от description service

На стороне `img2txt` должны поддерживаться маршруты:

- `POST /v1/description-jobs`
- `POST /v1/description-jobs/{job_id}/classification`

А на стороне Skin Cancer AI должен быть доступен callback:

- `POST /internal/description-results/{job_id}`

### Секреты (обязательно менять в проде)

| Переменная | Зачем | Как генерировать |
|------------|-------|------------------|
| `JWT_SECRET` | Подпись JWT-токенов для веб-сессий | `openssl rand -hex 32` |
| `IMAGE_ACCESS_SIGNING_SECRET` | HMAC-подпись токенов для `/history/image` | `openssl rand -hex 32` |
| `DB_PASS` | Пароль PostgreSQL | задайте надёжный |
| `MINIO_USER` / `MINIO_PASSWORD` | Доступ к объектному хранилищу | задайте |
| `SMTP_PASSWORD` | Пароль SMTP для отправки писем верификации | из вашего провайдера |

Не коммитьте `.env` в репозиторий. Файл уже в `.gitignore`.

### JWT

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JWT_ALGORITHM` | `HS256` | Алгоритм подписи |
| `JWT_EXPIRE_MINUTES` | `1440` (24 ч) | Время жизни токена |

### Email и верификация

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `SMTP_HOST` | пусто | Хост SMTP-сервера; если пусто — отправка отключена |
| `SMTP_PORT` | `587` | Порт |
| `SMTP_USE_TLS` | `true` | STARTTLS |
| `SMTP_USE_SSL` | `false` | SSL-обёртка |
| `SMTP_USER` | — | Логин |
| `SMTP_PASSWORD` | — | Пароль |
| `MAIL_FROM` | — | Адрес отправителя |
| `MAIL_FROM_NAME` | `Skin Cancer AI` | Имя отправителя |
| `EMAIL_VERIFICATION_TOKEN_TTL_HOURS` | `24` | Срок жизни ссылки верификации |
| `VERIFICATION_EMAIL_RESEND_COOLDOWN_SEC` | `120` | Минимальный интервал между повторными письмами |
| `PASSWORD_RESET_TOKEN_TTL_HOURS` | `1` | Срок жизни ссылки на сброс пароля |
| `PASSWORD_RESET_COOLDOWN_SEC` | `120` | Минимальный интервал между письмами сброса пароля |
| `FRONTEND_PUBLIC_URL` | `http://localhost:90` | URL, который видит пользователь в браузере (для ссылки в письме) |

### Rate Limiting (API v1)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `API_V1_RATE_LIMIT_PER_MINUTE` | `5` | Максимум запросов на пользователя за скользящее окно 60 с |
| `IMAGE_ACCESS_TOKEN_TTL_SEC` | `3600` | Срок жизни HMAC-токена изображения |

## Порты

| Порт хоста | Сервис | Назначение |
|------------|--------|------------|
| **90** | NGINX | Точка входа: `/` — фронтенд, `/backend/` — API |
| **7178** | PostgreSQL | Внешний доступ к БД (для отладки) |
| **1112** | MinIO Console | Веб-интерфейс хранилища |

Внутренние порты (не проброшены наружу по умолчанию): backend `:9000`, frontend `:3000`, ml `:8000`, redis `:6379`, minio S3 `:9000`.

## Проверка работоспособности

После запуска убедитесь, что всё работает:

```bash
# Все контейнеры healthy
docker compose ps

# Backend отвечает
curl -s http://localhost:90/backend/health
# → {"status":"ok"}

# ML-сервис жив (внутренний порт, через docker exec)
docker compose exec ml curl -s http://127.0.0.1:8000/health
# → {"status":"ok"}
```

Если `backend` показывает `health: starting` дольше минуты, проверьте, что PostgreSQL и Redis уже `healthy`:

```bash
docker compose logs backend --tail 50
```

## Обновление

```bash
git pull
docker compose up --build -d
```

Миграция схемы БД выполняется автоматически при старте (`init_db`). Если добавляются новые таблицы, они создаются; существующие данные не затрагиваются.

## Бэкапы

### PostgreSQL

```bash
docker compose exec postgres pg_dump -U postgres postgres > backup_$(date +%F).sql
```

### MinIO

Данные хранятся в Docker volume `skin_minio_data`. Для бэкапа:

```bash
docker run --rm -v skin_minio_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/minio_backup_$(date +%F).tar.gz /data
```

## Логи

```bash
# Все сервисы
docker compose logs -f

# Конкретный сервис
docker compose logs backend --tail 100
docker compose logs celery_worker --tail 100
docker compose logs ml --tail 100
```

ML-сервис также может писать логи в директорию `log/` (монтируется из хоста).

## Типичные проблемы

### Backend не стартует: `connection refused` к PostgreSQL

PostgreSQL ещё не готов. Compose обрабатывает это через `healthcheck` + `depends_on`, но если `start_period` слишком мал для вашего окружения — увеличьте его в `docker-compose.yml`.

### ML-сервис долго стартует

Это нормально. Загрузка весов моделей (YOLO, UNet, классификаторы) занимает 1–5 минут в зависимости от железа. `start_period: 600s` в Compose даёт на это 10 минут.

### MinIO: `Access Denied`

Проверьте, что `MINIO_USER` и `MINIO_PASSWORD` в `.env` совпадают с теми, которые были при первом создании volume. Если меняли креды после первого запуска — удалите volume и пересоздайте:

```bash
docker compose down -v  # ВНИМАНИЕ: удалит все данные MinIO
docker compose up --build -d
```

### Письма верификации не приходят

1. Проверьте, что `SMTP_HOST` не пустой.
2. Проверьте логи: `docker compose logs backend | grep -i smtp`.
3. Убедитесь, что `FRONTEND_PUBLIC_URL` указывает на адрес, доступный из браузера пользователя (не `localhost`, если деплой удалённый).

### Письма сброса пароля не приходят

1. Проверьте те же SMTP-параметры, что и для писем верификации.
2. Убедитесь, что `FRONTEND_PUBLIC_URL` ведёт на реальный адрес фронтенда.
3. Если запросы на сброс повторяются слишком часто, дождитесь окончания `PASSWORD_RESET_COOLDOWN_SEC`.

### `429 Too Many Requests` на API v1

Лимит задан переменной `API_V1_RATE_LIMIT_PER_MINUTE`. Увеличьте, если нужно, и перезапустите backend:

```bash
docker compose restart backend celery_worker
```

### Загрузка файла: `413 Request Entity Too Large`

NGINX ограничивает размер тела запроса. В `nginx/nginx.conf` установлено `client_max_body_size 20M`. Если нужно больше — измените и перезапустите nginx:

```bash
docker compose restart nginx
```
