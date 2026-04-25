# HTTP API — Skin Cancer AI (Backend)

Документация соответствует текущему коду FastAPI в `backend/app/`. OpenAPI/Swagger в приложении отключены (`docs_url=None`).

## Базовый URL

| Окружение | Пример |
|-----------|--------|
| Docker Compose + NGINX (порт 90) | `http://localhost:90/backend` |
| Прямой доступ к контейнеру backend (без префикса) | `http://backend:9000` |

Все пути ниже указаны **относительно базового URL** (в проде обычно с префиксом `/backend`).

Формат: `{origin}{BACKEND_URL}{path}` — например `POST /backend/signin`.

---

## Аутентификация

### Вариант A — браузер и фронтенд (JWT)

Заголовок на защищённых маршрутах:

```http
Authorization: Bearer <access_token>
```

Токен выдаётся в ответах `POST /signin` и `POST /signup` (поле `access_token`).

### Вариант B — программный доступ (API v1)

Префикс маршрутов: **`/api/v1`**. Заголовок:

```http
X-API-Key: <сырой_токен>
```

Токен выпускается в личном кабинете (`POST /me/api-token` и др.) только при **подтверждённом email**. В БД хранится хэш; в ответ при создании/ротации возвращается полный секрет один раз.

**Лимит:** настраивается переменной `API_V1_RATE_LIMIT_PER_MINUTE` (по умолчанию 5 запросов на пользователя за скользящее окно 60 с, Redis). При превышении — **429**.

---

## Общие сведения

- **Формат тел:** JSON с `Content-Type: application/json`, если не указано иное.
- **Ошибки FastAPI** часто приходят как `{"detail": "..." }` или `{"detail": [ ... ] }` (валидация).
- **CORS** в приложении разрешён для всех источников (`allow_origins=["*"]`) — в проде обычно сужают на уровне NGINX/политики.

---

## Health

### `GET /health`

Проверка Redis (`PING`) и PostgreSQL (`SELECT 1`).

| Код | Тело |
|-----|------|
| 200 | `{"status":"ok"}` |
| 503 | `{"detail":"unavailable"}` |

Аутентификация не требуется.

---

## Регистрация и вход

### `POST /signup`

Тело (JSON):

| Поле | Тип | Описание |
|------|-----|----------|
| `firstName` | string | Имя |
| `lastName` | string | Фамилия |
| `login` | string | Логин |
| `email` | string | Email |
| `password` | string | Пароль |

Успех (пример полей): `userData`, `error: null`, `access_token`, `token_type: "bearer"`, `requires_email_verification: true`, `verification_resend_after_seconds`. При ошибке валидации/БД — `userData` с `null`-полями и строка `error`.

### `POST /signin`

Тело:

| Поле | Тип |
|------|-----|
| `login` | string |
| `password` | string |

Успех: `userData` (в т.ч. `email_verified`), `access_token`, `token_type`, `verification_resend_after_seconds`, `error: null`.

---

## Профиль и аккаунт (JWT)

Все маршруты ниже требуют **`Authorization: Bearer`**, кроме явно отмеченных.

### `GET /me`

Профиль текущего пользователя: `{"userData": {...}, "error": null}` или ошибка.

### `PATCH /me/profile`

Тело: `UpdateProfileBody` — хотя бы одно из полей:

| Поле | Тип | Обязательно |
|------|-----|-------------|
| `firstName` | string \| null | нет |
| `lastName` | string \| null | нет |

### `GET /me/avatar`

Возвращает бинарное тело изображения аватара или **404**.

### `POST /me/avatar`

`multipart/form-data`, поле **`file`**. Допустимые типы: JPEG, PNG, WebP; максимум **5 МБ**. Ответ: `{"error": null}` или `{"error": "..."}`.

### `POST /change-password`

Тело:

| Поле | Тип |
|------|-----|
| `current_password` | string |
| `new_password` | string (≥ 8 символов, латиница и цифры) |

### `POST /resend-verification-email`

Повторная отправка письма верификации (ограничение по времени на стороне сервера). Ответ зависит от состояния пользователя и cooldown.

### `GET /verify-email` (без JWT)

Query: **`token`** — одноразовый токен из письма.

Ответ: `{"ok": true, "error": null}` или `{"ok": false, "error": "..."}`.

---

## API-ключ (JWT + подтверждённый email)

### `GET /me/api-token`

Статус ключа: `has_token`, `created_at`, `display_label`.

Ошибки: **404** (пользователь не найден), **403** (email не подтверждён).

### `POST /me/api-token`

Выпуск нового ключа. **409**, если ключ уже есть (нужна ротация или отзыв).

Успех: `{"token": "<сырой_scai_...>", "created_at": "..."}`.

### `POST /me/api-token/rotate`

Новый ключ взамен старого. **409**, если ключа не было.

### `DELETE /me/api-token`

Отзыв ключа. Ответ: `{"ok": true}`.

---

## Классификация (JWT + подтверждённый email)

Идентификация пользователя — из JWT (зависимость `require_verified_email_user_id`). Поле `user_id` в query **не используется**.

Одновременно может быть только одно активное задание в статусах `pending` / `processing`. Повторная загрузка — **429**.

### `POST /uploadfile`

`multipart/form-data`, поле **`file`**.

Успех: `{"job_id": <int>, "status": "pending"}`. Классификация выполняется в фоне (Celery).

### `GET /classification-jobs/active`

Текущее активное задание пользователя.

| Код | Описание |
|-----|----------|
| 204 | Нет активного задания |
| 200 | JSON: `job_id`, `status`, `file_name`, при необходимости `image_token` и др. |

Задание считается активным, пока:

- классификация находится в `pending` / `processing`
- или описание ещё не перешло в terminal state (`completed` / `error`)

### `GET /classification-jobs/{job_id}`

Статус и результат по `job_id`. **404**, если задание не найдено или не принадлежит пользователю.

В ответе при успехе:

- `status` — статус классификации
- `result` — JSON классификации после завершения
- `image_token` — токен изображения (если настроен секрет подписи)
- `description_status` — статус генерации описания
- `description` — готовое клиническое описание или `null`
- `description_error` — ошибка description pipeline или `null`
- `important_labels` — массив значимых признаков

### `POST /gethistory`

Тело: пустой объект **`{}`** (JSON).

Массив последних записей истории; у строк с файлами в MinIO может быть поле **`image_token`** для превью.

Каждая запись истории дополнительно может содержать:

- `description_status`
- `description`
- `description_error`
- `important_labels`

### `GET /history/image`

Query: **`token`** — HMAC-токен из поля `image_token` (не передавать `user_id` / `file_name` отдельно).

Успех: поток байтов изображения с подходящим `Content-Type`. Ошибки: **403** (невалидный/просроченный токен), **404**, **502/503** при проблемах с хранилищем или конфигурацией.

---

## API v1 — зеркало классификации (`X-API-Key`)

Базовый путь: **`/api/v1`**.

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/uploadfile` | Как `POST /uploadfile` |
| `GET` | `/api/v1/classification-jobs/active` | Как `GET /classification-jobs/active` |
| `GET` | `/api/v1/classification-jobs/{job_id}` | Как `GET /classification-jobs/{job_id}` |
| `POST` | `/api/v1/gethistory` | Как `POST /gethistory` |
| `GET` | `/api/v1/history/image?token=...` | Как `GET /history/image`, плюс **дополнительная** проверка rate limit по пользователю из токена |

Аутентификация: **`X-API-Key`**. При отсутствии или неверном ключе — **401**.

---

## Примеры `curl`

Замените `BASE` и токены на свои.

```bash
BASE="http://localhost:90/backend"

# Вход
curl -sS -X POST "$BASE/signin" \
  -H "Content-Type: application/json" \
  -d '{"login":"user","password":"secret"}'

# Профиль (подставьте JWT)
curl -sS "$BASE/me" -H "Authorization: Bearer YOUR_JWT"

# Загрузка на классификацию
curl -sS -X POST "$BASE/uploadfile" \
  -H "Authorization: Bearer YOUR_JWT" \
  -F "file=@./image.jpg"

# Опрос задания
curl -sS "$BASE/classification-jobs/1" \
  -H "Authorization: Bearer YOUR_JWT"

# API v1
curl -sS -X POST "$BASE/api/v1/uploadfile" \
  -H "X-API-Key: scai_..." \
  -F "file=@./image.jpg"
```

---

## Переменные окружения (связанные с API)

| Переменная | Назначение |
|------------|------------|
| `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` | Выдача и проверка JWT |
| `IMAGE_ACCESS_SIGNING_SECRET`, `IMAGE_ACCESS_TOKEN_TTL_SEC` | Подпись `image_token` для `/history/image` |
| `API_V1_RATE_LIMIT_PER_MINUTE` | Лимит для `/api/v1/*` |
| `FRONTEND_PUBLIC_URL`, SMTP-* | Письма верификации (не HTTP API, но влияют на регистрацию) |

Подробнее см. `.env.example` в корне репозитория.
