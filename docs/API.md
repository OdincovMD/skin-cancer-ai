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
| `email` | string | Email |
| `password` | string | Пароль |

Успех (пример полей): `userData`, `error: null`, `access_token`, `token_type: "bearer"`, `requires_email_verification: true`, `verification_resend_after_seconds`. Имя и фамилию пользователь может заполнить позже в профиле. При ошибке валидации/БД — `userData` с `null`-полями и строка `error`.

### `POST /signin`

Тело:

| Поле | Тип |
|------|-----|
| `email` | string |
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

Смена пароля в пользовательском интерфейсе выполняется через сценарий сброса пароля по email:

1. `POST /forgot-password`
2. переход по ссылке из письма
3. `POST /reset-password`

### `POST /resend-verification-email`

Повторная отправка письма верификации (ограничение по времени на стороне сервера). Ответ зависит от состояния пользователя и cooldown.

### `GET /verify-email` (без JWT)

Query: **`token`** — одноразовый токен из письма.

Ответ: `{"ok": true, "error": null}` или `{"ok": false, "error": "..."}`.

### `POST /forgot-password` (без JWT)

Тело:

| Поле | Тип |
|------|-----|
| `email` | string |

Маршрут инициирует сброс пароля по email. Если пользователь существует и для него разрешена отправка нового письма, на почту приходит одноразовая ссылка для сброса.

Успех: `{"error": null}`.

Возможные ошибки приходят в поле `error`, например при слишком частом повторном запросе или если отправка письма недоступна.

### `POST /reset-password` (без JWT)

Тело:

| Поле | Тип |
|------|-----|
| `token` | string |
| `new_password` | string (≥ 8 символов) |

Успех: `{"ok": true, "error": null}`.

Ошибка: `{"ok": false, "error": "..."}` — например, если токен недействителен, истёк или новый пароль не прошёл валидацию.

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

`multipart/form-data`.

Поля формы:

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `file` | binary | да | Изображение для анализа |
| `features_only` | boolean | нет | Если `true`, выполняется только основная классификация без внешнего текстового description pipeline |

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
- `bucketed_labels` — группированные признаки/ярлыки
- `description_result` — сырой JSON-ответ description pipeline или `null`
- `features_only` — `true`, если задание выполнялось без внешнего текстового description pipeline

### `POST /gethistory`

Тело: пустой объект **`{}`** (JSON).

Массив последних записей истории; у строк с файлами в MinIO может быть поле **`image_token`** для превью.

Каждая запись истории дополнительно может содержать:

- `description_status`
- `description`
- `description_error`
- `important_labels`
- `bucketed_labels`
- `description_result`
- `features_only`

### `GET /history/image`

Query: **`token`** — HMAC-токен из поля `image_token` (не передавать `user_id` / `file_name` отдельно).

Успех: поток байтов изображения с подходящим `Content-Type`. Ошибки: **403** (невалидный/просроченный токен), **404**, **502/503** при проблемах с хранилищем или конфигурацией.

---

## API v1 — зеркало классификации (`X-API-Key`)

Базовый путь: **`/api/v1`**.

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/uploadfile` | Как `POST /uploadfile`, плюс опциональный `features_only=true` в `multipart/form-data` |
| `GET` | `/api/v1/classification-jobs/active` | Как `GET /classification-jobs/active` |
| `GET` | `/api/v1/classification-jobs/{job_id}` | Как `GET /classification-jobs/{job_id}` |
| `POST` | `/api/v1/gethistory` | Как `POST /gethistory` |
| `GET` | `/api/v1/history/image?token=...` | Как `GET /history/image`, плюс **дополнительная** проверка rate limit по пользователю из токена |

Аутентификация: **`X-API-Key`**. При отсутствии или неверном ключе — **401**.

---

## Внешняя интеграция

### Интеграция клиентских систем с Skin Cancer AI

Для внешних скриптов, backend-сервисов и partner-интеграций используйте только публичный API v1:

1. Пользователь подтверждает email и выпускает API-ключ в личном кабинете.
2. Внешняя система отправляет изображение на `POST /api/v1/uploadfile`.
3. Система получает `job_id` и периодически опрашивает:
   - `GET /api/v1/classification-jobs/{job_id}` для статуса и результата
   - или `GET /api/v1/classification-jobs/active`, если нужно восстановить незавершённое задание
4. При необходимости можно получить историю через `POST /api/v1/gethistory` и изображение по `image_token` через `GET /api/v1/history/image`.

Для внешней интеграции не нужны JWT-cookie или браузерная сессия. Достаточно заголовка:

```http
X-API-Key: scai_...
```

Рекомендации для интегратора:

- Опрашивайте статус не чаще одного раза в 2 секунды, чтобы не упираться в rate limit.
- Сохраняйте `job_id` и `image_token` на своей стороне, если хотите позже восстановить состояние или показать превью.
- Обрабатывайте промежуточные состояния `pending` и `processing`.
- Для `features_only=true` ожидайте обычное поле `result` с итогом классификации, но без внешнего текстового описания и связанных полей description pipeline.

### Интеграция внешнего description service (`img2txt`)

Текущая интеграция в проекте рассчитана на внешний сервис описания изображений
[`OdincovMD/img2txt`](https://github.com/OdincovMD/img2txt).

Если включён `DESCRIPTION_ENABLED`, backend и Celery используют внешний сервис генерации описаний по двухшаговой схеме:

1. Celery регистрирует задание во внешнем сервисе через `POST {DESCRIPTION_SERVICE_URL}/v1/description-jobs`.
2. После готовности основной классификации Celery отправляет туда результат через `POST {DESCRIPTION_SERVICE_URL}/v1/description-jobs/{job_id}/classification`.
3. Внешний сервис возвращает итоговый статус и признаки callback-ом в `POST /internal/description-results/{job_id}`.

Ожидаемые заголовки интеграции:

- исходящие запросы из backend/celery во внешний сервис: `X-Service-Token: <DESCRIPTION_SERVICE_API_TOKEN>`
- входящий callback во backend: `X-Service-Token: <DESCRIPTION_CALLBACK_API_TOKEN>`

Эта интеграция не предназначена для публичных клиентов. Это сервис-сервисный контракт внутри инфраструктуры.

Со стороны Skin Cancer AI контракт выглядит так:

- регистрация задания: `POST {DESCRIPTION_SERVICE_URL}/v1/description-jobs`
- передача результата основной классификации: `POST {DESCRIPTION_SERVICE_URL}/v1/description-jobs/{job_id}/classification`
- приём callback от `img2txt`: `POST /internal/description-results/{job_id}`

Если API `img2txt` изменится, нужно синхронно обновить:

- `backend/app/services/description_service.py`
- `backend/app/routers/internal.py`
- этот раздел документации

---

## Внутренний API description-service

Этот маршрут предназначен не для браузера и не для публичных интеграций, а для callback от внешнего сервиса генерации описаний.

### `POST /internal/description-results/{job_id}`

Заголовок:

```http
X-Service-Token: <service_token>
```

Если `DESCRIPTION_CALLBACK_API_TOKEN` не настроен, сервис вернёт **503**. При неверном или отсутствующем токене — **401**.

Path-параметр:

- `job_id` — идентификатор задания во внешнем description service

Тело (JSON) — основные поля:

| Поле | Тип | Описание |
|------|-----|----------|
| `status` | string | Текущий статус description pipeline |
| `description` | string \| null | Сформированное описание |
| `important_labels` | string[] | Ключевые признаки |
| `all_labels` | string[] | Полный список признаков |
| `bucketed_labels` | string[] | Сгруппированные признаки |
| `features_only` | boolean | Был ли запрос на только признаки |
| `error` | string \| null | Ошибка pipeline |

Успех: `{"ok": true}`.

Ошибки:

- **401** — неверный `X-Service-Token`
- **404** — связанное задание description service не найдено
- **503** — callback-токен не сконфигурирован

---

## Примеры `curl`

Замените `BASE` и токены на свои.

```bash
BASE="http://localhost:90/backend"

# Вход
curl -sS -X POST "$BASE/signin" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"secret"}'

# Профиль (подставьте JWT)
curl -sS "$BASE/me" -H "Authorization: Bearer YOUR_JWT"

# Загрузка на классификацию
curl -sS -X POST "$BASE/uploadfile" \
  -H "Authorization: Bearer YOUR_JWT" \
  -F "file=@./image.jpg"

# Загрузка без внешнего текстового описания
curl -sS -X POST "$BASE/uploadfile" \
  -H "Authorization: Bearer YOUR_JWT" \
  -F "file=@./image.jpg" \
  -F "features_only=true"

# Опрос задания
curl -sS "$BASE/classification-jobs/1" \
  -H "Authorization: Bearer YOUR_JWT"

# API v1
curl -sS -X POST "$BASE/api/v1/uploadfile" \
  -H "X-API-Key: scai_..." \
  -F "file=@./image.jpg"

# API v1 без внешнего текстового описания
curl -sS -X POST "$BASE/api/v1/uploadfile" \
  -H "X-API-Key: scai_..." \
  -F "file=@./image.jpg" \
  -F "features_only=true"
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
