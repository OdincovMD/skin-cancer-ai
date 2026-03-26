# Melanoma Detection using Kittler's Method

[www.skin-cancer-ai.ru](https://skin-cancer-ai.ru) — Visit the project website for more information and to access the full system.

## Project Description

This project is designed for the automated recognition of melanoma in dermatoscopic images using Kittler's method. [Kittler's method](https://www.researchgate.net/publication/224895107_Dermatoscopy_of_unpigmented_lesions_of_the_skin_A_new_classification_of_vessel_morphology_based_on_pattern_analysis) is based on analyzing the hierarchical structure of image features, allowing for the construction of a detailed decision tree.

The analysis process begins with image preprocessing, including normalization, noise removal, and segmentation. The models then analyze key morphological features such as the presence and distribution of globules, dots, lines, reticular structures, and other patterns characteristic of melanoma. Based on the detected characteristics, a sequence of decisions is made, represented in the form of a tree. This tree allows the user not only to see the final classification result but also to understand which features were crucial in the diagnostic process.

The chosen approach ensures interpretability of the results, which is especially important in medical applications where understanding why a model arrived at a particular conclusion is essential. Thus, this project not only automates the image analysis process but also makes it transparent and explainable for specialists.

## Running with Docker

1. Copy `.env.example` to `.env` and set at least:
   - **PostgreSQL**: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME` (inside Docker Compose, `DB_HOST` is the service name `postgres`).
   - **Redis**: `REDIS_HOST=redis`, `REDIS_PORT`, `REDIS_DB` (used by Celery and health checks).
   - **MinIO**: `MINIO_USER`, `MINIO_PASSWORD`, `MINIO_URL` (e.g. `http://minio:9000` in Compose).
   - **App**: `BACKEND_URL` (e.g. `/backend` behind NGINX), `ML_URL` (e.g. `http://ml:8000`), optional `HOST` for NGINX.

2. Start the stack:

   ```bash
   docker compose up --build
   ```

   The Compose file sets **`name: skin`**, so containers are prefixed with the project name (e.g. `skin-backend-1`). Service hostnames on the internal network remain short: `postgres`, `redis`, `minio`, `ml`, `backend`, `frontend`, etc.

3. **Entry point**: NGINX listens on port **90** (see `docker-compose.yml`). Frontend is proxied at `/`, API at `/backend/`.

4. **Health**: Backend exposes **`GET /health`** (Redis + PostgreSQL). ML service exposes **`GET /health`**. Swagger/OpenAPI is disabled on the API in production-oriented setup.

## Project Structure

### 1. Backend

Implemented in Python and includes:

- **FastAPI** — HTTP API (routers: `auth`, `classification`, `health` under `backend/app/routers/`).
- **PostgreSQL** via **SQLAlchemy 2.x async** and **asyncpg** (no synchronous Postgres driver in the app path).
- **Redis** — Celery broker and result backend; also pinged in `/health`.
- **Celery** — background worker (`celery_worker` service) runs classification: loads the image from **MinIO**, calls the **ML** service, writes status and JSON result to PostgreSQL.
- **MinIO** — S3-compatible storage; boto3 is configured with **path-style** addressing and **SigV4** for compatibility.

#### Classification flow (API)

1. **`POST /uploadfile`** — Saves the file locally, records metadata, uploads to MinIO under `uploads/<filename>`, creates a `classification_results` row with status `pending`, enqueues a Celery task. Returns **`job_id`** and **`pending`** (does not block on ML).
2. **`GET /classification-jobs/active?user_id=`** — Returns the current `pending`/`processing` job for that user (or 204), so the UI can resume after reload.
3. **`GET /classification-jobs/{job_id}?user_id=`** — Polls job status and parsed `result` JSON.
4. **`POST /gethistory`** — Last N classification rows for the user (includes **`bucket_name`** for MinIO).
5. **`GET /history/image?token=`** — Streams the image from MinIO using a short-lived HMAC token (from `image_token` on history rows or job polling). Query params `user_id` / `file_name` are not accepted.

Only one active (`pending`/`processing`) classification per user is allowed; additional uploads receive **429**.

#### Database tables

1. **`users`** — Registration and login.
2. **`files`** — Uploaded file name, bucket, logical path.
3. **`classification_results`** — Links user + file, `status`, `result` (JSON text), timestamps.

Schema is created on first startup when the database is empty (`init_db`).

### 2. Frontend

**React** (Vite), **React Router**, **Redux** for session.

- **Home** — Upload, optional magnify, async job polling, decision tree when `final_class` is present; can resume an active job after page reload via the active-job API.
- **Profile** — Last classifications from history; supports **“Show image from storage”** for rows backed by MinIO (`/history/image`).
- **Sign in / Sign up** — Existing flows.

### 3. ML service

- **`ml/main.py`** — FastAPI app: **`POST /uploadfile`**, **`GET /health`**, `ThreadPoolExecutor` for CPU-bound inference.
- **`ml/mask_builder.py`** — Segmentation (YOLO + UNet stack as in the project).
- **`ml/weight/`** — Model weights (mounted into the container).
- Optional **model warmup** on process start to reduce first-request latency.

The decision tree on the frontend is aligned with the hierarchical labels in `frontend/src/imports/TREE.js`.

#### Parallel processing and logging

- The ML module can run several concurrent in-process workers (see `ThreadPoolExecutor` in `main.py`).
- Logs may be written under **`log/`** (mounted in Compose where configured).

### 4. NGINX

Reverse proxy: `/` → frontend, `/backend/` → FastAPI. Large uploads: `client_max_body_size` set for image posts.

## Acknowledgements

We would like to extend our heartfelt thanks to all the developers who have contributed to this project. Their expertise, creativity, and dedication have been essential in bringing this project to life. We also want to give special recognition to **[Кегелик Николай Александрович](https://github.com/Horokami)**, who played a key role in setting up and refining the frontend, ensuring a smooth and intuitive user experience.

## Useful Links

- [UNET - Skin Cancer Segmentation](https://www.kaggle.com/code/mihailodin1/skin-cancer-segmentation-unet) — Notebook for training the UNET model for skin cancer segmentation.
- [Color Classification](https://www.kaggle.com/code/mihailodin1/skin-cancer-color) — Notebook for training the color classification model used in our project.
- [One vs Several - Melanoma Classification](https://www.kaggle.com/code/mihailodin1/one-many-mell-cl) — Notebook for the "One vs Several" melanoma classification model, as part of the decision tree.

## Acknowledgements to the Tech Stack

We would also like to thank the following technologies and frameworks that helped make this project a success:

- **YOLO** for their powerful image segmentation capabilities.
- **PyTorch** for providing flexible and efficient machine learning environments.
- **Scikit-learn** for implementing various classification algorithms.
- **FastAPI** for enabling smooth deployment and API management.
- **Docker** and **Docker Compose** for streamlining development and deployment.
- **Celery** and **Redis** for asynchronous task processing.
- **MinIO** for S3-compatible object storage.

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code, provided you include the original copyright and license notice in any copies of the software. See the [LICENSE](LICENSE.txt) file for more details.
