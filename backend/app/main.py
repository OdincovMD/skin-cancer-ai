from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import os
import requests

from src.database import UserSignUp, Credentials, GetHistoryRequest
from minio_client import get_minio_client, is_file_in_minio, upload_file_to_minio, create_bucket_if_not_exists
from src.queries.orm import SyncOrm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET_NAME = "bucket"

# Подключение
s3_client = get_minio_client()
SyncOrm.create_tables()

@app.post("/signup")
async def signup(user_data: UserSignUp):
    try:
        result = SyncOrm.register_user(
            firstName=user_data.firstName,
            lastName=user_data.lastName,
            login=user_data.login,
            email=user_data.email,
            password=user_data.password
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                },
                "error": result,
            }
        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
            },
            "error": None,
        }
    
    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
            },
            "error": f"Ошибка при регистрации пользователя: {str(e)}",
        }
    
@app.post("/signin")
async def signin_user(credentials: Credentials):
    try:
        result = SyncOrm.signin_user(
            login=credentials.login,
            password=credentials.password
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                },
                "error": result,
            }

        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
            },
            "error": None,
        }

    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
            },
            "error": f"Ошибка при входе: {str(e)}",
        }

@app.post("/uploadfile")
async def handle_upload(user_id: int = Form(), file: UploadFile = Form()):

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file_name = file.filename
    file_content = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    create_bucket_if_not_exists(s3_client, BUCKET_NAME)
    SyncOrm.insert_file_record(file_name=file_name, bucket_name=BUCKET_NAME)
    # Проверка наличия файла в MinIO
    if not is_file_in_minio(s3_client, BUCKET_NAME, file_path):
        upload_file_to_minio(s3_client, BUCKET_NAME, file_path)
    else:
        # Загружаем файл в MinIO
        upload_file_to_minio(s3_client, BUCKET_NAME, file_path)

    try:
        file_id = SyncOrm.get_file_id_by_name(file_name=file_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    url = "http://ml:8000/uploadfile"
    files = {"file": (file_name, io.BytesIO(file_content), file.content_type)}
    try:
        response = requests.post(url, files=files)
        result = response.json()
        status = "completed"
    except requests.exceptions.RequestException as e:
        result = None
        status = "error"
        raise HTTPException(status_code=500, detail=f"Ошибка при отправке файла в ML-сервис: {str(e)}")
    
    SyncOrm.create_classification_request(user_id=user_id, file_id=file_id, status=status, result=json.dumps(result, ensure_ascii=True) if result else None)
    return result

@app.post("/gethistory")
async def get_history(request: GetHistoryRequest):
    print(request)
    try:
        history = SyncOrm.get_classification_requests(request.user_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении истории: {str(e)}")