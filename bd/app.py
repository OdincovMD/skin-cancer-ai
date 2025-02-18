from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
import os
from sqlalchemy.orm import Session
import crud
import models
import schema
from database import engine, get_db

# Конфигурация базы данных
# DATABASE_URL = "postgresql://postgres:postgres@host.docker.internal:5432/postgres"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# metadata = MetaData()

# Инициализация FastAPI
app = FastAPI()

# Создаем все таблицы в базе данных
models.Base.metadata.create_all(bind=engine)


@app.post("/users/", response_model=schema.UserCreateRequest)
def create_user_endpoint(user: schema.UserCreateRequest, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    created_user = crud.create_user(db=db, user=user)

    return created_user


@app.get("/users/{user_id}", response_model=schema.UserCreateRequest)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return db_user
# API для загрузки файла
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"message": f"Файл {file.filename} успешно загружен", "path": file_path}
