from sqlalchemy.orm import Session
import models
import schema
# from passlib.context import CryptContext

# Инициализируем контекст для хеширования паролей
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Функция для хеширования пароля
# def hash_password(password: str):
#     return pwd_context.hash(password)


# Функция для создания нового пользователя
def create_user(db: Session, user: schema.UserCreateRequest):
    # hashed_password = hash_password(user.password)

    db_user = models.User(
        username=user.username,
        password_hash=user.password_hash,
        role=user.role,
    )

    db.add(db_user)
    db.commit()  # Фиксируем изменения в базе данных
    db.refresh(db_user)  # Обновляем данные пользователя (получаем его с ID)

    return db_user


# Функция для получения пользователя по имени
def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()
