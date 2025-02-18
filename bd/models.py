from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

current_time = datetime.now()  # Текущее время

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)
    created_at = Column(TIMESTAMP, default=current_time)
