from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    lastName = Column(String, index=True)
    firstName = Column(String, index=True)
    login = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime(timezone=True), default=func.now())

    classification_results = relationship("ClassificationResults", back_populates="user")

class File(Base):
    __tablename__ = "files"

    file_id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    bucket_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)

    classification_results = relationship("ClassificationResults", back_populates="file")

class ClassificationResults(Base):
    __tablename__ = "classification_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    file_id = Column(Integer, ForeignKey('files.file_id'), nullable=False)

    request_date = Column(DateTime(timezone=True), default=func.now())
    status = Column(String, default="completed")
    result = Column(Text, nullable=True)  # Результат классификации

    user = relationship("User", back_populates="classification_results")
    file = relationship("File", back_populates="classification_results")

