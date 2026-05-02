from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    lastName = Column(String, index=True)
    firstName = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    email_verified = Column(Boolean, nullable=False, default=False)
    # SHA-256 hex (64) от одноразового токена из письма
    email_verification_token = Column(String(128), nullable=True)
    email_verification_expires_at = Column(DateTime(timezone=True), nullable=True)
    verification_email_last_sent_at = Column(DateTime(timezone=True), nullable=True)
    # Ключ объекта в MinIO (бакет bucket), например avatars/42/abc.jpg
    profile_avatar_key = Column(String(512), nullable=True)

    # Сброс пароля (SHA-256 hex от одноразового токена из письма)
    password_reset_token = Column(String(128), nullable=True)
    password_reset_expires_at = Column(DateTime(timezone=True), nullable=True)
    password_reset_last_sent_at = Column(DateTime(timezone=True), nullable=True)

    # Долгоживущий ключ для /api/v1 (хранится SHA-256 hex от полного токена)
    api_token_hash = Column(String(64), nullable=True, unique=True, index=True)
    api_token_created_at = Column(DateTime(timezone=True), nullable=True)

    classification_results = relationship("ClassificationResults", back_populates="user")
    identities = relationship("UserIdentity", back_populates="user")


class UserIdentity(Base):
    __tablename__ = "user_identities"
    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id", name="uq_user_identities_provider_user"),
        UniqueConstraint("user_id", "provider", name="uq_user_identities_user_provider"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(32), nullable=False, index=True)
    provider_user_id = Column(String(255), nullable=False, index=True)
    provider_email = Column(String(320), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    user = relationship("User", back_populates="identities")

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
    description_job = relationship(
        "DescriptionJob",
        back_populates="classification_result",
        uselist=False,
    )


class DescriptionJob(Base):
    __tablename__ = "description_jobs"

    id = Column(Integer, primary_key=True, index=True)
    classification_result_id = Column(
        Integer,
        ForeignKey("classification_results.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    service_job_id = Column(String(255), nullable=False, unique=True, index=True)
    status = Column(String(64), nullable=False, default="pending")
    description = Column(Text, nullable=True)
    important_labels = Column(Text, nullable=True)
    bucketed_labels = Column(Text, nullable=True)
    description_result = Column(Text, nullable=True)
    features_only = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)
    callback_sent = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )

    classification_result = relationship(
        "ClassificationResults",
        back_populates="description_job",
    )
