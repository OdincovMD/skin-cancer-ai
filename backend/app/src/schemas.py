import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UserSignUp(BaseModel):
    lastName: str
    firstName: str
    login: str
    email: str
    password: str


class Credentials(BaseModel):
    login: str
    password: str
    remember_me: bool = False


class ChangePasswordBody(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def new_password_rules(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9A-Za-z]{8,}", v):
            raise ValueError(
                "Новый пароль: не менее 8 символов, только латинские буквы и цифры."
            )
        return v


class UpdateProfileBody(BaseModel):
    firstName: Optional[str] = Field(default=None, max_length=100)
    lastName: Optional[str] = Field(default=None, max_length=100)

    @field_validator("firstName", "lastName", mode="before")
    @classmethod
    def strip_optional(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @model_validator(mode="after")
    def at_least_one_field(self):
        if self.firstName is None and self.lastName is None:
            raise ValueError("Укажите имя и/или фамилию.")
        return self


class ForgotPasswordBody(BaseModel):
    email: str = Field(min_length=1, max_length=320)


class ResetPasswordBody(BaseModel):
    token: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def new_password_rules(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9A-Za-z]{8,}", v):
            raise ValueError(
                "Пароль: не менее 8 символов, только латинские буквы и цифры."
            )
        return v


class DescriptionCallbackBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = Field(min_length=1)
    description: Optional[str] = None
    important_labels: list[str] = Field(default_factory=list)
    all_labels: list[str] = Field(default_factory=list)
    bucketed_labels: list[str] = Field(default_factory=list)
    features_only: bool = False
    error: Optional[str] = None
