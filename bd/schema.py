from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserCreateRequest(BaseModel):
    username: str
    password_hash: str
    role: str

    class Config:
        orm_mode = True
