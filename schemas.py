# schemas.py
from pydantic import BaseModel, EmailStr, Field, field_validator
import re

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8)

    @field_validator('password')
    def strong_password(cls, v):
        if not re.search(r"\d", v):
            raise ValueError('Password must contain a number')
        if not re.search(r"[!@#$%^&*]", v):
            raise ValueError('Password must contain a special char')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str