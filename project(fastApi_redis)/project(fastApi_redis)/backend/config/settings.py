from pydantic import BaseSettings, AnyHttpUrl
from typing import List

class Settings(BaseSettings):
    SECRET_KEY: str = "dev-only"
    DEBUG: bool = False

    REDIS_URL: str = "redis://redis:6379/0"

    CORS_ALLOWED_ORIGINS: List[AnyHttpUrl] = []

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
